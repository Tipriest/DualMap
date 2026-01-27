#!/usr/bin/env python3
import argparse
import json
import sys
import time
from typing import Optional, Tuple

DEFAULT_REQ_TOPIC = "/dualmap/search_request"
DEFAULT_RES_TOPIC = "/dualmap/search_result"


def _parse_bbox(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [float(x) for x in parts]
    if len(vals) not in (4, 6):
        raise ValueError(
            "bbox must have 4 or 6 floats: 'minx,miny,maxx,maxy' (or add zmin,zmax)"
        )
    return vals


def _build_request_json(
    name: str, bbox, score_th: Optional[float], continuous: bool
) -> str:
    payload = {"name": name, "bbox": bbox, "continuous": bool(continuous)}
    if score_th is not None:
        payload["score_th"] = float(score_th)
    return json.dumps(payload, ensure_ascii=False)


def run_ros2(args) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except Exception as e:
        print(
            f"[ros_search_test][ros2] Failed to import ROS2 deps: {e}",
            file=sys.stderr,
        )
        return 2

    class SearchTestNode(Node):
        def __init__(self):
            super().__init__("dualmap_search_test")
            self.req_pub = self.create_publisher(String, args.request_topic, 10)
            self.res_sub = self.create_subscription(
                String, args.result_topic, self._on_result, 10
            )

            self._got_result = False
            self._last_result = None

            # publish once shortly after start
            self.create_timer(0.2, self._publish_once)

        def _publish_once(self):
            if getattr(self, "_published", False):
                return
            self._published = True

            msg = String()
            msg.data = _build_request_json(
                args.name, args.bbox, args.score_th, args.continuous
            )
            self.req_pub.publish(msg)
            self.get_logger().info(
                f"[SearchTest][ROS2] published request -> {args.request_topic}: {msg.data}"
            )

        def _on_result(self, msg: String):
            self._got_result = True
            self._last_result = msg.data
            self.get_logger().info(
                f"[SearchTest][ROS2] received result <- {args.result_topic}: {msg.data}"
            )
            if not args.continuous:
                # stop once we get first result
                try:
                    rclpy.shutdown()
                except Exception:
                    pass

    rclpy.init()
    node = SearchTestNode()

    t0 = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.timeout > 0 and (time.time() - t0) > args.timeout:
                node.get_logger().warning(
                    "[SearchTest][ROS2] timeout waiting for result"
                )
                break
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return 0


def run_ros1(args) -> int:
    try:
        import rospy
        from std_msgs.msg import String
    except Exception as e:
        print(
            f"[ros_search_test][ros1] Failed to import ROS1 deps: {e}",
            file=sys.stderr,
        )
        return 2

    got_result = {"ok": False}

    def on_result(msg: String):
        got_result["ok"] = True
        print(
            f"[SearchTest][ROS1] received result <- {args.result_topic}: {msg.data}"
        )
        if not args.continuous:
            rospy.signal_shutdown("got_result")

    rospy.init_node("dualmap_search_test", anonymous=True)
    pub = rospy.Publisher(args.request_topic, String, queue_size=10)
    sub = rospy.Subscriber(args.result_topic, String, on_result, queue_size=10)

    payload = _build_request_json(
        args.name, args.bbox, args.score_th, args.continuous
    )
    pub.publish(String(data=payload))
    print(
        f"[SearchTest][ROS1] published request -> {args.request_topic}: {payload}"
    )

    t0 = time.time()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if args.timeout > 0 and (time.time() - t0) > args.timeout:
            print("[SearchTest][ROS1] timeout waiting for result")
            break
        rate.sleep()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="DualMap global object search test (JSON over std_msgs/String)."
    )
    ap.add_argument(
        "--ros",
        type=int,
        choices=[1, 2],
        default=2,
        help="Use ROS1(rospy) or ROS2(rclpy).",
    )
    ap.add_argument(
        "--name", type=str, required=True, help="Query object name, e.g. 'mug'."
    )
    ap.add_argument(
        "--bbox",
        type=_parse_bbox,
        required=True,
        help="BBox 'minx,miny,maxx,maxy' (or + zmin,zmax).",
    )
    ap.add_argument(
        "--score-th",
        type=float,
        default=None,
        help="Similarity threshold override.",
    )
    ap.add_argument(
        "--continuous",
        action="store_true",
        help="Keep listening/printing multiple updates.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait; <=0 means wait forever.",
    )
    ap.add_argument("--request-topic", type=str, default=DEFAULT_REQ_TOPIC)
    ap.add_argument("--result-topic", type=str, default=DEFAULT_RES_TOPIC)
    args = ap.parse_args()

    if args.ros == 2:
        return run_ros2(args)
    return run_ros1(args)


if __name__ == "__main__":
    raise SystemExit(main())
