# runner_ros1.py

import logging
import threading
import time
import json

import numpy as np
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

from applications.utils.runner_ros_base import RunnerROSBase
from dualmap.core import Dualmap
from utils.logging_helper import setup_logging


class RunnerROS1(RunnerROSBase):
    """
    ROS1-specific runner, handles topic subscriptions and data flow using rospy.
    """

    def __init__(self, cfg):
        rospy.init_node("runner_ros", anonymous=True)
        setup_logging(
            output_path=cfg.output_path, config_path=cfg.logging_config
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Runner ROS1]")
        self.logger.info(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.dualmap = Dualmap(cfg)
        super().__init__(cfg, self.dualmap)

        self.bridge = CvBridge()
        self.dataset_cfg = OmegaConf.load(cfg.ros_stream_config_path)
        self.intrinsics = self.load_intrinsics(self.dataset_cfg)
        self.extrinsics = self.load_extrinsics(self.dataset_cfg)

        # Image and Odometry Subscribers
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(
                self.dataset_cfg.ros_topics.rgb, CompressedImage
            )
            self.depth_sub = Subscriber(
                self.dataset_cfg.ros_topics.depth, CompressedImage
            )
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(self.dataset_cfg.ros_topics.rgb, Image)
            self.depth_sub = Subscriber(
                self.dataset_cfg.ros_topics.depth, Image
            )

        self.odom_sub = Subscriber(self.dataset_cfg.ros_topics.odom, Odometry)

        # Sync RGB + Depth + Odometry
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=10,
            slop=self.cfg.sync_threshold,
        )
        self.sync.registerCallback(self.synced_callback)

        # Fallback to camera_info topic if intrinsics not loaded
        rospy.Subscriber(
            self.dataset_cfg.ros_topics.camera_info,
            CameraInfo,
            self.camera_info_callback,
        )

        # ---- Search request/result (JSON over std_msgs/String) ----
        self._search_lock = threading.Lock()
        self._active_search_req = None
        self._last_published_uid = None

        # debug/throttle state for search loop
        self._search_debug_last_log_t = 0.0
        self._search_debug_last_state = None
        self._search_debug_log_period = float(
            getattr(cfg, "search_debug_log_period", 5.0)
        )

        self.search_request_topic = getattr(
            cfg, "search_request_topic", "/dualmap/search_request"
        )
        self.search_result_topic = getattr(
            cfg, "search_result_topic", "/dualmap/search_result"
        )
        self.search_rate_hz = float(getattr(cfg, "search_rate_hz", 1.0))
        self.search_sim_threshold = float(
            getattr(cfg, "search_sim_threshold", 0.30)
        )

        self._search_req_sub = rospy.Subscriber(
            self.search_request_topic,
            String,
            self._on_search_request,
            queue_size=10,
        )
        self._search_result_pub = rospy.Publisher(
            self.search_result_topic, String, queue_size=10
        )

        self._search_thread_stop = False
        self._search_thread = threading.Thread(
            target=self._search_loop_1hz, daemon=True
        )
        self._search_thread.start()
        # -----------------------------------------------------------

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synchronized RGB, Depth, and Odom messages."""
        timestamp = rgb_msg.header.stamp.to_sec()

        if self.cfg.use_compressed_topic:
            rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
            depth_img = self.decompress_image(depth_msg.data, is_depth=True)
        else:
            rgb_img = self.bridge.imgmsg_to_cv2(
                rgb_msg, desired_encoding="rgb8"
            )
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )

        depth_factor = getattr(self.dataset_cfg, "depth_factor", 1000.0)
        depth_img = self.process_depth_image(depth_img, depth_factor)

        translation = np.array(
            [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z,
            ]
        )
        quaternion = np.array(
            [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ]
        )

        pose_matrix = self.build_pose_matrix(translation, quaternion)
        self.push_data(rgb_img, depth_img, pose_matrix, timestamp)
        self.last_message_time = time.time()

    def camera_info_callback(self, msg):
        """Fallback callback to get intrinsics from CameraInfo if needed."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.K).reshape(3, 3)
            self.logger.warning("[Main] Camera intrinsics received and stored.")

    def _on_search_request(self, msg: String):
        data = self.parse_search_request_json(msg.data)
        name = data.get("name", None)
        bbox_list = data.get("bbox", None)
        if not name or bbox_list is None:
            self.logger.warning(
                "[ROS][Search] Invalid request JSON (need name + bbox)."
            )
            return

        score_th = float(data.get("score_th", self.search_sim_threshold))
        continuous = bool(data.get("continuous", False))
        bbox = self.build_o3d_aabb_from_list(bbox_list)
        if bbox is None:
            self.logger.warning("[ROS][Search] Invalid bbox format in request.")
            return

        with self._search_lock:
            self._active_search_req = {
                "name": str(name),
                "bbox": bbox,
                "score_th": score_th,
                "continuous": continuous,
            }
            self._last_published_uid = None

        self.logger.warning(
            f"[ROS][Search] Received request: name='{name}', score_th={score_th}, continuous={continuous}"
        )

    def _publish_search_result(self, payload: dict):
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self._search_result_pub.publish(out)

    def _search_loop_1hz(self):
        period = 1.0 / max(self.search_rate_hz, 1e-6)
        while (
            (not rospy.is_shutdown())
            and (not self.shutdown_requested)
            and (not self._search_thread_stop)
        ):
            req = None
            with self._search_lock:
                req = (
                    None
                    if self._active_search_req is None
                    else dict(self._active_search_req)
                )

            if req is None:
                time.sleep(0.1)
                continue

            now = time.time()

            if not self.dualmap.global_map_manager.has_global_map():
                if (
                    now - self._search_debug_last_log_t
                ) >= self._search_debug_log_period:
                    self._search_debug_last_log_t = now
                    self.logger.warning(
                        "[ROS][Search] Waiting: global map not ready yet."
                    )
                time.sleep(period)
                continue

            try:
                query_ft = self.dualmap.convert_inquiry_to_feat(req["name"])
            except Exception as e:
                if (
                    now - self._search_debug_last_log_t
                ) >= self._search_debug_log_period:
                    self._search_debug_last_log_t = now
                    self.logger.warning(
                        f"[ROS][Search] Failed to encode query '{req['name']}': {e}"
                    )
                time.sleep(period)
                continue

            expand_ratio = float(
                getattr(self.cfg, "search_bbox_expand_ratio", 0.10)
            )
            try:
                cand_num = len(
                    self.dualmap.global_map_manager.filter_global_objects_in_bbox(
                        query_bbox=req["bbox"], expand_ratio=expand_ratio
                    )
                )
            except Exception:
                cand_num = -1

            best_obj, best_score = (
                self.dualmap.global_map_manager.search_similar_object_in_bbox(
                    query_feat=query_ft,
                    query_bbox=req["bbox"],
                    sim_threshold=req["score_th"],
                    expand_ratio=expand_ratio,
                )
            )

            if best_obj is None:
                state = (
                    "MISS",
                    req["name"],
                    float(req["score_th"]),
                    float(best_score),
                    int(cand_num),
                )
                if (
                    self._search_debug_last_state != state
                    or (now - self._search_debug_last_log_t)
                    >= self._search_debug_log_period
                ):
                    self._search_debug_last_state = state
                    self._search_debug_last_log_t = now
                    self.logger.warning(
                        "[ROS][Search] No match yet: name='%s', candidates_in_bbox=%s, best_score=%.3f, score_th=%.3f",
                        req["name"],
                        "unknown" if cand_num < 0 else cand_num,
                        float(best_score),
                        float(req["score_th"]),
                    )
                time.sleep(period)
                continue

            center = (
                best_obj.bbox_2d.get_center()
                if best_obj.bbox_2d is not None
                else best_obj.pcd_2d.get_center()
            )
            uid_str = str(best_obj.uid)
            matched_class = getattr(best_obj, "class_name", None)

            if (not req["continuous"]) and (
                self._last_published_uid == uid_str
            ):
                time.sleep(period)
                continue

            self._publish_search_result(
                {
                    "name": req["name"],
                    "uid": uid_str,
                    "score": float(best_score),
                    "center": [
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                    ],
                }
            )

            self.logger.warning(
                "[ROS][Search] MATCH: query='%s' -> class='%s', uid=%s, score=%.3f (th=%.3f), center=%s, candidates_in_bbox=%s",
                req["name"],
                str(matched_class) if matched_class is not None else "unknown",
                uid_str,
                float(best_score),
                float(req["score_th"]),
                np.array(center).tolist(),
                "unknown" if cand_num < 0 else cand_num,
            )

            with self._search_lock:
                self._last_published_uid = uid_str
                if not req["continuous"]:
                    self._active_search_req = None

            time.sleep(period)

    def spin(self):
        """Main loop calling run_once() at configured ROS rate."""
        rate = rospy.Rate(self.cfg.ros_rate)
        while not rospy.is_shutdown() and not self.shutdown_requested:
            try:
                self.run_once(lambda: time.time())
            except Exception as e:
                self.logger.error(f"[RunnerROS1] Exception: {e}", exc_info=True)
            rate.sleep()


def run_ros1(cfg):
    """Launch the ROS1 runner in a background thread."""
    runner = RunnerROS1(cfg)
    runner.logger.warning(
        "[Main] ROS1 Runner started. Waiting for data stream..."
    )

    spin_thread = threading.Thread(target=runner.spin)
    spin_thread.start()

    try:
        while not rospy.is_shutdown() and not runner.shutdown_requested:
            time.sleep(0.1)
    except KeyboardInterrupt:
        runner.logger.warning("[Main] KeyboardInterrupt received.")
    finally:
        runner.shutdown_requested = True
        runner.logger.warning("[Main] Shutting down...")
        spin_thread.join(timeout=3.0)

        try:
            rospy.signal_shutdown("User requested shutdown")
        except Exception:
            pass

        runner.logger.warning("[Main] Exit complete.")

        import os

        os._exit(0)
