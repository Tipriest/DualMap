# runner_ros2.py

import logging
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

from applications.utils.ros_publisher import ROSPublisher
from applications.utils.runner_ros_base import RunnerROSBase
from dualmap.core import Dualmap
from utils.logging_helper import setup_logging


class RunnerROS2(Node, RunnerROSBase):
    """
    ROS2-specific runner. Uses rclpy and ROS2 message_filters for synchronization,
    subscription, and publishing.
    """

    def __init__(self, cfg):
        Node.__init__(self, "runner_ros")
        setup_logging(
            output_path=cfg.output_path, config_path=cfg.logging_config
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Runner ROS2]")
        self.logger.info(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.dualmap = Dualmap(cfg)
        RunnerROSBase.__init__(self, cfg, self.dualmap)

        self.bridge = CvBridge()
        self.dataset_cfg = OmegaConf.load(cfg.ros_stream_config_path)
        self.intrinsics = self.load_intrinsics(self.dataset_cfg)
        self.extrinsics = self.load_extrinsics(self.dataset_cfg)

        # Topic Subscribers
        # 分别订阅 rgb图像 depth图像 odom消息 但是每一个订阅都没有单独的回调
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(
                self, CompressedImage, self.dataset_cfg.ros_topics.rgb
            )
            self.depth_sub = Subscriber(
                self, CompressedImage, self.dataset_cfg.ros_topics.depth
            )
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(
                self, Image, self.dataset_cfg.ros_topics.rgb
            )
            self.depth_sub = Subscriber(
                self, Image, self.dataset_cfg.ros_topics.depth
            )

        self.odom_sub = Subscriber(
            self, Odometry, self.dataset_cfg.ros_topics.odom
        )

        # Sync messages
        # 三个消息的同步触发回调，三个消息的容差为0.1s
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=10,
            slop=self.cfg.sync_threshold,
        )
        self.sync.registerCallback(self.synced_callback)

        # CameraInfo callback
        self.create_subscription(
            CameraInfo,
            self.dataset_cfg.ros_topics.camera_info,
            self.camera_info_callback,
            10,
        )

        # Publisher and timer
        self.publisher = ROSPublisher(self, cfg)
        self.publish_executor = ThreadPoolExecutor(max_workers=2)

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
            getattr(cfg, "search_sim_threshold", 0.50)
        )

        self._search_req_sub = self.create_subscription(
            String, self.search_request_topic, self._on_search_request, 10
        )
        self._search_result_pub = self.create_publisher(
            String, self.search_result_topic, 10
        )

        self._search_thread_stop = False
        self._search_thread = threading.Thread(
            target=self._search_loop_1hz, daemon=True
        )
        self._search_thread.start()
        # -----------------------------------------------------------

        timer_period = 1.0 / self.cfg.ros_rate
        self.timer = self.create_timer(timer_period, self.run)

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synced RGB-D-Odom input."""
        self.received_synced_num += 1
        # self.logger.warning(
        #     f"[ROS][Sync][Msg Nums]: {self.received_synced_num}",
        # )
        timestamp = (
            rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        )

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

        # 上下翻转 RGB 与深度图
        if self.dataset_cfg.rgb_need_flip:
            rgb_img = self.flip_vertical(rgb_img)
        if self.dataset_cfg.depth_need_flip:
            depth_img = self.flip_vertical(depth_img)

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
        self.last_message_time = self.get_clock().now().nanoseconds / 1e9
        # self.logger.warning(
        #     f"update last_message_time: {self.last_message_time}",
        # )

    def camera_info_callback(self, msg):
        """Populate intrinsics from CameraInfo topic if not already loaded."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.k).reshape(3, 3)
            self.logger.warning("[Main] Camera intrinsics received and stored.")

    def run(self):
        """Periodic processing loop triggered by ROS2 timer."""
        self.run_once(lambda: self.get_clock().now().nanoseconds / 1e9)
        self.publish_executor.submit(self.publisher.publish_all, self.dualmap)

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
            rclpy.ok()
            and not self.shutdown_requested
            and not self._search_thread_stop
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

            # need global map ready
            if not self.dualmap.global_map_manager.has_global_map():
                # throttle
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
            # count candidates inside bbox (XY)
            try:
                cand_num = len(
                    self.dualmap.global_map_manager.filter_global_objects_in_bbox(
                        query_bbox=req["bbox"], expand_ratio=expand_ratio
                    )
                )
            except Exception:
                cand_num = -1  # unknown

            best_obj, best_score = (
                self.dualmap.global_map_manager.search_similar_object_in_bbox(
                    query_feat=query_ft,
                    query_bbox=req["bbox"],
                    sim_threshold=req["score_th"],
                    expand_ratio=expand_ratio,
                )
            )

            if best_obj is None:
                # not found yet: log best_score / threshold / candidates (throttled)
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

            # avoid spamming if not continuous
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

    def shutdown_all_threads(self):
        """Clean up all threads and timers."""
        self.logger.warning("[Main] Shutting down all threads and timers.")
        self._search_thread_stop = True
        try:
            if self._search_thread and self._search_thread.is_alive():
                self._search_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.timer.cancel()
        except Exception as e:
            self.logger.warning(f"[Main] Failed to cancel timer: {e}")
        self.publish_executor.shutdown(wait=True)

    def destroy_node(self):
        """Override base destroy_node with cleanup logic."""
        self.shutdown_all_threads()
        super().destroy_node()


def run_ros2(cfg):
    """Entry point for launching ROS2 runner."""
    rclpy.init()
    runner = RunnerROS2(cfg)
    runner.logger.warning(
        "[Main] ROS2 Runner started. Waiting for data stream..."
    )
    try:
        while rclpy.ok() and not runner.shutdown_requested:
            rclpy.spin_once(runner, timeout_sec=0.1)
    except KeyboardInterrupt:
        runner.logger.warning(
            "[Main] KeyboardInterrupt received. Shutting down."
        )
    finally:
        runner.destroy_node()
        rclpy.shutdown()
        runner.logger.warning("[Main] Done.")
