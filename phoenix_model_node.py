#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import os
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from sensor_fuser.frame import compute_intrinsics

from datetime import datetime
import time

simulation = False


class PhoenixModel(Node):
    def __init__(self):
        super().__init__('phoenix_model')

        self.latest_img = None
        self.processed_img = None
        self.counter = 0
        self.bridge = CvBridge()
        self.busy = False  # guard against overlap
        self.baseline = 0.24584925
        self.fx = 406.3323

        # Load TorchScript model
        model_path = (
            '/home/nexsos/ros2_ws/src/sensor_fuser/sensor_fuser/models/'
            'pretrained_NewCRF_V5_256x640.pt'
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_model = torch.jit.load(model_path, map_location=self.device).eval()
        self.get_logger().info(f"{self.device}")

        # Publishers
        self.publisher_thr_depth = self.create_publisher(Image, 'thr_depth', 10)
        self.publisher_thr_cropped = self.create_publisher(Image, 'thr_cropped', 10)

        # Subscriptions
        self.subscription = self.create_subscription(
            Image,
            'airsim_node/Drone1/thermal_front/Infrared',
            self.image_callback,
            10,
        )
        # Inference timer – you can relax this to 0.2 for less load
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.subscribe_camera_info = self.create_subscription(
            CameraInfo,
            "/airsim_node/Drone1/thermal_front/Infrared/camera_info",
            self.camera_info_callback,
            10,
        )
        self.publish_new_camera_info = self.create_publisher(
            CameraInfo, "/new_camera_info", 10
        )

        # Pre-create CLAHE to avoid realloc each frame
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Visualization control (set to False for max performance)
        self.enable_vis_saving = True
        self.vis_every_n = 10  # save every N frames, not every frame

    def log(self, text):
        pass
        # self.get_logger().info(text)

    def camera_info_callback(self, msg: CameraInfo):
        intr = compute_intrinsics(640, 512, 76.44, True)
        new_msg = CameraInfo()

        new_msg.header = msg.header
        new_msg.width = 640
        new_msg.height = 256

        fx = intr['fx']
        fy = intr['fy']
        cx = intr['cx']
        cy = intr['cy']

        new_msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        new_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        new_msg.distortion_model = "plumb_bob"

        new_msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        new_msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        self.publish_new_camera_info.publish(new_msg)

    def timer_callback(self):
        # Skip if nothing to publish to (saves work when offline)
        if (
            self.latest_img is None
            or (
                self.publisher_thr_depth.get_subscription_count() == 0
                and self.publisher_thr_cropped.get_subscription_count() == 0
            )
        ):
            return
        if self.busy:
            return
        self.busy = True
        try:
            self.processing()
        finally:
            self.busy = False

    def image_callback(self, msg: Image):
        # Always keep only the latest frame
        self.latest_img = msg

    def processing(self):
        # Read as 8-bit
        thr_np = self.bridge.imgmsg_to_cv2(
            self.latest_img,
            desired_encoding='mono16',
        )

        # Center crop: 128:384 => 256x640
        thr_cropped = thr_np[128:384, :]

        # ---------- VISUALIZATION PIPELINE (for display only) ----------
        img8 = cv2.normalize(thr_cropped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 0. Convert to float
        vis = thr_cropped.astype(np.float32)

        # 1. Robust clip around the median to ignore extreme outliers
        flat_vis = vis.ravel()
        vmed = np.median(flat_vis)
        # Take, say, the central 80% of values around the median
        lo = np.percentile(flat_vis, 10.0)
        hi = np.percentile(flat_vis, 90.0)

        # Clamp to [lo, hi], then re-center around median if needed
        vis = np.clip(vis, lo, hi)

        # 2. Normalize to [0,1]
        vis -= lo
        vis /= (hi - lo + 1e-8)

        # 3. Optional gamma to emphasize midtones
        #gamma = 0.8  # <1 -> brighten mid-range, >1 -> darken
       # vis = np.power(vis, gamma)

        # 4. Convert to uint8 for visualization
        vis_u8 = (vis * 255.0).astype(np.uint8)
        vis_cropped = vis_u8.copy()
        # ---------------------------------------------------------------

        # ---------- MODEL PREPROCESSING (unchanged, uses thr_cropped) ----------
        if not simulation:
            img = thr_cropped.astype(np.float32)
            flat = img.ravel()
            tmin = np.percentile(flat, 1.0)
            tmax = np.percentile(flat, 99.0)

            img = np.clip(img, tmin, tmax)
            img = (img - tmin) / (tmax - tmin + 1e-8)  # [0,1]

            img_8u = (img * 255.0).astype(np.uint8)
            img_8u = self.clahe.apply(img_8u)
            img = img_8u.astype(np.float32) / 255.0  # back to [0,1]
        else:
            img = thr_cropped.astype(np.float32) / 255.0

        # 3) add channel dim and normalize
        img = img[..., None]  # (256,640,1)
        mean, std = 0.45, 0.225
        img = (img - mean) / std

        self.processed_img = img
        self.run_inference(thr_cropped, self.latest_img.header, img8)

    def run_inference(self, thr_cropped, in_header, vis_cropped):
        if self.processed_img is None:
            return
        self.counter += 1

        # processed_img: (H, W, 1)
        inp = self.processed_img.transpose(2, 0, 1)[None, ...].astype(np.float32)

        # TorchScript inference
        tensor = torch.from_numpy(inp).to(self.device)  # (1,1,H,W)
        with torch.no_grad():
            pred_depth = self.depth_model(tensor)

        depth = pred_depth.squeeze().detach().cpu().numpy().astype(np.float32)

        # Optional visualization (throttled)
        if self.enable_vis_saving and self.counter % self.vis_every_n == 0:
            self._save_vis(depth, vis_cropped)

        depth_scaled = depth * 1000.0

        msg = self.bridge.cv2_to_imgmsg(
            depth_scaled.astype(np.uint16),
            encoding="mono16",
        )
        msg.header = in_header

        msg2_cropped = self.bridge.cv2_to_imgmsg(
            thr_cropped.astype(np.uint8),
            encoding="mono8",
        )
        msg2_cropped.header = in_header

        self.publisher_thr_cropped.publish(msg2_cropped)
        self.publisher_thr_depth.publish(msg)

        self.latest_img = None

    def _save_vis(self, depth_arr, input_img):
        x = np.nan_to_num(depth_arr)
        inv_depth = 1.0 / (x + 1e-6)

        vmax = np.percentile(inv_depth, 95.0)
        normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
        vis_rgb = (mapper.to_rgba(inv_depth)[..., :3] * 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        if input_img.ndim == 2:
            input_bgr = cv2.applyColorMap(input_img, cv2.COLORMAP_INFERNO)
        else:
            input_bgr = input_img.astype(np.uint8)

        stacked = cv2.vconcat([input_bgr, vis_bgr])

        out_dir = "/home/nexsos/Desktop/imge/test"
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(out_dir, f"depth_inference_{self.counter}.png"),
            stacked,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PhoenixModel()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()