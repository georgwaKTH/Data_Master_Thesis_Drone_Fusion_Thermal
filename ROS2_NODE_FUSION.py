import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
import message_filters

from . import point_cloud2
from sensor_fuser.frame import compute_intrinsics, depth_to_point_cloud, calculate_confidence, sensor_fusion, cloud_to_depth

NODE_NAME = 'SensorFusion'
QUEUESIZE = 3
TIME_DIFF = 0.2


def find_horizontal_regions(binary_img, width=120, height=40):
    bin01 = (binary_img == 0).astype(np.uint8)
    kernel = np.ones((height, width), np.uint8)
    eroded = cv2.erode(bin01, kernel, iterations=1)
    mask01 = cv2.dilate(eroded, kernel, iterations=1)
    return np.where(mask01 == 1, 0, 255).astype(np.uint8)


class SensorFuser(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.bridge = CvBridge()

        self.sub_thr = message_filters.Subscriber(self, Image, 'thr_depth')
        self.sub_stereo = message_filters.Subscriber(self, Image, 'stereo_depth')
        self.sub_thr_raw = self.create_subscription(Image, 'airsim_node/Drone1/thermal_front/Infrared', self.image_callback_thr_raw, 1)
        self.sub_scene = self.create_subscription(Image, '/airsim_node/Drone1/rgbd_front/Scene', self.image_callback_scene, 1)

        ts = message_filters.ApproximateTimeSynchronizer([self.sub_thr, self.sub_stereo], queue_size=QUEUESIZE, slop=TIME_DIFF)
        ts.registerCallback(self.listener_callback)

        self.publisher_ = self.create_publisher(PointCloud2, 'scan_cloud', 3)
        self.temp_pub = self.create_publisher(PointCloud2, 'temp_cloud_thr', 1)
        self.temp_pub_ste = self.create_publisher(PointCloud2, 'temp_cloud_ste', 1)
        self.temp_pub_processed_thr = self.create_publisher(Image, 'processed_thr_depth', 1)
        self.pub_reconstruct_depth = self.create_publisher(Image, 'reconstruct_depth', 1)

        self.latest_thr_raw = None
        self.latest_scene_gray = None
        self.prev_thr = None
        self.prev_ste = None

        self.INTR_STE = compute_intrinsics(640, 512, 76.44, thr=True)
        self.INTR_THR = compute_intrinsics(640, 512, 76.44, thr=True)

        
        self.MIN_RANGE = 0.5
        self.MAX_RANGE = 10.0
        self.FUSION_THRESH = 1.0
        self.crop = slice(128, 384)
        self.log_every = 30
        self.counter = 0

        self.kernel_dilate = np.ones((2, 2), np.uint8)
        self.kernel_sobel = np.ones((5, 5), np.uint8)

    def image_callback_thr_raw(self, msg: Image):
        self.latest_thr_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def image_callback_scene(self, msg: Image):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.latest_scene_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _compute_sobel_mask(self):
        if self.latest_scene_gray is None:
            return None
        sobelx = cv2.Sobel(self.latest_scene_gray, cv2.CV_32F, 1, 0, ksize=5)
        abs_sobel = cv2.convertScaleAbs(sobelx)
        _, sobel_mask = cv2.threshold(abs_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sobel_mask = cv2.dilate(sobel_mask, self.kernel_sobel, iterations=1)
        return sobel_mask[self.crop, :]

    def _process_thermal_depth(self, img_thr_m):
        if self.latest_thr_raw is None:
            return img_thr_m, None
        raw = self.latest_thr_raw
        temp_mask = cv2.bitwise_not(raw)
        thr = (temp_mask > temp_mask.mean()).astype(np.uint8)
        thr_inv = 1 - thr
        thr = thr[self.crop, :]
        thr_inv = thr_inv[self.crop, :]
        ground_mask = find_horizontal_regions(thr * 255)

        img_thr_m = img_thr_m.copy()
        img_thr_m[thr == 1] = 255.0  # mark sky as far/invalid
        valid = (img_thr_m > 0) & (img_thr_m < 255.0)
        if np.any(valid):
            p90 = np.percentile(img_thr_m[valid], 90)  # top 10% cutoff
            far_mask = img_thr_m >= p90
            img_thr_m[far_mask] = 255.0

        blurred = cv2.bilateralFilter(img_thr_m, 15, 3, 30)
        out = img_thr_m.copy()
        select = ground_mask == 255
        out[select] = blurred[select]

        mask_valid = select & (thr_inv > 0)
        if np.any(mask_valid):
            col_means = np.nanmean(
                np.where(mask_valid, out, np.nan).astype(np.float32),
                axis=0,
            )
            col_means = np.nan_to_num(col_means, nan=0.0)
            out[mask_valid] = np.broadcast_to(col_means, out.shape)[mask_valid]

        out = cv2.dilate(out, self.kernel_dilate, iterations=1)

        return out.astype(np.float32, copy=False), ground_mask

    def listener_callback(self, msg_thr, msg_stereo):
        self.counter += 1
        img_thr = self.bridge.imgmsg_to_cv2(msg_thr, desired_encoding='16UC1').astype(np.float32) / 1000.0
        img_ste = self.bridge.imgmsg_to_cv2(msg_stereo, desired_encoding='32FC1').astype(np.float32)
        img_ste = np.where(img_ste > 16000, 0, img_ste)[self.crop, :]

        sobel_mask = self._compute_sobel_mask()
        img_thr, _ = self._process_thermal_depth(img_thr)

        proc_thr_msg = self.bridge.cv2_to_imgmsg(img_thr, encoding='32FC1')
        proc_thr_msg.header.stamp = msg_thr.header.stamp
        proc_thr_msg.header.frame_id = 'Drone1/thermal_front_optical'
        self.temp_pub_processed_thr.publish(proc_thr_msg)

        pt1 = depth_to_point_cloud(img_thr, intr=self.INTR_THR, min_range=self.MIN_RANGE, max_range=self.MAX_RANGE)
        ps1 = depth_to_point_cloud(img_ste, intr=self.INTR_STE, min_range=self.MIN_RANGE, max_range=self.MAX_RANGE)

        if self.prev_thr is None or self.prev_ste is None:
            fused = sensor_fusion(pt1, ps1, thresh=self.FUSION_THRESH)
        else:
            ct = calculate_confidence(pt1, self.prev_thr)
            cs = calculate_confidence(ps1, self.prev_ste)
            fused = sensor_fusion(pt1, ps1, confidence_stereo=cs, confidence_thermal=ct, thresh=self.FUSION_THRESH)

        self.prev_thr = pt1
        self.prev_ste = ps1

        depth_image_reconstruct = cloud_to_depth(
            fused,
            self.INTR_THR['fx'], self.INTR_THR['fy'], self.INTR_THR['cx'], self.INTR_THR['cy'],
            640, 256
        )

        if sobel_mask is not None and ps1.size:
            stereo_recon = cloud_to_depth(
                ps1,
                self.INTR_STE['fx'], self.INTR_STE['fy'], self.INTR_STE['cx'], self.INTR_STE['cy'],
                640, 256
            )
            depth_image_reconstruct = np.where(sobel_mask == 255, stereo_recon, depth_image_reconstruct).astype(np.uint16)

        msg_depth_img = self.bridge.cv2_to_imgmsg(depth_image_reconstruct, encoding='mono16')
        msg_depth_img.header.stamp = msg_thr.header.stamp
        msg_depth_img.header.frame_id = 'Drone1/thermal_front_optical'
        self.pub_reconstruct_depth.publish(msg_depth_img)

        header_ = Header()
        header_.stamp = msg_thr.header.stamp
        header_.frame_id = 'Drone1/rgbd_front_optical'
        self.publisher_.publish(point_cloud2.create_cloud_xyz32(header_, fused))

        if self.temp_pub.get_subscription_count() > 0:
            self.temp_pub.publish(point_cloud2.create_cloud_xyz32(header_, pt1))
        if self.temp_pub_ste.get_subscription_count() > 0:
            self.temp_pub_ste.publish(point_cloud2.create_cloud_xyz32(header_, ps1))

        if self.counter % self.log_every == 0:
            self.get_logger().info(f'Fusion frame {self.counter}: thr_pts={len(pt1)} ste_pts={len(ps1)} fused_pts={len(fused)}')


def main(args=None):
    rclpy.init(args=args)
    node = SensorFuser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
