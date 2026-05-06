import numpy as np
import cv2
from scipy.spatial import cKDTree


def upscale_thr_image(thr_image, stereo_res):
    h_st, w_st = stereo_res
    return cv2.resize(thr_image, (w_st, h_st), interpolation=cv2.INTER_LINEAR)


def compute_intrinsics(width, height, fov_deg, thr=False):
    hfov = np.deg2rad(fov_deg / 2.0)
    fx = width / (2.0 * np.tan(hfov))
    fy = fx
    cx = width / 2.0
    cy = (512.0 / 2.0 - 128.0) if thr else (height / 2.0)
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}


def depth_to_point_cloud(depth_img, intr, min_range=None, max_range=None):
    depth = np.asarray(depth_img, dtype=np.float32)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    z = depth.ravel()
    if min_range is None:
        mask = z > 0
    else:
        mask = z > min_range
    if max_range is not None:
        mask &= z < max_range
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)
    u = u.ravel()[mask]
    v = v.ravel()[mask]
    z = z[mask]
    x = (u - intr['cx']) * z / intr['fx']
    y = (v - intr['cy']) * z / intr['fy']
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def calculate_confidence(p1, p2, n=100):
    if p1.size == 0 or p2.size == 0:
        return 0.0
    q1 = p1[::n]
    q2 = p2[::n]
    if q1.size == 0 or q2.size == 0:
        return 0.0
    tree = cKDTree(q2)
    distance, _ = tree.query(q1, workers=-1)
    if distance.size == 0:
        return 0.0
    return float(np.mean(np.exp(-distance)))


def sensor_fusion(reframed_thr_cloud, stereo_cloud, confidence_stereo=None, confidence_thermal=None, conf_thresh=0.5, thresh=0.1):
    if reframed_thr_cloud.size == 0:
        return stereo_cloud.astype(np.float32, copy=False)
    if stereo_cloud.size == 0:
        return reframed_thr_cloud.astype(np.float32, copy=False)

    tree = cKDTree(stereo_cloud)
    distance, indexes = tree.query(reframed_thr_cloud, workers=-1)
    close_mask = distance < thresh

    if not np.any(close_mask):
        if confidence_stereo is None:
            return np.empty((0, 3), dtype=np.float32)
        if confidence_stereo >= (confidence_thermal or 0.0):
            return stereo_cloud.astype(np.float32, copy=False)
        return reframed_thr_cloud.astype(np.float32, copy=False)

    points_stereo = stereo_cloud[indexes[close_mask]]
    points_thermal = reframed_thr_cloud[close_mask]

    if confidence_stereo is None or confidence_thermal is None:
        fused_close = (points_stereo + points_thermal) * 0.5
    else:
        denom = max(confidence_stereo + confidence_thermal, 1e-6)
        ws = confidence_stereo / denom
        wt = confidence_thermal / denom
        fused_close = points_stereo * ws + points_thermal * wt

    if confidence_stereo is None or confidence_thermal is None:
        return fused_close.astype(np.float32, copy=False)

    if confidence_stereo > conf_thresh or confidence_thermal > conf_thresh:
        if confidence_stereo >= confidence_thermal:
            keep = stereo_cloud
        else:
            keep = reframed_thr_cloud
        far_mask = ~close_mask
        if np.any(far_mask):
            extra = keep[indexes[far_mask]] if confidence_stereo >= confidence_thermal else keep[far_mask]
            return np.concatenate((fused_close, extra), axis=0).astype(np.float32, copy=False)
    return fused_close.astype(np.float32, copy=False)


def cloud_to_depth(points_cam, fx, fy, cx, cy, width, height, z_min=1.0, z_max=10.0):
    if points_cam.size == 0:
        return np.zeros((height, width), dtype=np.uint16)
    pts = np.asarray(points_cam, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    valid = (z > z_min) & (z < z_max)
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.uint16)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    u = np.rint(fx * x / z + cx).astype(np.int32)
    v = np.rint(fy * y / z + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]
    v = v[in_bounds]
    z_mm = np.rint(z[in_bounds] * 1000.0).astype(np.uint16)
    depth = np.zeros((height, width), dtype=np.uint16)
    flat = v * width + u
    order = np.argsort(z_mm)
    depth.ravel()[flat[order]] = z_mm[order]
    return depth
