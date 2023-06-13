import json, math
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import random
import os
import cv2

image_size = (512, 512)  # (w, h)


def read_meta_data(json_file):
    with open(json_file, "r") as f:
        meta = json.load(f)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = meta["x"]
    extrinsic[:3, 1] = meta["y"]
    extrinsic[:3, 2] = meta["z"]
    extrinsic[:3, 3] = meta["origin"]

    # intrinsic
    intrinsic = np.identity(3)
    focal_x = image_size[0] / 2 / math.tan(meta["x_fov"] / 2.0)
    focal_y = image_size[1] / 2 / math.tan(meta["y_fov"] / 2.0)
    intrinsic[0, 0] = focal_x
    intrinsic[1, 1] = focal_y
    intrinsic[0, 2] = image_size[0] / 2
    intrinsic[1, 2] = image_size[1] / 2

    # max depth
    max_depth = meta["max_depth"]

    # print(f"Check extrinsic and intrinsic:\n{extrinsic}\n{intrinsic}")
    return extrinsic, intrinsic, max_depth


def preprocess_depth_image(depth_image, max_depth):
    """Preprocess depth image, including:
    reshape to single layer, convert to meters, set inf depth to 0.

    Args:
        depth_image (ndarray): depth image with shape (h, w, 1/3)
        max_depth (float): 255 in depth image corresponds to max_depth in meters

    Returns:
        ndarray: depth image with shape (h, w)
    """
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]
    depth_image = depth_image.astype(np.float32)
    depth_image[depth_image >= 255] = 0  # Set inf depth to 0
    depth_image = depth_image * max_depth / 255  # convert to meters

    return depth_image


def convert_rgbd_into_colored_pcd(rgb_image, depth_image, extrinsic, intrinsic):
    """convert rgbd image into colored point cloud in world coordinates

    Args:
        rgb_image (ndarray): rgb image with shape (h, w, 3)
        depth_image (ndarray): depth image with shape (h, w)
        extrinsic (ndarray): extrinsic matrix with shape (4, 4)
        intrinsic (ndarray): intrinsic matrix with shape (3, 3)

    Returns:
        ndarray: colored point cloud with shape (h*w, 6)
    """
    # Create 2D array of pixel coordinates
    rows, cols = np.indices(depth_image.shape)
    pixel_coords = np.stack((cols, rows, np.ones_like(depth_image)), axis=-1)

    # Convert pixel coordinates to camera coordinates
    cam_coords = np.matmul(np.linalg.inv(intrinsic), pixel_coords.reshape(-1, 3).T)
    cam_coords = cam_coords.T.reshape(depth_image.shape[0], depth_image.shape[1], 3)
    cam_coords *= depth_image[..., np.newaxis]

    # Convert camera coordinates to world coordinates
    world_coords = np.matmul(cam_coords.reshape(-1, 3), extrinsic[:3, :3].T)
    world_coords += extrinsic[:3, 3]
    world_coords = world_coords.reshape(depth_image.shape[0], depth_image.shape[1], 3)

    # Flatten the world coordinates to create point cloud
    pcd = world_coords.reshape(-1, 3)

    # Flatten the RGB image to create a color array
    color = rgb_image.reshape(-1, 3)

    # Filter out any points with zero depth (i.e., black point at camera position)
    valid_depth_indices = depth_image.reshape(-1) != 0
    pcd = pcd[valid_depth_indices]
    color = color[valid_depth_indices]

    # Create colored point cloud by concatenating points and colors
    colored_pcd = np.concatenate((pcd, color), axis=1)

    return colored_pcd


def write_pcd_npz(coords, colors, pth_npz):
    """write point cloud to npz file

    Args:
        coords (ndarray): point cloud coordinates with shape (n, 3)
        colors (ndarray): point cloud colors with shape (n, 3)
        pth_npz (str): path to npz file
    """
    np.savez_compressed(
        pth_npz, coords=coords, R=colors[:, 0], G=colors[:, 1], B=colors[:, 2]
    )


if __name__ == "__main__":

    pth_rgbd = sys.argv[-1]

    colored_pcds = np.empty((0, 6))
    
    for i in range(10):
        
        img_name = str(i).zfill(5)
        
        # Read depth and color image:
        depth_image = iio.imread(f"{pth_rgbd}/{img_name}_depth.png")
        rgb_image = iio.imread(f"{pth_rgbd}/{img_name}.png")

        meta_path = f"{pth_rgbd}/{img_name}.json"

        # Read meta:
        extrinsic, intrinsic, max_depth = read_meta_data(meta_path)

        # Check and adjust depth image:
        depth_image = preprocess_depth_image(depth_image, max_depth)

        # Convert rgbd image into point cloud:
        colored_pcd = convert_rgbd_into_colored_pcd(
            rgb_image, depth_image, extrinsic, intrinsic
        )
        colored_pcds = np.concatenate((colored_pcds, colored_pcd), axis=0)

    sample_size = 4096
    if colored_pcds.shape[0] >= sample_size:

        sampled_index = np.random.choice(list(range(colored_pcds.shape[0])), sample_size, replace=False).tolist()
        colored_pcds_sampled = colored_pcds[sampled_index]

        # Write point cloud to npz file:
        write_pcd_npz(colored_pcds_sampled[:, :3], colored_pcds_sampled[:, -3:] / 255, f"{pth_rgbd}/pcd4096.npz")

        # Convert to Open3D.PointCLoud:
        pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(colored_pcds_sampled[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(colored_pcds_sampled[:, -3:] / 255)
        
        # Visualize:
        pth_out = f"{pth_rgbd}/pcd4096.ply"
        o3d.io.write_point_cloud(pth_out, pcd_o3d)

