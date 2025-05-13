import os
import open3d as o3d
import pandas as pd
import numpy as np
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors

download_dir = "/home/zifan/ARKitScenes/data"
output_dir = "/home/zifan/SpatialLM/ARKitScenes_Resized"
os.makedirs(output_dir, exist_ok=True)
download_list_csv = "/home/zifan/ARKitScenes/download_list.csv"
df_download = pd.read_csv(download_list_csv)
estimated_height = 2.5

for _, row in df_download.iterrows():
    video_id = str(row['video_id'])
    fold = row['fold']
    
    input_file = os.path.join(download_dir, "3dod", fold, video_id, f"{video_id}_3dod_mesh.ply")
    
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}")
        continue
    
    point_cloud = load_o3d_pcd(input_file)
    points, colors = get_points_and_colors(point_cloud)
    
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    height = max_z - min_z
    print(f"Video ID {video_id}: original height = {height}")
    
    scale = estimated_height / height
    points = points * scale
    
    resized_point_cloud = o3d.geometry.PointCloud()
    resized_point_cloud.points = o3d.utility.Vector3dVector(points)
    resized_point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    
    output_file = os.path.join(output_dir, f"{video_id}_resized.ply")
    o3d.io.write_point_cloud(output_file, resized_point_cloud)
    print(f"Resized point cloud saved: {output_file}")