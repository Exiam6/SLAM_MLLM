import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_scene_file(filepath):
    """
    Parse a scene .txt file to extract:
      - object bboxes: DataFrame with columns object, x, y, width, length, height
      - wall segments: list of ((x1,y1),(x2,y2)) tuples
    """
    objects = []
    walls = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("bbox_") and "Bbox(" in line:
                # e.g. bbox_0=Bbox(sofa, x,y,z,yaw,w,l,h)
                content = line.split("Bbox(", 1)[1].rstrip(")")
                parts = content.split(",")
                obj    = parts[0]
                x, y   = float(parts[1]), float(parts[2])
                w      = float(parts[5])
                l      = float(parts[6])
                h      = float(parts[7])
                objects.append({
                    'object': obj,
                    'x':       x,
                    'y':       y,
                    'width':   w,
                    'length':  l,
                    'height':  h
                })
            # Parse wall lines
            elif line.startswith("wall_") and "Wall(" in line:
                # e.g. wall_0=Wall(x1,y1,z1, x2,y2,z2, height, 0.0)
                content = line.split("Wall(", 1)[1].rstrip(")")
                parts = content.split(",")
                x1, y1 = float(parts[0]), float(parts[1])
                x2, y2 = float(parts[3]), float(parts[4])
                walls.append(((x1, y1), (x2, y2)))
    df_objs = pd.DataFrame(objects)
    return df_objs, walls

def plot_map(df, walls, out_path):
    """
    Plot top-down map with:
      - walls drawn as line segments
      - object bboxes drawn as rectangles
    """
    fig, ax = plt.subplots(figsize=(6,6))

    # Draw walls
    for (x1, y1), (x2, y2) in walls:
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=2)

    # Draw objects
    for _, row in df.iterrows():
        lower_left = (row['x'] - row['width']/2,
                      row['y'] - row['length']/2)
        rect = plt.Rectangle(
            lower_left,
            row['width'], row['length'],
            edgecolor='blue',
            facecolor='none',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(row['x'], row['y'], row['object'],
                ha='center', va='center', fontsize=6, color='blue')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(Path(out_path).stem)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith('.txt'):
            continue
        scene_id = Path(fname).stem
        scene_path = os.path.join(input_dir, fname)
        df_objs, walls = parse_scene_file(scene_path)

        print(f"Scene {scene_id}: {len(walls)} wall segments, {len(df_objs)} objects")
        out_png = os.path.join(output_dir, f"{scene_id}.png")
        plot_map(df_objs, walls, out_png)

        objects_list = df_objs.to_dict(orient='records')
        walls_list = [
            {"start": [x1, y1], "end": [x2, y2]}
            for (x1, y1), (x2, y2) in walls
        ]
        out_json = os.path.join(output_dir, f"{scene_id}.json")
        with open(out_json, 'w') as jf:
            json.dump({"objects": objects_list, "walls": walls_list}, jf, indent=2)

    print(f"Done. Maps saved in: {output_dir}")

if __name__ == "__main__":
    INPUT_DIR  = "/home/zifan/SpatialLM/ARKitScenes_Outputs"
    OUTPUT_DIR = "/home/zifan/SpatialLM/scene_maps"

    main(INPUT_DIR, OUTPUT_DIR)