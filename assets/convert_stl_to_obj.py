#!/usr/bin/env python3
import os
import argparse
import trimesh

def convert_stl_to_obj(root_dir: str):
    """
    Recursively finds all .stl files under `root_dir` and exports each
    to a .obj in the same directory.
    """
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".stl"):
                stl_path = os.path.join(dirpath, fn)
                obj_path = os.path.splitext(stl_path)[0] + ".obj"

                # Load & export
                mesh = trimesh.load(stl_path)
                mesh.export(obj_path)

                print(f"Converted {stl_path} → {obj_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Recursively convert all STL files to OBJ"
    )
    p.add_argument(
        "root_dir",
        help="Root folder to scan (e.g. path/to/dim_cpp)",
    )
    args = p.parse_args()

    # sanity check
    if not os.path.isdir(args.root_dir):
        print(f"Error: '{args.root_dir}' is not a directory.")
        exit(1)

    convert_stl_to_obj(args.root_dir)
