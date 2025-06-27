import blenderproc as bproc
import argparse

import os
os.environ['MPLBACKEND'] = 'agg'

CUBE = "./assets/cube.obj"
OUTPUT_DIR = "./output"
OUTPUT_UNPACKED_DIR = "./output_unpacked"

bproc.init()

# load the objects into the scene
# objs = bproc.loader.load_obj(SCENE)
objs = bproc.loader.load_obj(CUBE)


# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

# define the camera resolution
bproc.camera.set_resolution(512, 512)

# Generate 5 camera positions that all point to the center of the scene
import math

# Center of the scene (target point)
target = [0, 0, 0]

# Generate 5 camera positions in a circle around the target
radius = 5.0  # Distance from center
height = 2.0  # Height above the target

for i in range(5):
    # Calculate angle for circular positioning
    angle = (i / 5.0) * 2 * math.pi
    
    # Calculate camera position
    cam_x = radius * math.cos(angle)
    cam_y = radius * math.sin(angle)
    cam_z = height
    
    position = [cam_x, cam_y, cam_z]
    
    # Calculate rotation to look at target
    # Use BlenderProc's utility to create look-at matrix
    matrix_world = bproc.math.build_transformation_mat(position, bproc.camera.rotation_from_forward_vec(
        forward_vec=[target[0] - position[0], target[1] - position[1], target[2] - position[2]]
    ))
    
    bproc.camera.add_camera_pose(matrix_world)

# activate normal and depth rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container for compatibility with the notebook
bproc.writer.write_hdf5(OUTPUT_DIR, data)

# Custom function to extract HDF5 data and save as discrete PNGs with metadata
import h5py
import json
import numpy as np
from PIL import Image

def extract_hdf5_to_pngs(hdf5_dir, output_dir):
    """Extract HDF5 files and save as PNG images with JSON metadata"""
    import glob
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files
    hdf5_files = glob.glob(os.path.join(hdf5_dir, "*.hdf5"))
    
    for hdf5_file in hdf5_files:
        frame_idx = os.path.basename(hdf5_file).split('.')[0]
        
        with h5py.File(hdf5_file, 'r') as f:
            metadata = {}
            
            # Extract and save each dataset
            for key in f.keys():
                data = f[key][()]
                
                if key == "colors":
                    # Save RGB image as PNG
                    if len(data.shape) == 3:  # RGB image
                        # Convert to uint8 if needed
                        if data.dtype != np.uint8:
                            data = (data * 255).astype(np.uint8)
                        img = Image.fromarray(data)
                        img.save(os.path.join(output_dir, f"rgb_{frame_idx}.png"))
                        metadata[key] = f"rgb_{frame_idx}.png"
                
                elif key == "depth":
                    # Save depth as PNG (scaled to 16-bit)
                    if len(data.shape) == 2:  # Depth image
                        # Handle invalid values (NaN, inf) by clipping to valid range
                        data_clean = np.nan_to_num(data, nan=0.0, posinf=65.535, neginf=0.0)
                        # Clip to valid range for 16-bit (0-65.535 meters -> 0-65535 mm)
                        data_clipped = np.clip(data_clean, 0, 65.535)
                        # Scale depth to 16-bit range
                        depth_scaled = (data_clipped * 1000).astype(np.uint16)  # Convert to mm
                        img = Image.fromarray(depth_scaled, mode='I;16')
                        img.save(os.path.join(output_dir, f"depth_{frame_idx}.png"))
                        metadata[key] = f"depth_{frame_idx}.png"
                        metadata[f"{key}_scale"] = 0.001  # Scale factor to convert back to meters
                
                elif key == "normals":
                    # Save normals as PNG
                    if len(data.shape) == 3:
                        # Normalize normals to 0-255 range
                        normals_scaled = ((data + 1) * 127.5).astype(np.uint8)
                        img = Image.fromarray(normals_scaled)
                        img.save(os.path.join(output_dir, f"normals_{frame_idx}.png"))
                        metadata[key] = f"normals_{frame_idx}.png"
                
                else:
                    # For other data types, save metadata only
                    if isinstance(data, np.ndarray):
                        if data.size < 100:  # Small arrays, save as list
                            try:
                                metadata[key] = data.tolist()
                            except (ValueError, TypeError):
                                metadata[key] = f"Array shape: {data.shape}, dtype: {data.dtype}"
                        else:
                            metadata[key] = f"Array shape: {data.shape}, dtype: {data.dtype}"
                    elif isinstance(data, bytes):
                        # Handle bytes data by converting to string or decoding
                        try:
                            metadata[key] = data.decode('utf-8')
                        except UnicodeDecodeError:
                            metadata[key] = f"Binary data ({len(data)} bytes)"
                    else:
                        try:
                            # Try to convert to JSON-serializable format
                            if hasattr(data, 'tolist'):
                                metadata[key] = data.tolist()
                            else:
                                # Test if it's JSON serializable
                                json.dumps(data)
                                metadata[key] = data
                        except (TypeError, ValueError):
                            metadata[key] = str(data)
            
            # Save metadata as JSON
            with open(os.path.join(output_dir, f"metadata_{frame_idx}.json"), 'w') as json_file:
                json.dump(metadata, json_file, indent=2)

# Extract HDF5 data to PNGs
extract_hdf5_to_pngs(OUTPUT_DIR, OUTPUT_UNPACKED_DIR)