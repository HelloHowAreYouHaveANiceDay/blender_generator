"""
Helper functions for working with HDF5 files from BlenderProc
"""

import os
import glob
import json
import h5py
import numpy as np
from PIL import Image


def extract_hdf5_to_pngs(hdf5_dir, output_dir):
    """
    Extract HDF5 files and save as PNG images with JSON metadata
    
    Args:
        hdf5_dir (str): Directory containing HDF5 files
        output_dir (str): Directory to save PNG files and metadata
    """
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
    
    print(f"Extracted {len(hdf5_files)} HDF5 files to PNG format in '{output_dir}'")
