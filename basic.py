import blenderproc as bproc
import argparse
import glob
import random

import os
os.environ['MPLBACKEND'] = 'agg'

ASSETS = "./assets/glb"
CUBE = "./assets/cube.obj"
OUTPUT_DIR = "./output"
OUTPUT_UNPACKED_DIR = "./output_unpacked"

bproc.init()

# load 20 random objects from the assets directory
glb_files = glob.glob(os.path.join(ASSETS, "*.glb"))
print(f"Found {len(glb_files)} GLB files in {ASSETS}")

# Select up to 20 random GLB files
selected_files = random.sample(glb_files, min(20, len(glb_files)))
print(f"Loading {len(selected_files)} random GLB files:")

objs = []
for i, glb_file in enumerate(selected_files):
    print(f"  {i+1}. {os.path.basename(glb_file)}")
    try:
        # Try to load GLB file using Blender's native import
        import bpy
        
        # Clear selection and import GLB
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_scene.gltf(filepath=glb_file)
        
        # Get the newly imported objects
        loaded_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        
        # Convert to BlenderProc objects
        loaded_mesh_objs = []
        for blender_obj in loaded_objs:
            mesh_obj = bproc.types.MeshObject(blender_obj)
            loaded_mesh_objs.append(mesh_obj)
        
        # Position objects randomly in a 3D grid to avoid overlap
        for obj in loaded_mesh_objs:
            # Random position in a 4x4x4 grid centered at origin
            x = (i % 4 - 1.5) * 2  # -3, -1, 1, 3
            y = ((i // 4) % 4 - 1.5) * 2
            z = ((i // 16) % 4 - 1.5) * 2
            obj.set_location([x, y, z])
        
        objs.extend(loaded_mesh_objs)
        print(f"    Successfully loaded {len(loaded_mesh_objs)} objects")
        
    except Exception as e:
        print(f"    Failed to load {os.path.basename(glb_file)}: {e}")
        continue


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

# Extract HDF5 data to PNGs using helper function
from helpers.hdf5 import extract_hdf5_to_pngs
extract_hdf5_to_pngs(OUTPUT_DIR, OUTPUT_UNPACKED_DIR)