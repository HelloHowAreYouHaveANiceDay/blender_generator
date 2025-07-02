import blenderproc as bproc
import argparse
import glob
import random
import math
import sys
import os

# Set environment variables
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
selected_files = random.sample(glb_files, min(50, len(glb_files)))
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
        
        # Position objects randomly above the ground plane for physics simulation
        for obj in loaded_mesh_objs:
            # Random position above ground with some spread
            x = random.uniform(-4, 4)  # Wider spread for more interesting physics
            y = random.uniform(-4, 4)
            z = random.uniform(3, 8)   # Start high up for falling simulation
            obj.set_location([x, y, z])
            
            # Enable rigid body physics for each object (make them active/falling)
            obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        
        objs.extend(loaded_mesh_objs)
        print(f"    Successfully loaded {len(loaded_mesh_objs)} objects")
        
    except Exception as e:
        print(f"    Failed to load {os.path.basename(glb_file)}: {e}")
        continue

# Create a ground plane for objects to fall onto
print("Creating ground plane for physics simulation...")
ground_plane = bproc.object.create_primitive("PLANE")
ground_plane.set_scale([20, 20, 1])  # Much larger plane to catch all falling objects
ground_plane.set_location([0, 0, 0])  # Place at origin
ground_plane.enable_rigidbody(active=False)  # Make it passive (static obstacle)

# Create walls around the ground plane to form a bowl/container
print("Creating walls to contain objects...")
wall_height = 2.0  # Lower walls so cameras can see over them
wall_thickness = 0.5
wall_distance = 10.0  # Distance from center to wall

# North wall
north_wall = bproc.object.create_primitive("CUBE")
north_wall.set_scale([wall_distance, wall_thickness, wall_height])
north_wall.set_location([0, wall_distance, wall_height])
north_wall.enable_rigidbody(active=False)

# South wall
south_wall = bproc.object.create_primitive("CUBE")
south_wall.set_scale([wall_distance, wall_thickness, wall_height])
south_wall.set_location([0, -wall_distance, wall_height])
south_wall.enable_rigidbody(active=False)

# East wall
east_wall = bproc.object.create_primitive("CUBE")
east_wall.set_scale([wall_thickness, wall_distance, wall_height])
east_wall.set_location([wall_distance, 0, wall_height])
east_wall.enable_rigidbody(active=False)

# West wall
west_wall = bproc.object.create_primitive("CUBE")
west_wall.set_scale([wall_thickness, wall_distance, wall_height])
west_wall.set_location([-wall_distance, 0, wall_height])
west_wall.enable_rigidbody(active=False)

print("Running physics simulation...")
# Run physics simulation and fix final poses
bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=3,   # Simulate for at least 3 seconds
    max_simulation_time=15,  # Simulate for at most 15 seconds  
    check_object_interval=0.5,  # Check every 0.5 seconds if objects stopped
    object_stopped_location_threshold=0.01,  # Movement threshold
    object_stopped_rotation_threshold=0.01,  # Rotation threshold
    substeps_per_frame=10,   # Higher accuracy
    solver_iters=20          # Better stability
)
print("Physics simulation complete. Objects settled.")


# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

# define the camera resolution
bproc.camera.set_resolution(512, 512)

# Generate 5 camera positions that all point to the center of the scene

# Center of the scene (target point) - now focusing on ground level
target = [0, 0, 0.5]  # Slightly above ground to better frame the objects

# Generate 5 camera positions in a circle around the target
radius = 7.0   # Move cameras inside the bowl walls (walls are at 10 units)
height = 8.0   # Much higher up to look down into the bowl from above

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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers.hdf5 import extract_hdf5_to_pngs
extract_hdf5_to_pngs(OUTPUT_DIR, OUTPUT_UNPACKED_DIR)