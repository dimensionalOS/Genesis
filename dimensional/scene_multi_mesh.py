import os
import time
import threading
import numpy as np
import cv2
import genesis as gs

gs.init(backend=gs.gpu)


def load_glb_model(scene, glb_path, scale=1.0, group_by_material=True, pos=(0,0,0)):
    """
    Load a GLB model into the scene.
    
    Args:
        scene (gs.Scene): The Genesis scene to add the model to
        glb_path (str): Path to the GLB file
        scale (float): Scale factor for the model
        group_by_material (bool): Whether to group meshes by material
        pos (tuple): Position offset for the model
        
    Returns:
        list: List of entities added to the scene
    """
    # Create a default surface for the model
    surface = gs.options.surfaces.Default()
    
    # Create a mesh morph from the GLB file
    mesh_morph = gs.options.morphs.Mesh(
        file=glb_path,
        scale=scale,
        group_by_material=group_by_material,
        pos=pos,
        fixed=True,  # Make it static so it doesn't fall/move
        euler=(-90, 180, 0)  # Rotate 180 degrees around Y to face forward
    )
    
    # Add the mesh to the scene
    entity = scene.add_entity(
        morph=mesh_morph,
        surface=surface
    )
    
    return [entity]

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(35.0, -35.0, 25.0),
        camera_lookat=(0.0, 0.0, 5.0),
        camera_fov=50,
        max_FPS=60,
    ),
    show_viewer=True,
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0.0, 0.0, -10.0),
    ),
    renderer = gs.renderers.Rasterizer(),
)

# Add a ground plane
plane = scene.add_entity(gs.morphs.Plane())

# Example of loading a GLB model
# Replace 'path/to/your/model.glb' with the actual path to your GLB file
model_entities = load_glb_model(scene, '../assets/office.glb', scale=1.0, pos=(0, 0, 0.0))

scene.build()

for step in range(10000):
    scene.step()
    
