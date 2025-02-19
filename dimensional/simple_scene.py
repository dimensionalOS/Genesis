import genesis as gs
gs.init(backend=gs.gpu)  
 
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(5.0, -5.0, 5.0),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=50,
    )
)

# Add ground plane
plane = scene.add_entity(gs.morphs.Plane())

# Add the GLB model with fixed=True to make it static
office = scene.add_entity(
    gs.morphs.Mesh(
        file="../assets/office.glb",
        fixed=True,  # Make it static
        euler=(-90, 180, 0),  # Adjust orientation if needed
        pos=(-2, 0, -13.21),  # Position offset (x, y, z)
        scale=1.0,
    )
)

scene.build()

for i in range(10000):
    scene.step() 