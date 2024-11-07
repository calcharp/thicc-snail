import bpy
import numpy as np

def make_contreras_snail(label="snail", 
                         b = .1, d = 4, z = 1, a = 5, phi = 0, psi = 0, 
                         c_depth=0.1, c_n = 10, n_depth = 0, n = 0, 
                         h_0 = 1, eps = 0.5,
                         time = 20, n_points_time = 1000, 
                         n_points_aperture=100):

    # snail axes are [XYZ][Aperture Angle Theta][Time]

    _lambda: np.array = np.zeros((3, n_points_time, n_points_aperture))
    gamma: np.array = np.zeros((3, n_points_time, n_points_aperture))
    C: np.array = np.zeros((3, n_points_time, n_points_aperture))

    # vector of points in time reshaped for broadcasting
    t: np.array = np.linspace(0, time, n_points_time).reshape((n_points_time, 1))
    theta: np.array = np.linspace(0, 2*np.pi, n_points_aperture).reshape((1, n_points_aperture))

    # precalculating some repeated operations for efficiency
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    bsq = b**2
    dsq = d**2

    radial_ribbing = (1 + n_depth + n_depth*np.sin(n*theta))
    spiral_ribbing = (1 + c_depth + c_depth*np.sin(c_n*t))
    #spiral_ribbing = 1 + c_depth + (c_depth*np.sin(c_n*t)) + np.abs((c_depth*np.sin(c_n*t)))

    # shape = (3, n_points_time, 1) -> [xyz][ti)me][constant with respect to the angle theta along the aperture]
    gamma = np.exp(b*t)*np.array([
        d*sin_t, d*cos_t, 
        np.full(n_points_time, z).reshape(n_points_time, 1)
    ])

    # Defining the normal and binormal vectors along the frennet frame for all time points and angles about the aperture
    # 3 x n_points_time x 1 for both
    N: np.array = np.array([
        b*cos_t - sin_t,
        -b*sin_t - cos_t,
        np.zeros((n_points_time, 1))
    ])/np.sqrt(b**2 + 1)

    B: np.array = np.array([
        b*z*(b*sin_t + cos_t),
        b*z*(b*cos_t - sin_t),
        np.full((n_points_time, 1), d*(bsq + 1))
    ])/np.sqrt(
        (bsq + 1)*((bsq + 1)*dsq + bsq*(z**2))
    )

    # Define the rotation matrix for the aperture given that psi 
    # is the rotation angle about the B axis in the local frennet frame
    R: np.array = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    # Define the unrotated and unscaled generating curve
    # Shape = (1, 1, n_points_aperture) | constant for [xyz][time], [changes for the angle]
    GC: np.array = (a*sin_theta*np.sin(phi) - cos_theta*np.cos(phi)).reshape(1, 1, n_points_aperture)*B + (a*sin_theta*np.cos(phi) + cos_theta*np.sin(phi)).reshape(1, 1, n_points_aperture)*N
    GC_outer: np.array = radial_ribbing*GC 

    # einsum lets me broadcast the rotation matrix multiplication over the time points and the aperture angles
    rGC_outer = np.einsum('ij,jkl->ikl', R, GC_outer)
    rGC = np.einsum('ij,jkl->ikl', R, GC)
    timescale = (np.exp(b*t) - (1 / (t + 1)))
    #adjusting the wave height from 1 to (1+c_depth) so the percentage is always > 100%
    outer_timescale = spiral_ribbing*timescale
    # eps is a modified version of the shell width parameter described by Okabe 2017
    inner_timescale = timescale - ((timescale**eps)*h_0)

    # We tranpose the angle and time axes so the final shape is [XYZ][Aperture][Time], which is easier to index for mesh
    outer_mesh = np.transpose(gamma + (outer_timescale * rGC_outer), (0, 2, 1))
    inner_mesh = np.transpose(gamma + (inner_timescale * rGC), (0, 2, 1))

    # find the indices of the vertices and their faces that will be used for Blender mesh conversion
    indices = np.array(range(n_points_aperture*n_points_time))
    index_grid = indices.reshape(n_points_time, n_points_aperture)
    v1 = index_grid[:, :-1]                       
    v2 = index_grid[:, 1:]                         
    v3 = np.roll(index_grid, -1, axis=0)[:, 1:]     
    v4 = np.roll(index_grid, -1, axis=0)[:, :-1]    
    # Stack the adjusted indices to form faces with vertical wrapping
    faces = np.stack((v1, v2, v3, v4), axis=-1).reshape(-1, 4)

    # Reshape the inner and outer meshes so their shapes are compatable with Blender
    return {"label": label,
            "outer_mesh" : outer_mesh, 
            "inner_mesh" : inner_mesh,
            "outer_mesh_vertices": outer_mesh.reshape(3, n_points_aperture*n_points_time),
            "inner_mesh_vertices": inner_mesh.reshape(3, n_points_aperture*n_points_time),
            "vertex_indices": indices,
            "face_vertex_indices": faces
            }
            

snail = make_contreras_snail(z = 1.3, a = 1, d=1, phi=0, psi=0,
                             b=.15,
                             n_depth=0, n=0, 
                             c_n=0, c_depth=0,  
                             time=200, n_points_time=500, 
                             n_points_aperture=500, 
                             h_0 = 40, eps=.8)
                             
                             
verts = snail['outer_mesh_vertices'].T.tolist()
faces = snail['face_vertex_indices'].tolist()
label = snail['label']


mesh = bpy.data.meshes.new(f"{label}Mesh")   
obj = bpy.data.objects.new(label, mesh) 
mesh.from_pydata(verts, [], faces)   
mesh.update(calc_edges=True)              
bpy.context.collection.objects.link(obj) 

material = bpy.data.materials.new(name=f"{label}_Material")
material.use_nodes = True
material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (122/256, 87/256, 29/256, 1)

obj.data.materials.append(material)


# Use Blender to find the bounding box and calculate the scaling
bpy.context.view_layer.update()  # Make sure all changes are up-to-date

# Get the bounding box points (these are given as tuples, no need to use `.co`)
bbox = obj.bound_box
x_coords = [v[0] for v in bbox]  # Extracting the x-coordinates from the bounding box points
current_length_x = max(x_coords) - min(x_coords)

# Calculate the scaling factor to make the x length equal to 1
scaling_factor = 1.0 / current_length_x

# Apply uniform scaling to the entire object in Blender
obj.scale = (scaling_factor, scaling_factor, scaling_factor)

# Update the view layer to reflect changes
bpy.context.view_layer.update()









