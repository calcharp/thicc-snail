import numpy as np

def cf_shell(label="shell",
             total_time=20,
             time_step=0.25/30,
             b=0.1, d=4, z=0, a=1, phi=0, psi=0,
             n=0, n_amp=0,
             c=0, c_amp=0,
             k=0, 
             points_on_aperture=40,
             length=1,
             smooth=True 
             ):
    
    # Number of time steps
    num_timesteps = int(total_time / time_step) + 1
    # Array to hold the matrices for each time step and theta
    results = np.zeros((num_timesteps, points_on_aperture, 3))
    
    thetas = np.linspace(0, 2 * np.pi, points_on_aperture, endpoint=False)

    for i, t in enumerate(np.linspace(0, total_time, num_timesteps)):
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        axial_term = 1 + c_amp * np.sin(c * t)
        
        # Rotation matrix based on psi
        R = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi),  np.cos(psi), 0],
                      [0,            0,           1]])

        for j, theta in enumerate(thetas):
            spiral_term = 1 + n_amp * np.sin(n * theta)
            
            # Normal and Binormal vectors at this theta
            _N = np.array([
                b * cos_t - sin_t,
                -b * sin_t - cos_t,
                0
            ]) / np.sqrt(b**2 + 1)

            _B = np.array([
                b * z * (b * sin_t + cos_t),
                b * z * (b * cos_t - sin_t),
                d * (b**2 + 1)
            ]) / np.sqrt((b**2 + 1) * ((b**2 + 1) * (d**2) + (b**2) * (z**2)))

            # Compute the transformation for current t and theta
            vector_combination = np.array([
                (a * np.sin(theta) * np.cos(phi) + np.cos(theta) * np.sin(phi)) * spiral_term * _N +
                (a * np.sin(theta) * np.sin(phi) - np.cos(theta) * np.cos(phi) + k) * spiral_term * _B
            ])

            rotated_points = np.matmul(R, vector_combination.T).T
            results[i, j, :] = axial_term * (np.exp(b * t) - (1 / (t + 1))) * rotated_points

    return results
