import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import cc3d
from typing import Tuple, List
from numpy.typing import NDArray
import os
import scipy.stats as sps
import math
from   pathlib       import Path
from   scipy.ndimage import distance_transform_edt
from   typing        import Union
import utils









def make_tubes_volume(MEAN_RADIUS, tubes_fill, solid_tubes, SHAPE, seed=0):
    
    dim         = 120
    phi_max     = 75
    theta_max   = 75
    length      = 80
    maxiter     = 10


    params_generic = {
        "shape":        SHAPE,
        "r":            MEAN_RADIUS,
        "phi_max":      phi_max,
        "theta_max":    theta_max,
        "length":       length,
        "maxiter":      maxiter,
    }
    vol = ps.generators.cylinders(**params_generic, porosity=1-tubes_fill)
    
    # Sphere are void
    if not solid_tubes: vol = 1-vol
    
    return vol
    


# --- Simulation Parameters ---
chunk_size      = 10   # Set for 1h of simulations (5 samples, 20 min per sample)
gres            = "gpu:k40m" #"gpu:k40m"#"gpu:a100"
n_proc          = 1
cpu             = 12 
gpu             = 64

# --- Domains Parameters ---
DIM             = 120
SHAPE           = [DIM, DIM, DIM] # Shape must be a List for the function signature you provided
AXIS_OF_FLOW    = 0 
N_SAMPLES       = 5


##########################
# CREATE SPHERICAL PORES #
##########################

output_root = "Test_CylinPore_120_120_120"
os.makedirs(output_root, exist_ok=True)

volumes             = []
solid_spheres       = False
total_samples       = 0
for SPHERES_FILL in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:    # Porosities large enough so that spheres touch
    for MEAN_RADIUS in [8,12,14,16]:     
        created = 0
        for n in range(N_SAMPLES*50):
            if created >= N_SAMPLES: break

            print(f" Filling {SPHERES_FILL*100}% with Cylinders, Mean Radius {MEAN_RADIUS} ({n})")
            # Create volumes
            seed = int(n*10000+MEAN_RADIUS*1000+SPHERES_FILL*100)
            vol  = make_tubes_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=seed)
            
            # Transform sample for simulation:
            vol[:, :, 0]   = 0
            vol[:, :, -1]  = 0
            vol[:, 0, :]   = 0
            vol[:, -1, :]  = 0
            
            # Check porosity
            actual_porosity = np.sum(vol) / vol.size
            print(f"Actual Porosity: {actual_porosity*100:.2f}%")
            
            # Sanity checks
            if not utils.is_percolating(vol, axis=0):
                print(f"Sample {n} do not percolate and got removed.")
            elif not utils.is_well_resolved(vol, min_pore_mean=3, max_pore_mean=6):
                print(f"Sample {n} has geometry out of scope and got removed.")
            else:        
                print(f"Sample {n} got included.")
                folder_base = f"Sample_{total_samples:05d}"
                utils.create_simulation_force_condition(vol,  output_root, folder_base, reflect=False, bc=5, n_proc=n_proc)
                total_samples +=1
                created+=1
            print("-" * 30)

        

utils.generate_slurm_run_scripts_chunks(list(range(0, total_samples + 1)),
                                  n_proc,
                                  gres,
                                  output_root,
                                  chunk_size,
                                  cpu, 
                                  gpu,
                                  f"Run_{0}_{total_samples}.sh")