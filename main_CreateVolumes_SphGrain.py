import numpy as np
import porespy as ps
import os
import scipy.stats as sps # Import for statistical distributions
import utils




def make_spheres_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=0):
    
    MIN_RADIUS          = min(MEAN_RADIUS/6,4)
    # Standard deviation estimation
    StdDev              = MEAN_RADIUS/3 
    # Define the normal distribution object (Mean=5, StdDev=3)
    radius_distribution = sps.norm(loc=MEAN_RADIUS, scale=StdDev)
    
    # Call the function using the specific signature you requested
    vol = ps.generators.polydisperse_spheres(
        shape   =SHAPE,
        porosity=1-SPHERES_FILL,
        dist    =radius_distribution,   # Pass the statistical distribution object
        r_min   =MIN_RADIUS,            # Ensure the smallest generated sphere is at least 1 voxel
        seed    =seed                   # for reproducibility
    )
    # Sphere are void
    if not solid_spheres: vol = 1-vol
    
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

output_root = "Test_SphGrain_120_120_120"
os.makedirs(output_root, exist_ok=True)

volumes             = []
solid_spheres       = True
total_samples       = 0

config_pairs = [
    
    (0.2, 8),
    (0.2, 10),
    (0.2, 12),
    (0.2, 14),
    
    (0.3, 10),
    (0.3, 12),
    (0.3, 14),
    (0.4, 16),
    
    (0.4, 12),
    (0.4, 14),
    (0.4, 16),
    (0.5, 18),
    
    (0.5, 14),
    (0.5, 16),
    (0.5, 18),
    (0.5, 20),
    
    (0.6, 16),
    (0.6, 18),
    (0.6, 20),
    (0.6, 22),
    
    (0.7, 20),
    (0.7, 22),
    (0.7, 24),
    (0.7, 26),
    
    (0.8, 24),
    (0.8, 26),
    (0.8, 28),
    (0.8, 30),
    ]

for SPHERES_FILL, MEAN_RADIUS in config_pairs:    # Porosities large enough so that spheres touch         
        created = 0
        for n in range(N_SAMPLES*50):
            if created >= N_SAMPLES: break

            print(f" Filling {SPHERES_FILL*100}% with Sphere, Mean Radius {MEAN_RADIUS} ({n})")
            # Create volumes
            seed = int(n*1000+MEAN_RADIUS*100+SPHERES_FILL*10)
            vol = make_spheres_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=seed)
            
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
