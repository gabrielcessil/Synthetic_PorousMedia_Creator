import numpy as np
import porespy as ps
import os
import scipy.stats as sps # Import for statistical distributions
import utils
import matplotlib.pyplot as plt




# --- Simulation Parameters ---
chunk_size      = 5   # Set for 1h of simulations (5 samples, 20 min per sample)
gres            = "gpu:k40m" #"gpu:k40m"#"gpu:a100"
n_proc          = 1
cpu             = 12 
gpu             = 64

# --- Domains Parameters ---
DIM             = 120
SHAPE           = [DIM, DIM, DIM] # Shape must be a List for the function signature you provided
AXIS_OF_FLOW    = 0 
N_SAMPLES       = 1#5


##########################
# CREATE SPHERICAL PORES #
##########################

output_root = "Test_NonOvSphGrain_120_120_120"
os.makedirs(output_root, exist_ok=True)

volumes             = []
solid_spheres       = True
total_samples       = 0


# Porosity, radius 1, radius 2
pairs = [
    (0.2, 24),
    #(0.2, 32),
    
    #(0.3, 22),
    #(0.3, 24),
    #(0.3, 26),
    #(0.3, 32),

    
    #(0.3, 26),
    #(0.3, 32),
    
    #(0.4, 32),
    #(0.4, 38),
    
    

]
"""
porosities = [0.5, 0.4, 0.3, 0.2, 0.1]
radius     = [24,  12,  9,  6,  3]
"""

solid_sphre = True

def make_solid_spheres(por, rad, SHAPE, seed):
    circFills   = 1 - por
    protrusion  = rad//4
    # First try
    vol         = ps.generators.random_spheres(im_or_shape=SHAPE, 
                                               volume_fraction=circFills,
                                               r=rad, 
                                               edges="extended", 
                                               seed=seed, 
                                               protrusion=0)
    
    vol[:, :, 0]   = 0
    vol[:, :, -1]  = 0
    vol[:, 0, :]   = 0
    vol[:, -1, :]  = 0
    
    plt.imshow(vol[50], cmap='binary', interpolation='none')
    plt.colorbar()  
    plt.show()
    actual_por = np.count_nonzero(1-vol)/(120*120*120)
    
    for i in range(rad):
        print(f"{i}th trial: ")
        print(100*(actual_por - por)/por, actual_por, por) 
        if not utils.is_well_resolved(1-vol, min_pore_mean=3, max_pore_mean=10):
            print("Not well resolved condition. Finalizing...")
            break
        if abs(actual_por - por) < 0.15:
            print("Conditions achieved. Finalizing...")
            break
   
        #rad             = rad-1
        seed            = seed+1
        protrusion      = protrusion+1
        vol             = ps.generators.random_spheres(im_or_shape=vol,
                                                       volume_fraction=circFills, 
                                                       r=rad,
                                                       edges="extended", 
                                                       seed=seed, 
                                                       clearance=0,
                                                       protrusion=protrusion)
        plt.imshow(vol[50], cmap='binary', interpolation='none')
        plt.colorbar()  
        plt.show()
        actual_por = np.count_nonzero(1-vol)/(120*120*120)
        
        print("Target Porosity: ", por, "; Porosity: ", np.count_nonzero(1-vol)/(120*120*120))
        print()
        
    vol     = 1 - vol
    
    return vol
    
def make_fluid_spheres(por, rad, SHAPE, seed):
    
    circFills  = por
    
    clearance  = -rad//2
    
    # First try
    vol         = ps.generators.random_spheres(im_or_shape=SHAPE, 
                                               volume_fraction=circFills,
                                               r=rad, 
                                               edges="extended", 
                                               seed=seed, 
                                               clearance=clearance)
    plt.imshow(vol[50], cmap='binary', interpolation='none')
    plt.colorbar()  
    plt.show()
    actual_por = np.count_nonzero(vol)/(120*120*120)
    
    # Filling 
    while not (actual_por >= por*0.9 and actual_por <= por*1.1) and rad > 4:
        rad             = rad-1
        seed            = seed+1
        clearance       = clearance-1
        vol             = ps.generators.random_spheres(im_or_shape=vol,
                                                       volume_fraction=circFills, 
                                                       r=rad,
                                                       edges="extended", 
                                                       seed=seed, 
                                                       clearance=clearance)
        plt.imshow(vol[50], cmap='binary', interpolation='none')
        plt.colorbar()  
        plt.show()
        actual_por = np.count_nonzero(vol)/(120*120*120)
        
        print("Target Porosity: ", por, "; Porosity: ", np.count_nonzero(vol)/(120*120*120))
   
    return vol 

for por, rad in pairs:
        created = 0
        for n in range(N_SAMPLES*50):
            if created >= N_SAMPLES: break
            
            
 
            # Create volumes
            seed    = int(n*1000+rad*100+por*10)
            
            # Fill is fluid
            vol = make_solid_spheres(por, rad, SHAPE, seed)
            print("Target Porosity: ", por, "; Porosity: ", np.count_nonzero(vol)/(120*120*120))

            
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
            
            break

utils.generate_slurm_run_scripts_chunks(list(range(0, total_samples + 1)),
                                  n_proc,
                                  gres,
                                  output_root,
                                  chunk_size,
                                  cpu, 
                                  gpu,
                                  f"Run_{0}_{total_samples}.sh")
#"""