####################
"""
CODE TO ACCESS CERTAIN FOLDER, AND ITERATE ACROSS SUBFOLDERS
CREATING A ARRAY FOR INPUTS (DISTANCE TRANSFORMS)
AND ANOTHER ARRAY FOR VELOCITIES
"""
#####################




import os
import re
import random
from typing import List, Tuple, Optional
import torch
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from   typing        import Union

# -------------------------------------------------------------------
# 1) HELPER FUNCTIONS
# -------------------------------------------------------------------

def list_sample_dirs(base_dir: str) -> List[str]:
    """
    List all directories named 'DeePore_Sample_XXXXX' inside base_dir,
    sorted by the numeric suffix (e.g. 00010 -> 10).
    """
    pattern = re.compile(r"^Sample_(\d+)$")
    samples: List[Tuple[int, str]] = []
    
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue
        m = pattern.match(name)
        if m:
            num_part = int(m.group(1))
            samples.append((num_part, name))
    
    samples.sort(key=lambda t: t[0])
    return [name for _, name in samples]


def get_latest_vis_summary_path(sample_dir: str) -> str:
    """
    Inside sample_dir, find all subdirectories named 'visY' where Y is an integer.
    Select the highest Y and return the path to 'summary.pvti' inside it.
    """
    vis_pattern = re.compile(r"^vis(\d+)$")
    vis_candidates: List[Tuple[int, str]] = []
    
    for name in os.listdir(sample_dir):
        full_path = os.path.join(sample_dir, name)
        if not os.path.isdir(full_path):
            continue
        m = vis_pattern.match(name)
        if m:
            y = int(m.group(1))
            vis_candidates.append((y, full_path))
    
    if not vis_candidates:
        raise RuntimeError(f"No 'visY' subdirectories found in: {sample_dir}")
    
    vis_candidates.sort(key=lambda t: t[0])
    _, latest_vis_dir = vis_candidates[-1]
    
    summary_path = os.path.join(latest_vis_dir, "summary.pvti")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"'summary.pvti' not found in: {latest_vis_dir}")
    
    return summary_path


def read_raw_volume(
    raw_path: str,
    shape: Tuple[int, int, int],
    dtype: np.dtype,
    order: str = "C",
) -> np.ndarray:
    """
    Read a .raw file as a 3D NumPy array with the given shape and dtype.
    """
    flat = np.fromfile(raw_path, dtype=dtype)
    expected_size = int(np.prod(shape))
    
    if flat.size != expected_size:
        raise ValueError(
            f"Raw file size mismatch for {raw_path}: "
            f"found {flat.size} elements, expected {expected_size} "
            f"for shape {shape}"
        )
    
    return flat.reshape(shape, order=order)

def force_calculation(
    matriz_binaria: np.ndarray,
    tau:            Union[float, int],
    Re:             float = 0.01,
    Dens:           float = 1.0,
) -> float:
    
    dist_transform  = distance_transform_edt(matriz_binaria)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)
    Visc            = (tau - 0.5) / 3.0
    Fx              = (Re * 8.0 * (Visc ** 2)) / (Dens * (R ** 3))
    
    return Fx

# -------------------------------------------------------------------
# 2) Main function
# -------------------------------------------------------------------

def make_arrays(
    base_dir:     str,
    n_samples:    int,
    raw_filename: str = "domain.raw",
    raw_shape:    Tuple[int, int, int] = (256, 256, 256),
    raw_dtype:    np.dtype = np.uint8,
    output_dir:   str = ".",
    dataset_name: str = "LBPM_Dataset",
    seed:         Optional[int] = 42
):
    """
    Create a simple PyTorch dataset with exactly N samples using random sampling.
    
    Args:
        base_dir: Directory containing DeePore_Sample_XXXXX folders
        n_samples: Exact number of samples to include in the dataset
        raw_filename: Name of the raw file (default: "domain.raw")
        raw_shape: Shape of the 3D volume (D, H, W)
        raw_dtype: Data type of raw file
        output_dir: Where to save the dataset
        dataset_name: Base name for output files
        seed: Random seed for reproducibility
        oversample_factor: Factor to oversample when some samples might fail
        
    Returns:
        inputs, targets, metadata
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Get all available sample directories
    all_sample_dirs = list_sample_dirs(base_dir)
    total_available = len(all_sample_dirs)
    
    print(f"Found {total_available} total sample directories")
    print(f"Requested: {n_samples} samples")
    
    if n_samples > total_available:
        raise ValueError(f"Requested {n_samples} samples but only {total_available} available")
    
    # Calculate how many to try (oversample in case some fail)
    samples_to_try = min(n_samples , total_available)
    print(f"Will try to process {samples_to_try} samples")
    
    # Randomly select samples to try
    selected_samples = random.sample(all_sample_dirs, samples_to_try)
    
    # Lists to collect successful data
    input_tensors   = []    # Binary solid/pore (1 channel)
    target_tensors  = []    # Velocity fields (3 channels)
    valid_samples   = []
    failed_samples  = []
    
    # Process until we get exactly n_samples
    for sample_idx, sample_name in enumerate(selected_samples):
        # Stop if we already have enough samples
        if len(input_tensors) >= n_samples:
            break
            
        sample_dir = os.path.join(base_dir, sample_name)
        raw_path   = os.path.join(sample_dir, raw_filename)
        
        if not os.path.isfile(raw_path):
            print(f"[SKIP] {sample_name}: no {raw_filename}")
            failed_samples.append((sample_name, f"Missing {raw_filename}"))
            continue
        
        try:
            print(f"[{sample_idx+1}/{samples_to_try}] Processing {sample_name} "
                  f"(collected: {len(input_tensors)}/{n_samples})...")
            
            # 1. Load binary domain
            vol        = read_raw_volume(raw_path, raw_shape, raw_dtype)
            
            # Convert to binary: solid=0, pore=1
            # Assuming raw values: 0=solid, 1=pore (adjust if needed)
            binary_vol   = (vol > 0).astype(np.float32)  # Shape: (D, H, W)
            
            
            # 2. Load simulation data
            summary_path = get_latest_vis_summary_path(sample_dir)
            mesh         = pv.read(summary_path)
            
            # Extract velocity components
            vx          = mesh["Velocity_x"].reshape(raw_shape).astype(np.float32)
            vy          = mesh["Velocity_y"].reshape(raw_shape).astype(np.float32)
            vz          = mesh["Velocity_z"].reshape(raw_shape).astype(np.float32)

            # Normalize / Adimensionalize
            Tau             = 1.5
            Re              = 0.1
            Dens            = 1.0
            gz              = force_calculation(binary_vol, Tau, Re, Dens)
            vx, vy, vz      = vx/(Tau*gz), vy/(Tau*gz), vz/(Tau*gz)
            
            # Stack velocities to create 3-channel tensor
            velocity_field  = np.stack([vx, vy, vz], axis=0)    # Shape: (3, D, H, W)
            
            # 3. Add channel dimension to binary volume
            binary_vol      = binary_vol[np.newaxis, ...]       # Shape: (1, D, H, W)
            
            # 4. Convert to PyTorch tensors
            input_tensor    = torch.from_numpy(binary_vol)      # (1, D, H, W)
            target_tensor   = torch.from_numpy(velocity_field)  # (3, D, H, W)
            
            # 5. Basic sanity checks
            solid_mask      = (input_tensor == 0).expand_as(target_tensor)
            edt             = mesh["SignDist"].reshape(raw_shape).astype(np.float32)
            edt_vol         = (edt>0).astype(np.uint8)
            test1 = np.array_equal(edt_vol, vol)
            test2 = (target_tensor[solid_mask] != 0).sum().item() > 0
            test3 = np.all(binary_vol[0,:, :, 0] == 0) and np.all(binary_vol[0,:, :, -1] == 0)
            test4 = np.all(binary_vol[0,:, 0, :] == 0) and np.all(binary_vol[0,:, -1, :] == 0)
            test5 = np.all(binary_vol[0,0, :, :] == 0) and np.all(binary_vol[0,-1, :, :] == 0)
                
                
            print("Sanity checks: ")
            print(" - Raw and Velocities allign   (Must be True):   ", test1)
            print(" - Velocities in solid cells   (Must be False):  ", test2)
            print(" - Outer walls exist in x-face (Must be True):   ", test3)
            print(" - Outer walls exist in y-face (Must be True):   ", test4)
            print(" - Outer walls exist in z-face (Must be False):  ", test5)
            
            if not test1 or test2 or not test3 or not test4 or test5: raise Exception("Sanity checks failed.")
    
            # 6. Append to lists
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
            valid_samples.append(sample_name)
            
            print(f"    ✓ Added: input {input_tensor.shape}, target {target_tensor.shape}")
            
            
            
            
            """
            grid                        = pv.UniformGrid(raw_shape)
            grid["Binary_Volume"]       = binary_vol.ravel(order='C')
            grid["Distance_Transform"]  = edt.ravel(order='C')
            grid["Vx"]                  = vx.ravel(order='C')
            grid["Vy"]                  = vy.ravel(order='C')
            grid["Vz"]                  = vz.ravel(order='C')
            grid.save("Example_Output.vti")
            break
            """
            
        except Exception as e:
            print(f"[FAIL] {sample_name}: {e}")
            failed_samples.append((sample_name, str(e)))
            continue
    
    # Use exactly n_samples (or fewer if we didn't get enough)
    final_n         = min(len(input_tensors), n_samples)
    input_tensors   = input_tensors[:final_n]
    target_tensors  = target_tensors[:final_n]
    valid_samples   = valid_samples[:final_n]
    
    # 7. Stack all samples (C, D, H, W) to (N, C, D, H, W)
    print(f"\nStacking {final_n} samples...")
    all_inputs      = torch.stack(input_tensors, dim=0)
    all_targets     = torch.stack(target_tensors, dim=0)
    
    print("Final dataset shapes:")
    print(f"  Inputs:  {all_inputs.shape}  (samples, channels, depth, height, width)")
    print(f"  Targets: {all_targets.shape} (samples, channels, depth, height, width)")
    
    # 8. Create output directory
    output_path     = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 9. Save PyTorch files
    inputs_path     = output_path / f"{dataset_name}_inputs.pt"
    targets_path    = output_path / f"{dataset_name}_targets.pt"
    
    torch.save(all_inputs, inputs_path)
    torch.save(all_targets, targets_path)
    
    # 10. Save metadata
    metadata = {
        "num_samples": final_n,
        "requested_samples": n_samples,
        "input_shape": all_inputs.shape[1:],  # (1, D, H, W)
        "target_shape": all_targets.shape[1:],  # (3, D, H, W)
        "sample_names": valid_samples,
        "raw_shape": raw_shape,
        "description": f"Simple PyTorch dataset with {final_n} randomly sampled examples",
        "input_dtype": "float32 (binary: 0=solid, 1=pore)",
        "target_dtype": "float32 (velocity components)",
        "random_seed": seed,
        "total_available_samples": total_available,
        "failed_samples": failed_samples[:20],  # Save first 20 failures
    }
    
    metadata_path = output_path / f"{dataset_name}_metadata.pt"
    torch.save(metadata, metadata_path)
    
    print("\nDataset saved to:")
    print(f"  Inputs:  {inputs_path}")
    print(f"  Targets: {targets_path}")
    print(f"  Metadata: {metadata_path}")
    
    # 11. Quick verification
    print("\nVerification:")
    print(f"  Input range:  [{all_inputs.min():.3f}, {all_inputs.max():.3f}]")
    print(f"  Target range: [{all_targets.min():.3f}, {all_targets.max():.3f}]")
    print(f"  Input unique values: {torch.unique(all_inputs)}")
    
    # Porosity statistics
    porosities = all_inputs.mean(dim=(1, 2, 3, 4)).numpy()  # Mean over spatial dimensions
    print(f"  Porosity range: [{porosities.min():.3f}, {porosities.max():.3f}]")
    print(f"  Average porosity: {porosities.mean():.3f}")
    
    return all_inputs, all_targets, metadata


# Configuration
BASE_DIR        = "/home/gabriel/remote/hal/dissertacao/simulations/bentheimer_120_120_120"
SAMPLES_DIR     = "/Samples"

N_SAMPLES       = 9  
DATASET_NAME    = f"LBPM_{N_SAMPLES}samples"
RAW_SHAPE       = (120, 120, 120)
RAW_DTYPE       = np.uint8
OUTPUT_DIR      = BASE_DIR+"/Arrays"

# Random sampling of N samples
print("Creating dataset with random sampling")
make_arrays(
        base_dir    =BASE_DIR+SAMPLES_DIR,
        n_samples   =N_SAMPLES,
        raw_shape   =RAW_SHAPE,
        raw_dtype   =RAW_DTYPE,
        output_dir  =OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        seed        =42,
)
        

# VERIFY SAVED FIELD
"""
import torch
import pyvista as pv
raw_shape = (120,120,120)
def load_my_data(inputs_path, targets_path):
    inputs          = torch.load(inputs_path, map_location="cpu")
    targets         = torch.load(targets_path, map_location="cpu")
    targets         = targets.permute(0,2,3,4,1)                    # (B,C,Z,Y,X) -> (B,Z,Y,X,C)
    inputs          = inputs.permute(0,2,3,4,1)                     # (B,C,Z,Y,X) -> (B,Z,Y,X,C)
    targets         = targets.detach().cpu().numpy()
    inputs          = inputs.detach().cpu().numpy()
    test_data_solid = inputs
    test_data_vel   = targets
    
    
    binary_vol      = inputs[0,:,:,:,0]
    vz              = targets[0,:,:,:,2]
    grid            = pv.UniformGrid(raw_shape)
    grid["VOL"]     = binary_vol.ravel(order='C')
    grid["Vz"]      = vz.ravel(order='C')
    grid.save("Example_LOADED.vti")
    
    return test_data_solid, test_data_vel

inputs_path                     = "Test_Bentheimer_120_120_120/LBPM_9samples_inputs.pt" # Samples 120^3
targets_path                    = "Test_Bentheimer_120_120_120/LBPM_9samples_targets.pt" # Samples 120^3

load_my_data(inputs_path, targets_path)
"""