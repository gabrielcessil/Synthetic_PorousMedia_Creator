import os
import glob
import numpy as np
import porespy as ps
from plots import plot_histogram
import pyvista as pv
import csv
from scipy import ndimage
#--- REQUIRED CONFIGURATION ---
# Base directory to start the search


# Real rock test domain (Pre-salt)
#"""
BASE_DIRECTORIES = [
    "./Test_PreSal_120_120_120/Samples"
]
#"""
# Synthetic test domains
"""
BASE_DIRECTORIES = [
                    "./Test_CylinGrain_120_120_120/Samples_CylinGrain", 
                    "./Test_CylinPore_120_120_120/Samples_CylinPore", 
                    "./Test_SphGrain_120_120_120/Samples_SphGrain", 
                    "./Test_SphPore_120_120_120/Samples_SphPore"]

"""

# Real rocks test domain
"""
BASE_DIRECTORIES = [
    "./Test_Oliveira_Leopard_120_120_120/Samples",
    "./Test_Oliveira_Bentheimer_120_120_120/Samples",
    "./Test_Oliveira_Kirby_120_120_120/Samples",
    "./Test_Oliveira_CastleGate_120_120_120/Samples",
    "./Test_Oliveira_Berea_120_120_120/Samples",
    "./Test_Oliveira_BereaSinterGray_120_120_120/Samples",
    "./Test_Oliveira_BereaBuff_120_120_120/Samples",
    "./Test_Oliveira_BereaUpperGray_120_120_120/Samples",
    ]
"""

# Name of the raw data file
RAW_FILENAME = "domain.raw"
# Volume dimensions (X, Y, Z) - YOU MUST CHECK AND SET THESE
VOL_SHAPE = (120, 120, 120)
# Data type of the raw file (e.g., np.uint8) - YOU MUST CHECK AND SET THIS
VOL_DTYPE = np.uint8


overall_csv = os.path.join("./", "datasets_radius.csv")
overall_info_rows = []
for BASE_DIR in BASE_DIRECTORIES:
    out_csv = os.path.join(BASE_DIR, "dataset_radius.csv")
    dataset_info_rows = []

    # ------------------------------
    # Use recursive glob to find all RAW_FILENAME files in all subdirectories
    raw_files = glob.glob(os.path.join(BASE_DIR, "**", RAW_FILENAME), recursive=True)
    print(f"Found {len(raw_files)} {RAW_FILENAME} files")
    
    
    all_pores    = []
    all_throuats = []
    for file_idx, raw_path in enumerate(raw_files):
        # Print the subfolder name for context
        parent_folder = os.path.basename(os.path.dirname(raw_path))
        print(f"\nProcessing: {parent_folder}/{RAW_FILENAME}")
    
        # Load raw binary data and reshape it
        # Note: Requires correct VOL_SHAPE and VOL_DTYPE
        vol = np.fromfile(raw_path, dtype=VOL_DTYPE).reshape(VOL_SHAPE).astype(np.uint8)
        snow = ps.filters.snow_partitioning(vol)
        network = ps.networks.regions_to_network(regions=snow.regions)
        pore_diameters = network['pore.inscribed_diameter'] /2    
        all_pores.extend(pore_diameters)
        
        throat_diameters = network['throat.inscribed_diameter'] /2
        all_throuats.extend(throat_diameters)
        
        p_min = float(np.min(pore_diameters))
        p_mean = float(np.mean(pore_diameters))
        p_std = float(np.std(pore_diameters))
        p_max = float(np.max(pore_diameters))
        
        t_min = float(np.min(throat_diameters))
        t_mean = float(np.mean(throat_diameters))
        t_std = float(np.std(throat_diameters))
        t_max = float(np.max(throat_diameters))
        
        dataset_info_rows.append({
            "file_idx": file_idx,
            "parent_folder": parent_folder,
            "raw_path": raw_path,
        
            "pore_r_min": p_min,
            "pore_r_mean": p_mean,
            "pore_r_std": p_std,
            "pore_r_max": p_max,
        
            "throat_r_min": t_min,
            "throat_r_mean": t_mean,
            "throat_r_std": t_std,
            "throat_r_max": t_max,
        })
    
        # Print summary statistics
        print(f"Pore radius: {np.min(pore_diameters):.2f} {np.mean(pore_diameters):.2f} {np.std(pore_diameters):.2f} {np.max(pore_diameters):.2f}")
        print(f"Throat radius: {np.min(throat_diameters):.2f} {np.mean(throat_diameters):.2f} {np.std(throat_diameters):.2f} {np.max(throat_diameters):.2f}")
        
    
    if not all_pores: print(f"No samples found at {BASE_DIR}")
    else:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dataset_info_rows[0].keys())
            writer.writeheader()
            writer.writerows(dataset_info_rows)
        
        print(f"Saved: {out_csv}")
    
        print("-----")
        print(f"Pore radius: {np.min(all_pores):.2f} {np.mean(all_pores):.2f} {np.std(all_pores):.2f} {np.max(all_pores):.2f}")
        print(f"Throat radius: {np.min(all_throuats):.2f} {np.mean(all_throuats):.2f} {np.std(all_throuats):.2f} {np.max(all_throuats):.2f}")
        print("-----")
    
    overall_info_rows.append({
        "Dataset dir": BASE_DIR,
        
        "pore_r_min":  np.min(all_pores),
        "pore_r_mean": np.mean(all_pores),
        "pore_r_std":  np.std(all_pores),
        "pore_r_max":  np.max(all_pores),
        
        "throat_r_min":  np.min(all_throuats),
        "throat_r_mean": np.mean(all_throuats),
        "throat_r_std":  np.std(all_throuats),
        "throat_r_max":  np.max(all_throuats),
        })
    
    all_pores    = np.asarray(all_pores, dtype=float)
    all_throuats = np.asarray(all_throuats, dtype=float)
    
    plot_histogram(all_pores, all_throuats, BASE_DIR+"/radius_histogram.png")

    # Plot 3D example of pore locations
    pore_coords = network["pore.coords"]          # (Np, 3)
    pore_coords = np.round(pore_coords).astype(int)
    dt          = snow.dt
    peak_values = dt[pore_coords[:,0], pore_coords[:,1], pore_coords[:,2]]
    print("Peak mean: ", np.mean(peak_values), "Inscribed mean: ", np.mean(pore_diameters))
    grid = pv.ImageData()
    grid.dimensions = dt.shape            
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.point_data["edt"] = dt.ravel(order="F")  # VTK-friendly flatten
    cube_structure = np.ones((4, 4, 4))
    pore_mask = np.zeros_like(vol, dtype=bool)
    pore_mask[pore_coords[:,0], pore_coords[:,1], pore_coords[:,2]] = True
    expanded_peaks = ndimage.binary_dilation(pore_mask, structure=cube_structure)
    bin_peaks = vol.astype(np.int16).copy()
    bin_peaks[expanded_peaks] = 10        
    grid.point_data["binary_with_peaks"] = bin_peaks.ravel(order="F")
    grid.save("edt.vti")

    
with open(overall_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=overall_info_rows[0].keys())
    writer.writeheader()
    writer.writerows(overall_info_rows)
