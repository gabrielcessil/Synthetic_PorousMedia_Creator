import os
import re
from typing import List, Tuple
import torch
import numpy as np
import pyvista as pv
from   scipy.ndimage import distance_transform_edt
import h5py
# -------------------------------------------------------------------
# 1) Helpers for folder / file discovery
# -------------------------------------------------------------------

def list_sample_dirs(base_dir: str, sample_dir_pattern: str) -> List[str]:
    """
    List all directories named 'DeePore_Sample_XXXXX' inside base_dir,
    sorted by the numeric suffix (e.g. 00010 -> 10).

    Returns a list of folder names (not full paths).
    """
    pattern = re.compile( sample_dir_pattern)

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


def get_raw_path(sample_dir: str, raw_filename: str) -> str:
    """
    Return the full path to the raw file (e.g. domain.raw) inside the sample_dir.
    """
    raw_path = os.path.join(sample_dir, raw_filename)
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    return raw_path


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

    # Pick highest Y
    vis_candidates.sort(key=lambda t: t[0])
    _, latest_vis_dir = vis_candidates[-1]

    summary_path = os.path.join(latest_vis_dir, "summary.pvti")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"'summary.pvti' not found in: {latest_vis_dir}")

    return summary_path


# -------------------------------------------------------------------
# 2) Reading the raw volume and the pvti
# -------------------------------------------------------------------

def read_raw_volume(
    raw_path: str,
    shape: Tuple[int, int, int],
    dtype: np.dtype,
    order: str = "C",
) -> np.ndarray:
    """
    Read a .raw file as a 3D NumPy array with the given shape and dtype.

    Parameters
    ----------
    raw_path : str
        Full path to the .raw file.
    shape : (nx, ny, nz)
        Shape of the 3D volume.
    dtype : np.dtype
        Data type stored in the raw file (e.g. np.uint8, np.float32).
    order : {'C', 'F'}
        Memory order used when reshaping.

    Returns
    -------
    np.ndarray
        3D array with the specified shape and dtype.
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


def read_summary_pvti(summary_path: str) -> pv.DataSet:
    """
    Read a summary.pvti file as a PyVista object.
    """
    return pv.read(summary_path)


# -------------------------------------------------------------------
# 3) Main high-level functions
# -------------------------------------------------------------------

def load_sample_raw_and_pvti(
    base_dir: str,
    sample_dir_name: str,
    raw_filename: str,
    raw_shape: Tuple[int, int, int],
    raw_dtype: np.dtype,
    raw_order: str = "C",
):
    """
    For a single sample:

    - Go to {base_dir}/{sample_dir_name}
    - Load raw_filename (e.g. domain.raw) as a 3D array
    - Find the highest visY folder inside the sample and read summary.pvti

    Returns
    -------
    (raw_volume, pvti_mesh)
      raw_volume : np.ndarray
      pvti_mesh  : pyvista.DataSet
    """

    sample_dir = os.path.join(base_dir, sample_dir_name)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Sample folder not found: {sample_dir}")

    # domain.raw dentro do sample
    raw_path = get_raw_path(sample_dir, raw_filename)

    raw_volume = read_raw_volume(
        raw_path,
        shape=raw_shape,
        dtype=raw_dtype,
        order=raw_order,
    )

    # summary.pvti na maior visY dentro do sample
    summary_path = get_latest_vis_summary_path(sample_dir)
    pvti_mesh    = read_summary_pvti(summary_path)

    return raw_volume, pvti_mesh

# True if everything is okay
def sanity_check(vol, vel, solid_value=0):
    solid_mask = (vol == solid_value)
    return not np.any(vel[solid_mask] != 0)


# ---- main builder using HDF5 ----

h5_path             = "Train_Deepore_256_256_256_dataset.h5"
simulations_folder  = "Train_Deepore_256_256_256/DeePore_Samples/"
sample_dir_pattern  = r"^DeePore_Sample_(\d+)$"
raw_name            = "domain.raw"
raw_shape           = (256, 256, 256)  # (D, H, W)
raw_dtype           = np.uint8
N_samples           = 100


base_dir            = os.path.join(os.getcwd(), simulations_folder)
sample_dirs         = list_sample_dirs(base_dir, sample_dir_pattern)

with h5py.File(h5_path, "w") as f:
    D, H, W     = raw_shape

    # Máximo de pontos porosos por amostra (50% de 256^3)
    max_points  = (D * H * W) // 2  # 8_388_608 para 256^3

    # Metadados gerais
    f.attrs["description"] = (
        "LBPM velocity + EDT only where edt>0, "
        "fixed-size per sample with max 50% porosity"
    )
    f.attrs["raw_shape"]   = raw_shape
    f.attrs["vel_dtype"]   = "float16"
    f.attrs["coorX_dtype"] = "uint8"
    f.attrs["coorY_dtype"] = "uint8"
    f.attrs["coorZ_dtype"] = "uint8"
    f.attrs["edt_dtype"]   = "float16"
    f.attrs["max_points"]  = max_points

    # Create datasets for each information, expansible in samples dimension
    # Velocity Z
    vel_z_ds = f.create_dataset(
        "vel_z",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float16",
        chunks=(1, max_points),
    )
    # Velocity Y
    vel_y_ds = f.create_dataset(
        "vel_y",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float16",
        chunks=(1, max_points),
    )
    # Velocity X
    vel_x_ds = f.create_dataset(
        "vel_x",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float16",
        chunks=(1, max_points),
    )
    # Coordinates (x,y,z)
    coorZ_ds = f.create_dataset(
        "coorZ",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="uint8",
        chunks=(1, max_points),
    )
    coorY_ds = f.create_dataset(
        "coorY",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="uint8",
        chunks=(1, max_points),
    )
    coorX_ds = f.create_dataset(
        "coorX",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="uint8",
        chunks=(1, max_points),
    )
    # Distance transform
    edt_ds = f.create_dataset(
        "edt",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float16",
        chunks=(1, max_points),
    )
    # Number of porous voxels
    n_valid_ds = f.create_dataset(
        "n_valid",
        shape=(0,),
        maxshape=(None,),
        dtype="int64",
    )
    # Sample name
    string_dt = h5py.string_dtype(encoding="utf-8")
    sample_names_ds = f.create_dataset(
        "sample_names",
        shape=(0,),
        maxshape=(None,),
        dtype=string_dt,
    )

    sample_count = 0

    for sample_name in sample_dirs:
        sample_dir = os.path.join(base_dir, sample_name)
        raw_path   = os.path.join(sample_dir, raw_name)

        if not os.path.isfile(raw_path):
            print(f"[SKIP] {sample_name}: no {raw_name}")
            continue

        try:
            # Load Binary domain
            vol = read_raw_volume(raw_path, raw_shape, raw_dtype)

            # Load Simulation data
            summary_path = get_latest_vis_summary_path(sample_dir)
            mesh         = read_summary_pvti(summary_path)
            vel_x = mesh["Velocity_x"].reshape(raw_shape)
            vel_y = mesh["Velocity_y"].reshape(raw_shape)
            vel_z = mesh["Velocity_z"].reshape(raw_shape)

            # Define porosity
            sign_dist   = mesh["SignDist"].reshape(raw_shape)
            porous_mask = (vol == 1)

            if not np.any(porous_mask):
                print(f"[SKIP] {sample_name}: no pore space (SignDist > 0)")
                continue

            # Recalculate the distance transform with Scipy
            edt_full = distance_transform_edt(porous_mask).astype("float32")

            # Get indexes from porous cells
            i, j, k = np.where(porous_mask)
            N_points = k.size
            
            # Check if every porous cell can be stores in 'max_points' columns
            if N_points > max_points:
                raise RuntimeError(
                    f"Sample {sample_name} has {N_points} pore points, "
                    f"exceeds max_points={max_points}"
                )

            # Flatten data from porous region
            vel_z_flat = vel_z[porous_mask].astype(np.float16)
            vel_y_flat = vel_y[porous_mask].astype(np.float16)
            vel_x_flat = vel_x[porous_mask].astype(np.float16)
            edt_flat   = edt_full[porous_mask].astype(np.float16)
            # Type convertion
            i_coords = i.astype(np.uint8)
            j_coords = j.astype(np.uint8)
            k_coords = k.astype(np.uint8)

            # coords (N_points, 3) com [k, j, i]
            

            # --- 3) Padding para tamanho fixo max_points ---
            vel_z_row   = np.zeros(max_points, dtype=np.float16) 
            vel_y_row   = np.zeros(max_points, dtype=np.float16)
            vel_x_row   = np.zeros(max_points, dtype=np.float16)
            coorZ_row   = np.zeros(max_points, dtype=np.uint8)
            coorY_row   = np.zeros(max_points, dtype=np.uint8)
            coorX_row   = np.zeros(max_points, dtype=np.uint8)
            edt_row     = np.zeros(max_points, dtype=np.float16)
            
            # Fill data
            vel_z_row[:N_points]  = vel_z_flat
            vel_y_row[:N_points]  = vel_y_flat
            vel_x_row[:N_points]  = vel_x_flat
            coorZ_row[:N_points]  = k_coords
            coorY_row[:N_points]  = j_coords
            coorX_row[:N_points]  = i_coords
            edt_row[:N_points]    = edt_flat

            # --- 4) Aumenta datasets em 1 amostra e escreve a linha ---
            idx = sample_count

            vel_z_ds.resize((idx + 1, max_points))
            vel_y_ds.resize((idx + 1, max_points))
            vel_x_ds.resize((idx + 1, max_points))
            
            coorZ_ds.resize((idx + 1, max_points))
            coorY_ds.resize((idx + 1, max_points))
            coorX_ds.resize((idx + 1, max_points))
            
            edt_ds.resize((idx + 1, max_points))
            n_valid_ds.resize((idx + 1,))
            sample_names_ds.resize((idx + 1,))

            vel_z_ds[idx, :]  = vel_z_row
            vel_y_ds[idx, :]  = vel_y_row
            vel_x_ds[idx, :]  = vel_x_row
            coorZ_ds[idx, :]  = coorZ_row
            coorY_ds[idx, :]  = coorY_row
            coorX_ds[idx, :]  = coorX_row
            edt_ds[idx, :]    = edt_row

            n_valid_ds[idx]      = N_points
            sample_names_ds[idx] = sample_name

            sample_count += 1
            print(
                f"[OK] {sample_name}: {N_points} pontos porosos "
                f"(padded to {max_points})"
            )

        except Exception as e:
            print(f"[FAIL] {sample_name}: {e}")
        
        if sample_count>=N_samples: break
    print(f"Finished. Total samples written: {sample_count}")