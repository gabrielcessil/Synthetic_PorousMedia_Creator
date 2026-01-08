import os
import numpy         as np
import math
from   pathlib       import Path
from   typing        import Tuple,  List
from   scipy.ndimage import distance_transform_edt
from   typing        import Union
from   numpy.typing  import NDArray
import cc3d
import matplotlib.pyplot as plt
import porespy as ps


def generate_slurm_run_scripts_chunks(
    sample_indices: List[int],
    n_proc: int,
    gres: str,
    output_root: str,
    samples_per_job: int,
    cpu: int, 
    gpu: int,
    dispatcher_name: str = "submit_all_sims_chain.sh",
):
    """
    Gera:
      1) Vários scripts SLURM do tipo run_sims_<START>_<END>.sh,
         cada um rodando até `samples_per_job` simulações em sequência.
      2) Um script "dispatcher" que encadeia esses jobs via dependência:
         j1 -> j2 -> j3 ..., todos com prioridade baixa (nice=1000).
    """

    # Garantir ordem consistente dos índices
    sample_indices = sorted(sample_indices)

    output_root_path = Path(output_root+dispatcher_name).resolve()
    scripts_dir      = output_root_path.parent
    
    print("scripts_dir: ", scripts_dir)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. Quebrar lista de amostras em chunks de tamanho samples_per_job -----
    chunks = [
        sample_indices[i : i + samples_per_job]
        for i in range(0, len(sample_indices), samples_per_job)
    ]

    chunk_script_names = []  # para o dispatcher saber os nomes

    for chunk_id, chunk_samples in enumerate(chunks):
        # range de índices desse chunk
        start_idx = chunk_samples[0]
        end_idx   = chunk_samples[-1]

        range_id = f"{start_idx:05d}_{end_idx:05d}"  # ex: 00220_00229

        chunk_script_name = f"run_sims_{range_id}.sh"
        chunk_script_path = scripts_dir / chunk_script_name
        chunk_script_names.append(chunk_script_name)

        # ---------- SLURM header para o job desse chunk ----------
        chunk_content = f"""#!/bin/bash

# ---------------- SLURM Job Settings ----------------
#SBATCH --oversubscribe
#SBATCH --job-name=Perm_{range_id}
#SBATCH --partition=all_gpu
#SBATCH --gres={gres}:{n_proc}
#SBATCH --mem-per-gpu={gpu}G                  # RAM 64GB per a100.
#SBATCH --cpus-per-gpu={cpu}                  # 12 Cores per a100

#SBATCH -t 7-0:00
#SBATCH -o perm_{range_id}_%j.out
#SBATCH -e perm_{range_id}_%j.err

# ---------------- Environment Setup ----------------
module load lbpm/gpu/poro_dev_78ba76

# Ir para o diretório raiz das amostras

echo "=== Chunk {chunk_id:03d} | Samples {start_idx:05d} to {end_idx:05d} ==="

# ---------------- Job Execution (Sequential within this chunk) ----------------
"""

        # ---------- Bloco de comandos sequenciais para esse chunk ----------
        first = True
        for sample_i in chunk_samples:
            folder_base = f"Sample_{sample_i:05d}"

            if first:
                command_block = f"""
echo "--- Launching simulation for {folder_base} ---"
cd {folder_base}
echo "Current Simulation: " ${{PWD##*/}}
mpirun -np {n_proc} lbpm_permeability_simulator simulation.db
"""
                first = False
            else:
                command_block = f"""
echo "--- Launching simulation for {folder_base} ---"
cd ../{folder_base}
echo "Current Simulation: " ${{PWD##*/}}
mpirun -np {n_proc} lbpm_permeability_simulator simulation.db
"""

            chunk_content += command_block

        chunk_content += '\necho "--> All simulations in this chunk finished."\n'

        # escrever o script do chunk
        chunk_script_path.write_text(chunk_content, encoding="utf-8")

    # ----- 2. Gerar script dispatcher com encadeamento e prioridade baixa -----
    dispatcher_path = scripts_dir / dispatcher_name

    dispatcher_content = "#!/bin/bash\n\n"

    prev_var = ""

    for idx, chunk_script_name in enumerate(chunk_script_names):
        var_name = f"j{idx+1}"

        if idx == 0:
            # primeiro job: sem dependência, mas com nice alto (baixa prioridade)
            dispatcher_content += (
                f'{var_name}=$(sbatch --parsable --qos=low_prio {chunk_script_name})\n'
            )
            dispatcher_content += (
                f'echo "Submitted {chunk_script_name} as job ${var_name} (QOS=low_prio)"\n\n'
            )
        else:
            # jobs seguintes: dependem do anterior
            dispatcher_content += (
                f'{var_name}=$(sbatch --parsable --qos=low_prio '
                f'--dependency=afterok:${prev_var} {chunk_script_name})\n'
            )
            dispatcher_content += (
                f'echo "Submitted {chunk_script_name} as job ${var_name} '
                f'(QOS=low_prio, afterok:${prev_var})"\n\n'
            )

        prev_var = var_name

    dispatcher_content += 'echo "--> All chunk jobs submitted with low priority and chained dependencies."\n'

    dispatcher_path.write_text(dispatcher_content, encoding="utf-8")

    print(f"[SUCCESS] Generated {len(chunks)} chunk scripts in: {scripts_dir}")
    print(f"[SUCCESS] Generated dispatcher script: {dispatcher_path}")
    print(f"Run: chmod +x {dispatcher_path.name} && ./{dispatcher_path.name}")
    
    
def is_well_resolved(data: NDArray, min_pore_mean, max_pore_mean):
    snow                = ps.filters.snow_partitioning(data)
    network             = ps.networks.regions_to_network(regions=snow.regions)
    pore_diameters      = network['pore.inscribed_diameter'] /2
    throat_diameters    = network['throat.inscribed_diameter'] /2
    print("Mean Pore: ", np.mean(pore_diameters)/2, "; Mean Throat: ", np.mean(throat_diameters)/2)
    return np.mean(pore_diameters)/2 >= min_pore_mean and np.mean(pore_diameters)/2 <= max_pore_mean

def is_percolating(
    data: NDArray,
    axis: int,
) -> Tuple[NDArray, List[int], List[int]]:
    # Data array must be binary and have 1 as pore
    connectivity = 26  # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labeled_components, num_labels = cc3d.connected_components(
        data, connectivity=connectivity, return_N=True)
    if axis == 0:
        labels_inlet = np.unique(labeled_components[0, :, :])
        labels_outlet = np.unique(labeled_components[-1, :, :])
    elif axis == 1:
        labels_inlet = np.unique(labeled_components[:, 0, :])
        labels_outlet = np.unique(labeled_components[:, -1, :])
    elif axis == 2:
        labels_inlet = np.unique(labeled_components[:, :, 0])
        labels_outlet = np.unique(labeled_components[:, :, -1])
    else:
        raise ValueError()

    labels_inlet = set(labels_inlet)
    labels_outlet = set(labels_outlet)
    connected_labels = labels_inlet.intersection(labels_outlet)
    # at least the rock phase (0) will always appear on both inlet and outlet
    
    
    if 0 in connected_labels:    connected_labels.remove(0)
    
    connected_labels = list(connected_labels)

    # get all labels which were pore
    all_labels = set(range(num_labels + 1))
    all_labels.remove(0)

    # get all labels which were pore and are disconnected
    disconnected_labels = all_labels.difference(connected_labels)
    disconnected_labels = list(disconnected_labels)

    return len(connected_labels) > 0


def next_multiple_after(x, multiple_of):
    return int(((x // multiple_of) + 1) * multiple_of)



def generate_slurm_run_script(sample_indices: List[int], n_proc: int, gres: str, output_root: str, script_name: str = "run_all_sims_sequential.sh"):
    """
    Generates a SLURM/Bash script with explicit sequential commands (no loops) 
    for every sample index provided in sample_indices. This style replicates 
    the structure of your original Run_0_2.sh script.
    
    Args:
        sample_indices (List[int]): List of the actual 0-based sample indices (e.g., [0, 1, 2, ...]).
        n_proc (int): Number of MPI tasks/cores to request per simulation (e.g., 4).
        output_root (str): The base directory containing all Sample_ folders.
        script_name (str): The name of the output script file.
    """
    
    # --- 1. SLURM Header and Environment Setup ---
    script_content = f"""#!/bin/bash

# ---------------- SLURM Job Settings ----------------
# NOTE: This script runs all simulations SEQUENTIALLY. It is best for small 
# numbers of samples or if your job queue favors long, single-task jobs.

#SBATCH --oversubscribe
#SBATCH --job-name=Perm_FullRun_Sequential       # Job name for identification
#SBATCH --partition=all_gpu                      # Partition (queue) to submit to: 'k40m', 'a100' or 'a30'
#SBATCH --gres={gres}:{n_proc}                      # Request {n_proc} GPUs (or resources)

#SBATCH -t 7-0:00                              # Max wall time: 7 days (increased for safety)
#SBATCH -o run_outputs_%j.out                  # File to write standard output (%%j = job ID)
#SBATCH -e run_error_%j.err                    # File to write standard error (%%j = job ID)

# ---------------- Environment Setup ----------------

# Load the appropriate module (as suggested by your example script)
module load lbpm/gpu/poro_dev_78ba76

# Change into the root directory where all sample folders reside


# ---------------- Job Execution (Sequential) --------------------


cd DeePore_Samples

"""
    
    # --- 2. Append sequential command block for each sample ---
    for sample_i in sample_indices:
        # Format the folder name with zero-padding (e.g., Sample_00000)
        folder_base = f"Sample_{sample_i:05d}"
        
        
        if sample_i == sample_indices[0]:
            
            command_block = f"""
echo "--- Launching simulation for {folder_base} ---"
cd {folder_base}
echo \"Current Simulation: \"${{PWD##*/}}
mpirun -np {n_proc} lbpm_permeability_simulator simulation.db
# Move back two directories to the root SAMPLE_ROOT directory
"""
        else:
            command_block = f"""
echo "--- Launching simulation for {folder_base} ---"
cd ../{folder_base}
echo \"Current Simulation: \"${{PWD##*/}}
mpirun -np {n_proc} lbpm_permeability_simulator simulation.db

# Move back two directories to the root SAMPLE_ROOT directory
"""
        
        script_content += command_block
    
    script_content += "\n\necho \"--> All sample simulations launched successfully.\""

    # --- 3. Write the script ---
    # Writes the file one directory up from the execution location, assuming this Python script 
    # runs inside the root of your project directory.
    script_path = os.path.join(Path(os.getcwd()).parent, script_name)
    Path(script_path).write_text(script_content, encoding="utf-8")
    print(f"\n[SUCCESS] Generated sequential run script: {script_path}")
    print("Remember to make the script executable: chmod +x run_all_sims_sequential.sh")
    


def order_ceil(value: float) -> float: 
    """ 
    Round up to the nearest power of 10 (ceil-like behavior). 
    Ex: 6.0 -> 10.0, 0.034 -> 0.1, 15.0 -> 100.0, 10.0 -> 10.0
    """ 
    if value <= 0: return 0.0
    log_value = np.log10(value) 
    exponent = math.ceil(log_value) 
    return 10 ** exponent

def timestep_calculation(    
        matriz_binaria: np.ndarray,
        tau: Union[float, int],
        Re: float = 0.01,
        Dens: float = 1.0,
        safety_factor: float = 10.0)  -> int:
    
    L = np.max(matriz_binaria.shape)
    T = int(safety_factor*3*L**2*Dens / (Re*(tau-0.5)))    
    return order_ceil(T)
    
def pressure_calculation(
    matriz_binaria: np.ndarray,
    tau:        Union[float, int],
    Re:         float = 0.01,
    Dens:       float = 1.0,
    )->float:
    
    L               = matriz_binaria.shape[0]
    dist_transform  = distance_transform_edt(matriz_binaria)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)
    Visc            = (tau - 0.5) / 3.0
    dP              = (Re * 8.0 * (Visc ** 2) * L) / (Dens * (R ** 3))

    return dP

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

def write_lbpm_db(
    path: str,
    *,
    db_name:    str = "simulation.db",   # used if `path` is a directory
    bc:         int = 0,
    din:        float = 0.0,
    dout:       float = 0.0,
    fz:         float = 0.0,
    fx:         float = 0.0,
    fy:         float = 0.0,
    tau:        float = 0.9,
    timestep_max: int = 50000,
    tolerance: float = 1e-6,
    # Domain
    domain_filename:str = "domain.raw",
    read_type:      str = "8bit",
    nproc:          Tuple[int, int, int] = (1, 1, 4),
    n:              Tuple[int, int, int] = (256, 256, 128),
    N:              Tuple[int, int, int] = (256, 256, 512),
    offset:         Tuple[int, int, int] = (0, 0, 0),
    voxel_length:   float = 1.0,
    read_values:    Tuple[int, int] = (0, 1),
    write_values:   Tuple[int, int] = (0, 1),
    inlet_layers:   Tuple[int, int, int] = (0, 0, 0),
    outlet_layers:  Tuple[int, int, int] = (0, 0, 0),
    # Visualization
    write_silo:     bool = True,
    save_8bit_raw:  bool = True,
    save_phase_field: bool = True,
    save_pressure:  bool = True,
    save_velocity:  bool = True,
    # Analysis
    analysis_interval:          int = 5000,
    subphase_analysis_interval: int = 5000,
    n_threads:                  int = 0,
    visualization_interval:     int = 5000,
    restart_interval:           int = 100_000_000,
    restart_file:               str = "Restart",
) -> str:
    def tsv3(v): return f"{v[0]}, {v[1]}, {v[2]}"
    def tsv2(v): return f"{v[0]}, {v[1]}"
    def b(v):    return "true" if v else "false"
    def ffmt(x): return f"{x:.6g}"

    text = f"""MRT {{
   tau         = {ffmt(tau)}
   din         = {din}   // inlet density (controls pressure)
   dout        = {dout}  // outlet density (controls pressure)
   F           = {ffmt(fx)}, {ffmt(fy)}, {ffmt(fz)}   // Fx, Fy, Fz
   timestepMax = {timestep_max}
   tolerance   = {ffmt(tolerance)}
}}
Domain {{
   Filename = "{domain_filename}"
   ReadType = "{read_type}"      // data type

   nproc = {tsv3(nproc)}
   n     = {tsv3(n)}
   N     = {tsv3(N)}

   offset         = {tsv3(offset)} // offset to read sub-domain
   voxel_length   = {ffmt(voxel_length)}     // voxel length (in microns)
   ReadValues     = {tsv2(read_values)}    // labels within the original image
   WriteValues    = {tsv2(write_values)}    // associated labels to be used by LBPM (0:solid, 1..N:fluids)
   BC             = {bc}       // boundary condition type (0 for periodic)
   InletLayers    = {tsv3(inlet_layers)}   // specify layers along the inlet
   OutletLayers   = {tsv3(outlet_layers)}  // specify layers along the outlet
}}
Visualization {{
   format            = "vtk"
   write_silo        = {b(write_silo)}     // SILO databases with assigned variables
   save_8bit_raw     = {b(save_8bit_raw)}  // labeled 8-bit binary files with phase assignments
   save_phase_field  = {b(save_phase_field)}  // phase field within SILO database
   save_pressure     = {b(save_pressure)}    // pressure field within SILO database
   save_velocity     = {b(save_velocity)}    // velocity field within SILO database
}}
Analysis {{
   analysis_interval             = {analysis_interval}        // logging interval for timelog.csv
   subphase_analysis_interval    = {subphase_analysis_interval}  // logging interval for subphase.csv
   N_threads                     = {n_threads}                // number of analysis threads (GPU version only)
   visualization_interval        = {visualization_interval}   // interval to write visualization files
   restart_interval              = {restart_interval}         // interval to write restart file
   restart_file                  = "{restart_file}"           // base name of restart file
}}
"""
    p = Path(path)
    # If `path` is a directory or lacks a suffix, write inside it
    if p.suffix == "" or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        p = p / db_name
    else:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(text, encoding="utf-8")
    return text


def create_simulation_pressure_condition(x_sample, output_root, folder_base, n_proc=4):

    x_sample = x_sample.copy() 
    
    # Transform x_sample for simulation:
    x_sample[:, :, 0]   = 0
    x_sample[:, :, -1]  = 0
    x_sample[:, 0, :]   = 0
    x_sample[:, -1, :]  = 0
    
    # Sanity Checks:
    if not is_percolating(x_sample, max_tentativas=50):
        print("Sample failed to percolate and got removed.")
    else:
        
        folder_rbc  = os.path.join(output_root, folder_base)
        os.makedirs(folder_rbc, exist_ok=True)
        
        Re   = 0.1
        tau  = 1.5
        Dens = 1.0
        dP = pressure_calculation(           
                x_sample,
                tau     = tau,
                Re      = Re,
                Dens    = Dens
            )
        
        timestep_max = timestep_calculation(    
                matriz_binaria  =x_sample,
                tau             =tau,
                Re              =Re,
                Dens            =Dens,
                safety_factor   =10.0
                )
        
        # --- Save 3D domain as .raw ---
        write_lbpm_db(path=folder_rbc, 
                      tau       = tau,
                      bc        = 3,
                      din       = 1.0+dP*3,
                      dout      = 1.0,
                      nproc     = (1, 1, n_proc),
                      n         = (x_sample.shape[2], x_sample.shape[1], int(x_sample.shape[0]/n_proc)),
                      N         = (x_sample.shape[2], x_sample.shape[1], x_sample.shape[0]),
                      analysis_interval         =5000, # Excel
                      visualization_interval    =5000, # Silo
                      timestep_max              =timestep_max,
                      subphase_analysis_interval=timestep_max,
                      restart_interval          =timestep_max)
        
        raw_path = os.path.join(folder_rbc, "domain.raw")
        x_sample.astype(np.uint8).tofile(raw_path)
        
        
def create_simulation_force_condition(x_sample, output_root, folder_base, reflect=True, outlet_layers=0, bc=0, n_proc=4):
    
    x_sample = x_sample.copy() 
    
    if x_sample.shape[0]%n_proc!=0: raise Exception(f"Domain length must be divisible by n_proc={n_proc}")
    
    # Transform x_sample for simulation:
    x_sample[:, :, 0]   = 0
    x_sample[:, :, -1]  = 0
    x_sample[:, 0, :]   = 0
    x_sample[:, -1, :]  = 0
    
    
    # Sanity Checks:
    if not is_percolating(x_sample, axis=0): print("Sample failed to percolate and got removed.")
    else:

        
        folder_rbc  = os.path.join(output_root, folder_base)
        os.makedirs(folder_rbc, exist_ok=True)
        
        # Make periodic in z directipn
        if reflect:
            flipped             = np.flip(x_sample, axis=0)
            x_sample            = np.concatenate([x_sample, flipped], axis=0)
        
        # Calculate a force that ensures the desired conditions of Reynolds, Viscosity and Density
        Re   = 0.1
        tau  = 1.5
        Dens = 1.0
        force_z = force_calculation(
            x_sample,
            tau     = tau,
            Re      = Re,
            Dens    = Dens
        )
        
        timestep_max = timestep_calculation(    
                matriz_binaria  = x_sample,
                tau             = tau,
                Re              = Re,
                Dens            = Dens,
                safety_factor   = 10.0
        )
        
        # --- Save 3D domain as .raw ---
        write_lbpm_db(path  =folder_rbc, 
                      tau   =tau,
                      bc    =bc,
                      fz    =force_z,   
                      nproc = (1, 1, n_proc),
                      n     = (x_sample.shape[2], x_sample.shape[1], int(x_sample.shape[0]/n_proc)),
                      N     = (x_sample.shape[2], x_sample.shape[1], x_sample.shape[0]),
                      outlet_layers                 = (0,0,outlet_layers),
                      timestep_max                  = timestep_max,
                      analysis_interval             = 5000, # Excel
                      visualization_interval        = timestep_max, # Silo
                      subphase_analysis_interval    = timestep_max,
                      restart_interval              = timestep_max)
        
        raw_path = os.path.join(folder_rbc, "domain.raw")
        x_sample.astype(np.uint8).tofile(raw_path)
        
            
        # Include guiding image in each folder
        plt.figure()
        plt.imshow(x_sample[0], cmap='binary', interpolation='none')
        plt.axis('off')
        plt.tight_layout(pad=0) 
        plt.savefig(folder_rbc+"/domain.svg", bbox_inches='tight')
        plt.close()
