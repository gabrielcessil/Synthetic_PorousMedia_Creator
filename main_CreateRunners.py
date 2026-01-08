import utils

chunk_size      = 5   # Set for 1h of simulations (5 samples, 20 min per sample)
gres            = "gpu:k40m" #"gpu:k40m"#"gpu:a100"
n_proc          = 1
cpu             = 12 
gpu             = 64

output_root     = "./Test_SphPore_120_120_120/"
n_samples       = 60


indices     = list(range(0, n_samples+1))
utils.generate_slurm_run_scripts_chunks(
    indices,
    n_proc,
    gres,
    output_root,
    chunk_size,
    cpu, 
    gpu,
    f"Run_{len(indices)}_samples.sh")