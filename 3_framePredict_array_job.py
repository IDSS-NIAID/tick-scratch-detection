import os
import sys
import subprocess
import argparse
from datetime import datetime

def create_slurm_array_script(root_folder, subfolders, output_dir):
    """Generate a SLURM bash script for a job array."""
    num_tasks = len(subfolders)
    job_name = f"framePredict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create a quoted, newline-separated list of subfolders for Bash to read safely
    subfolders_str = "\n".join([f'"{os.path.join(root_folder, f)}"' for f in subfolders])
    
    slurm_script = f"""#!/bin/bash --login
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/{job_name}_%A_%a.out
#SBATCH --error={output_dir}/{job_name}_%A_%a.err
#SBATCH --array=0-{num_tasks-1}   # Array range based on number of subfolders
#SBATCH --time=24:00:00           
#SBATCH --ntasks=1                # One task per array element
#SBATCH --cpus-per-task=6         
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G                         

module load miniconda3
conda activate tick

# Read subfolders into an array, preserving whitespace
mapfile -t SUBFOLDERS <<EOF
{subfolders_str}
EOF

# Get the folder for this task using the array index
FOLDER=${{SUBFOLDERS[$SLURM_ARRAY_TASK_ID]}}

# Run the processing script for this folder
python framePredict_single_folder.py --input "$FOLDER"
"""
    script_path = os.path.join(output_dir, f"{job_name}.sbatch")
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    return script_path, job_name

def submit_array_job(root_folder):
    """Submit a SLURM job array for all subfolders in the root folder."""
    # Ensure root folder exists
    if not os.path.isdir(root_folder):
        print(f"Error: '{root_folder}' is not a valid directory.")
        sys.exit(1)

    # Create an output directory for logs and SLURM scripts
    output_dir = os.path.join(root_folder, "slurm_logs")
    os.makedirs(output_dir, exist_ok=True)

    # Get list of subfolders
    subfolders = [f for f in os.listdir(root_folder) 
                  if os.path.isdir(os.path.join(root_folder, f)) 
                  and f != "slurm_logs"]  # Exclude the logs folder

    if not subfolders:
        print(f"No subfolders found in '{root_folder}'.")
        sys.exit(1)

    print(f"Found {len(subfolders)} subfolders to process in array.")

    # Create and submit the job array script
    slurm_script_path, job_name = create_slurm_array_script(root_folder, subfolders, output_dir)
    
    try:
        result = subprocess.run(['sbatch', slurm_script_path], 
                              check=True, capture_output=True, text=True)
        print(f"Submitted job array '{job_name}': {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job array: {e.stderr}")

if __name__ == "__main__":
    # Parse command-line argument for root folder
    parser = argparse.ArgumentParser(description="Submit a SLURM job array for subfolders.")
    parser.add_argument("--root", required=True, help="Path to the root folder containing subfolders")
    args = parser.parse_args()

    # Submit job array
    submit_array_job(args.root)
