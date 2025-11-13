#!/bin/bash
#SBATCH --job-name=nlp_hw4_part1
#SBATCH --time=03:00:00              # 3 hours should be enough
#SBATCH --mem=16G                     # 16GB RAM
#SBATCH --gres=gpu:1                  # 1 GPU
#SBATCH --cpus-per-task=4             # 4 CPU cores
#SBATCH --output=slurm_%j.log         # Log file with job ID
#SBATCH --error=slurm_%j.err          # Error file with job ID

# Uncomment and modify these based on your HPC setup
# #SBATCH --partition=gpu              # GPU partition name
# #SBATCH --account=your_account       # Your account/project
# #SBATCH --mail-type=END,FAIL         # Email notifications
# #SBATCH --mail-user=your_email       # Your email

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "========================================="

# Load modules (modify based on your HPC)
# Example for common HPC setups:
# module load python/3.8
# module load cuda/11.3
# module load cudnn/8.2

# Activate virtual environment (uncomment and modify if needed)
# source ~/myenv/bin/activate
# OR
# conda activate nlp_env

# Print Python and CUDA info
echo "Python version:"
python3 --version
echo ""

echo "PyTorch CUDA available:"
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Navigate to directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo "========================================="
echo ""

# Run the experiment
echo "Starting experiments..."
python3 run_all_experiments.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Experiments completed successfully!"
    echo "Finished at: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "Experiments failed! Check logs for details."
    echo "Failed at: $(date)"
    echo "========================================="
    exit 1
fi
