# Running T5 Experiments on NYU Greene HPC

## Quick Start

### 1. Submit the Job

```bash
# Navigate to your project directory
cd /scratch/$USER/NLP-A-4/hw4-code/part-2-code

# Submit the batch job
sbatch run_experiments.sbatch
```

You'll see output like:
```
Submitted batch job 12345678
```

### 2. Monitor Your Job

```bash
# Check job status
squeue -u $USER

# Watch the output in real-time
tail -f slurm_experiments_<JOB_ID>.out

# Check latest results
tail -f experiment_results_latest.csv

# View full log
tail -f experiments_log_latest.txt
```

### 3. Check Results

After completion, results will be in:
- `experiment_results_<timestamp>/experiment_results.csv` - Summary table
- `experiment_results_<timestamp>/experiments_log.txt` - Full training logs
- `experiment_results_latest.csv` - Quick access to latest results
- `slurm_experiments_<JOB_ID>.out` - Slurm output

## Important Configuration

### Before Submitting

**Edit `run_experiments.sbatch` and update:**

1. **Email** (line 10):
   ```bash
   #SBATCH --mail-user=YOUR_NETID@nyu.edu
   ```

2. **Working directory** (line 18):
   ```bash
   cd /scratch/$USER/NLP-A-4/hw4-code/part-2-code
   ```
   Make sure this matches your actual path on Greene!

3. **Time limit** (line 6):
   ```bash
   #SBATCH --time=48:00:00  # Adjust if needed
   ```

### Resource Requests

Current settings:
- **GPU**: 1 GPU (RTX 8000 or V100)
- **Memory**: 32GB RAM
- **CPUs**: 4 cores
- **Time**: 48 hours (adjust based on your needs)

Each experiment takes ~30-60 minutes, so 6 experiments should complete in 3-6 hours.

## Useful Slurm Commands

```bash
# Submit job
sbatch run_experiments.sbatch

# Check queue
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# Check job details
scontrol show job <JOB_ID>

# View past jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS

# Check GPU usage (while job is running)
ssh <node_name>
nvidia-smi
```

## Experiment Configuration

The script runs 6 experiments:

| Experiment | Learning Rate | Batch Size | Max Epochs | Patience |
|------------|---------------|------------|------------|----------|
| exp1_lr5e5 | 5e-5 | 16 | 20 | 5 |
| exp2_lr2e5 | 2e-5 | 16 | 20 | 5 |
| exp3_lr1e5 | 1e-5 | 16 | 20 | 5 |
| exp4_lr5e5_bs8 | 5e-5 | 8 | 20 | 5 |
| exp5_lr1e4_patience8 | 1e-4 | 16 | 30 | 8 |
| exp6_lr5e4 | 5e-4 | 16 | 20 | 5 |

## Customizing Experiments

To modify experiments, edit the `run_experiments.sbatch` file:

```bash
# Add a new experiment
run_experiment "my_custom_exp" 3e-5 32 25 6
#              name            lr   bs  epochs patience
```

## Troubleshooting

### Job won't start?
```bash
# Check queue position
squeue -u $USER

# Check partition availability
sinfo -p gpu
```

### Job failed?
```bash
# Check error log
cat slurm_experiments_<JOB_ID>.err

# Check last few lines of output
tail -50 slurm_experiments_<JOB_ID>.out
```

### Out of memory?
Increase memory in the sbatch script:
```bash
#SBATCH --mem=64GB  # or higher
```

### Out of time?
Increase time limit:
```bash
#SBATCH --time=72:00:00  # 72 hours
```

## After Completion

1. **Check results**:
   ```bash
   cat experiment_results_latest.csv
   ```

2. **Find best model**:
   ```bash
   sort -t',' -k2 -nr experiment_results_latest.csv | head -2
   ```

3. **Submit test results**:
   The best model's test predictions are in:
   ```
   results/t5_ft_ft_experiment_test.sql
   records/t5_ft_ft_experiment_test.pkl
   ```

## Tips

- ✓ Always use `/scratch/$USER` for large files and I/O operations
- ✓ Use `--mail-type=END,FAIL` to get email notifications
- ✓ Check `squeue -u $USER` to verify your job is running
- ✓ Use `tail -f` to monitor progress in real-time
- ✓ Keep training runs under 48 hours (Greene limit may vary)
