# Instructions for Running T5 Experiments on Greene HPC

## Step 1: Upload Your Code to Greene

From your local machine:

```bash
# Upload your entire project to Greene
scp -r "/Users/nishant/Desktop/NLP A-4/hw4-code/part-2-code" ns6287@greene.hpc.nyu.edu:/scratch/ns6287/NLP-A-4/hw4-code/

# Or use rsync (faster for updates):
rsync -avz --progress "/Users/nishant/Desktop/NLP A-4/hw4-code/part-2-code/" ns6287@greene.hpc.nyu.edu:/scratch/ns6287/NLP-A-4/hw4-code/part-2-code/
```

## Step 2: SSH into Greene

```bash
ssh ns6287@greene.hpc.nyu.edu
```

## Step 3: Navigate to Your Project Directory

```bash
cd /scratch/ns6287/NLP-A-4/hw4-code/part-2-code
```

## Step 4: Verify Your Setup

Make sure your paths are correct:

```bash
# Check if overlay exists
ls -lh /scratch/ns6287/overlay-25GB-500K.ext3

# Check if Singularity image exists
ls -lh /scratch/ns6287/ubuntu-20.04.3.sif

# Check if your code is here
ls -la
```

## Step 5: Submit the Job

```bash
sbatch run_experiments.sbatch
```

You should see:
```
Submitted batch job 12345678
```

## Step 6: Monitor Your Job

### Check if job is running:
```bash
squeue -u ns6287
```

Output will look like:
```
JOBID    PARTITION     NAME     USER  ST       TIME  NODES
12345678 c12m85-a100-1 t5_sql_exp ns6287  R      5:23      1
```

ST = Status:
- `PD` = Pending (waiting for resources)
- `R` = Running
- `CG` = Completing

### Watch the output in real-time:
```bash
# Find your output file (named with job ID)
ls -lt *.out | head -1

# Watch it live
tail -f 12345678_t5_sql_exp.out
```

### Check results as they come in:
```bash
# Watch the CSV results file
tail -f experiment_results_latest.csv

# Watch detailed logs
tail -f experiments_log_latest.txt
```

## Step 7: After Job Completes

### View final results:
```bash
cat experiment_results_latest.csv
```

### Find the best experiment:
```bash
# Sort by F1 score (column 2)
tail -n +2 experiment_results_latest.csv | sort -t',' -k2 -nr | head -1
```

### Check which models were saved:
```bash
ls -R checkpoints/ft_experiments/
```

## Step 8: Download Results to Your Local Machine

From your local machine:

```bash
# Download results
scp ns6287@greene.hpc.nyu.edu:/scratch/ns6287/NLP-A-4/hw4-code/part-2-code/experiment_results_latest.csv ~/Desktop/

# Download test predictions (for submission)
scp ns6287@greene.hpc.nyu.edu:/scratch/ns6287/NLP-A-4/hw4-code/part-2-code/results/t5_ft_ft_experiment_test.sql ~/Desktop/
scp ns6287@greene.hpc.nyu.edu:/scratch/ns6287/NLP-A-4/hw4-code/part-2-code/records/t5_ft_ft_experiment_test.pkl ~/Desktop/

# Or download everything
scp -r ns6287@greene.hpc.nyu.edu:/scratch/ns6287/NLP-A-4/hw4-code/part-2-code/results ~/Desktop/
```

## Useful Commands

### Cancel a job:
```bash
scancel 12345678
```

### Check job details:
```bash
scontrol show job 12345678
```

### View past jobs:
```bash
sacct -u ns6287 --format=JobID,JobName,State,Elapsed,MaxRSS
```

### Check GPU usage (if job is running):
```bash
# First find which node your job is on
squeue -u ns6287

# Then SSH to that node
ssh c12m85-a100-1-XXX  # replace with actual node name

# Check GPU
nvidia-smi

# Exit node
exit
```

## Troubleshooting

### Job pending for too long?
The partition might be busy. Check:
```bash
sinfo -p c12m85-a100-1
```

### Job failed immediately?
Check the error file:
```bash
cat 12345678_t5_sql_exp.err
```

### Out of memory?
Edit `run_experiments.sbatch` and add:
```bash
#SBATCH --mem=64GB
```

### Need more time?
Edit `run_experiments.sbatch` and change:
```bash
#SBATCH --time=48:00:00
```

## What the Script Does

Runs **6 experiments** automatically:

1. `exp1_lr5e5` - LR=5e-5, BS=16, 20 epochs
2. `exp2_lr2e5` - LR=2e-5, BS=16, 20 epochs
3. `exp3_lr1e5` - LR=1e-5, BS=16, 20 epochs
4. `exp4_lr5e5_bs8` - LR=5e-5, BS=8, 20 epochs
5. `exp5_lr1e4_patience8` - LR=1e-4, BS=16, 30 epochs
6. `exp6_lr5e4` - LR=5e-4, BS=16, 20 epochs

Each experiment takes ~30-60 minutes.
Total runtime: ~3-6 hours.

## Output Files

- `12345678_t5_sql_exp.out` - Main output log
- `12345678_t5_sql_exp.err` - Error log (should be empty)
- `experiment_results_latest.csv` - Summary table
- `experiments_log_latest.txt` - Detailed training logs
- `experiment_results_YYYYMMDD_HHMMSS/` - Timestamped results directory
- `checkpoints/ft_experiments/exp1_lr5e5/` - Saved models
- `results/t5_ft_ft_experiment_test.sql` - Final test predictions

## Tips

✓ The job runs completely independently - you can log out!
✓ You'll get an email when it finishes
✓ Use `tail -f` to watch progress in real-time
✓ The best model is automatically selected based on dev F1
✓ Test predictions are generated automatically at the end
