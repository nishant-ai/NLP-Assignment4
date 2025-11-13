# NLP HW4 Part-1: HPC Quick Start Guide

## Quick Start (TL;DR)

```bash
# Make script executable
chmod +x run_all_experiments.py

# Run all experiments (choose one)
python3 run_all_experiments.py   # Recommended - Python version
# OR
bash run_all_experiments.sh      # Bash version

# Check results when you log back in
cat experiment_completed.txt
```

## What This Script Does

The script automatically runs ALL Part-1 experiments:

### Q1: Train and Evaluate Original Model
- Trains BERT on original IMDB data (25k examples)
- Evaluates on original test set
- **Output**: `out_original.txt`
- **Time**: ~45 minutes

### Q2: Evaluate on Transformed Data
- Evaluates Q1 model on transformed test set
- Tests robustness to data transformations (typos)
- **Output**: `out_transformed.txt`
- **Time**: ~5 minutes

### Q3: Train with Data Augmentation
- Trains BERT on original data + 5k transformed examples
- Evaluates on both original and transformed test sets
- **Outputs**: `out_augmented_original.txt`, `out_augmented_transformed.txt`
- **Time**: ~60 minutes

**Total Runtime**: ~2 hours

## Output Structure

After completion, you'll have:

```
submission_package_YYYYMMDD_HHMMSS/
├── README.txt                          # Package description
├── results_summary.txt                 # All accuracy scores + analysis
├── out_original.txt                    # Q1 submission
├── out_transformed.txt                 # Q2 submission
├── out_augmented_original.txt          # Q3 submission
└── out_augmented_transformed.txt       # Q3 submission

logs_YYYYMMDD_HHMMSS/
├── main_log.txt                        # Main execution log
├── q1_train_eval_original.log          # Q1 detailed log
├── q2_eval_transformed.log             # Q2 detailed log
├── q3_train_augmented.log              # Q3 training log
└── q3_eval_augmented_original.log      # Q3 eval log

out/                                    # Q1 model checkpoint
out_augmented/                          # Q3 model checkpoint
```

## Checking Results

### Quick Check
```bash
# Check if completed
cat experiment_completed.txt

# View accuracy summary
cat logs_*/results_summary.txt
```

### Results Summary Format
The `results_summary.txt` contains:
- Individual accuracy scores for all experiments
- Performance drop from transformation
- Improvement from data augmentation
- Comparison between original and augmented models

Example:
```
Q1: Original Model on Original Test Set
  Accuracy: 0.9234 (92.34%)

Q2: Original Model on Transformed Test Set
  Accuracy: 0.8651 (86.51%)

Accuracy drop from transformation: 5.83 percentage points
```

## Submission Files

Upload these files to Gradescope:

**Q1**: `out_original.txt`
**Q2**: `out_transformed.txt`
**Q3**: `out_augmented_original.txt` + `out_augmented_transformed.txt`

All files are in: `submission_package_YYYYMMDD_HHMMSS/`

## Model Checkpoints

The trained models are saved in:
- `./out/` - Original BERT model (Q1)
- `./out_augmented/` - Augmented BERT model (Q3)

These directories contain the full model weights and can be loaded with:
```python
model = AutoModelForSequenceClassification.from_pretrained("./out")
```

## Running on HPC

### Using SLURM (if available)
```bash
#!/bin/bash
#SBATCH --job-name=nlp_hw4_part1
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.log

module load python/3.8
module load cuda/11.3

# Activate your environment
source ~/myenv/bin/activate

# Run experiments
python3 run_all_experiments.py
```

Save as `submit_job.sh` and run:
```bash
sbatch submit_job.sh
```

### Without SLURM
```bash
# Run in background with nohup
nohup python3 run_all_experiments.py > experiment.log 2>&1 &

# Or use screen/tmux
screen -S nlp_hw4
python3 run_all_experiments.py
# Ctrl+A, D to detach
```

## Monitoring Progress

### Check if still running
```bash
# If using SLURM
squeue -u $USER

# If using nohup/background
ps aux | grep run_all_experiments

# Check latest log
tail -f logs_*/main_log.txt
```

### Estimated Progress
- **0-45 min**: Q1 Training + Eval
- **45-50 min**: Q2 Evaluation
- **50-110 min**: Q3 Training + Eval
- **110-115 min**: Q3 Additional Eval
- **115-120 min**: Summary generation

## Troubleshooting

### Check for errors
```bash
# View main log
cat logs_*/main_log.txt

# Check specific experiment
cat logs_*/q1_train_eval_original.log

# Look for failures
grep -i "error\|failed" logs_*/*.log
```

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size in main.py (default is 8)
- Request more GPU memory

**Package not found**:
```bash
pip install transformers datasets torch evaluate
```

**Timeout**:
- Increase time limit in SLURM script
- Use debug mode first: `python3 main.py --train --eval --debug_train`

## Performance Expectations

### Q1 (Original Model)
- Training accuracy: Should reach >91% on test set
- If below 91%, check logs for issues

### Q2 (Transformation)
- Should see accuracy drop of >4% for full points
- Typo transformation typically causes 4-8% drop

### Q3 (Augmentation)
- Should improve performance on transformed data
- May slightly reduce performance on original data
- This trade-off is expected and part of the analysis

## After Completion

1. **Verify outputs exist**:
   ```bash
   ls submission_package_*/
   ```

2. **Check accuracy scores**:
   ```bash
   cat logs_*/results_summary.txt
   ```

3. **Download submission files**:
   ```bash
   # Create tarball
   tar -czf hw4_part1_submission.tar.gz submission_package_*/

   # Download using scp
   scp user@hpc:/path/to/hw4_part1_submission.tar.gz .
   ```

4. **Upload to Gradescope**:
   - Extract tarball locally
   - Upload individual .txt files to Gradescope

## Additional Notes

- All experiments use fixed random seed (seed=0) for reproducibility
- Training uses BERT-base-cased model
- Batch size: 8, Learning rate: 5e-5, Epochs: 3
- The script handles all file copying and organization automatically
- Logs capture everything needed for debugging and analysis

## Questions?

Check the logs:
1. `main_log.txt` - Overall progress
2. `results_summary.txt` - Final scores
3. Individual experiment logs for detailed output

Good luck! 🚀
