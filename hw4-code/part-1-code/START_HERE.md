# 🚀 NLP HW4 Part-1: Automated Experiment Runner

## TL;DR - Run This Command

```bash
python3 run_all_experiments.py
```

That's it! The script will:
- ✅ Run all 3 questions (Q1, Q2, Q3)
- ✅ Generate all required output files
- ✅ Calculate accuracy scores and analysis
- ✅ Package everything for Gradescope submission
- ⏱️ Total time: ~2 hours

---

## What You Get

### 📦 Submission Package (ready for Gradescope)
```
submission_package_YYYYMMDD_HHMMSS/
├── out_original.txt                    ← Upload for Q1
├── out_transformed.txt                 ← Upload for Q2
├── out_augmented_original.txt          ← Upload for Q3
├── out_augmented_transformed.txt       ← Upload for Q3
└── results_summary.txt                 ← Use for your writeup
```

### 📊 Results Summary
Automatically calculated for you:
- Q1 accuracy on original data
- Q2 accuracy on transformed data (shows robustness)
- Accuracy drop from transformation
- Q3 improvements from data augmentation
- Trade-offs between original and augmented models

### 💾 Model Checkpoints
- `./out/` - Original BERT model (Q1)
- `./out_augmented/` - Augmented BERT model (Q3)

---

## Quick Start Guide

### Option 1: Run Locally/Interactively
```bash
python3 run_all_experiments.py
```

### Option 2: Run on HPC with SLURM
```bash
# Edit submit_job_slurm.sh to match your HPC setup
# (uncomment and modify module loads, paths, etc.)

sbatch submit_job_slurm.sh

# Check status
squeue -u $USER

# Check progress
tail -f slurm_*.log
```

### Option 3: Run in Background (no SLURM)
```bash
nohup python3 run_all_experiments.py > experiment.log 2>&1 &

# Or use screen/tmux
screen -S nlp_hw4
python3 run_all_experiments.py
# Ctrl+A, D to detach
```

---

## Files Created for You

| File | Purpose |
|------|---------|
| **run_all_experiments.py** | Main experiment runner (Python - recommended) |
| **run_all_experiments.sh** | Main experiment runner (Bash alternative) |
| **submit_job_slurm.sh** | SLURM job submission script |
| **HPC_QUICK_START.md** | Detailed HPC usage guide |
| **EXPERIMENT_SUMMARY.md** | Visual flow of all experiments |
| **START_HERE.md** | This file! |

---

## Timeline & Progress

```
 0 min  ━━━━━━━━━━━━━━━━━━━┓
                            ┃  Q1: Training original model
45 min  ━━━━━━━━━━━━━━━━━━━┛
                            ┃  Q2: Eval on transformed data
50 min  ━━━━━━━━━━━━━━━━━━━┛
                            ┃  Q3: Training augmented model
105 min ━━━━━━━━━━━━━━━━━━━┛
                            ┃  Q3: Eval on original data
110 min ━━━━━━━━━━━━━━━━━━━┛
                            ┃  Generate summary & package
115 min ━━━━━━━━━━━━━━━━━━━┛  DONE!
```

---

## Checking Results When You Log Back In

### 1. Check if completed
```bash
cat experiment_completed.txt
```

### 2. View accuracy summary
```bash
cat logs_*/results_summary.txt
```

### 3. Find submission files
```bash
ls submission_package_*/
```

Example output:
```
Q1: Original Model on Original Test Set
  Accuracy: 0.9234 (92.34%)

Q2: Original Model on Transformed Test Set
  Accuracy: 0.8651 (86.51%)

Accuracy drop from transformation: 5.83 percentage points
  ↑ Use this for your Q2 analysis

Q3: Augmented Model on Transformed Test Set
  Accuracy: 0.8834 (88.34%)

Improvement from augmentation: 1.83 percentage points
  ↑ Use this for your Q3 analysis

Q3: Augmented Model on Original Test Set
  Accuracy: 0.9187 (91.87%)

Effect of augmentation on original data: -0.47 percentage points
  ↑ Discuss this trade-off in Q3
```

---

## What Each Question Does

### Q1: Train and Evaluate Original Model (45 min)
- Trains BERT on 25k IMDB reviews
- Evaluates on original test set
- **Target**: >91% accuracy
- **Deliverable**: `out_original.txt`

### Q2: Test Robustness with Transformations (5 min)
- Applies typo transformation to test data
- Evaluates Q1 model on noisy data
- **Target**: >4% accuracy drop (more = better for this Q!)
- **Deliverable**: `out_transformed.txt`

### Q3: Improve with Data Augmentation (60 min)
- Augments training data with 5k transformed examples
- Trains new model on 30k examples (25k original + 5k augmented)
- Evaluates on both original and transformed test sets
- **Goal**: Better on transformed, understand trade-offs
- **Deliverables**: `out_augmented_original.txt`, `out_augmented_transformed.txt`

---

## Requirements

### Python Packages
```bash
pip install transformers datasets torch evaluate
```

### Hardware
- GPU with 8GB+ VRAM (16GB recommended)
- 16GB RAM
- 10GB disk space for models and data

### Expected Performance
- Q1: >91% accuracy required
- Q2: >4% drop for full points
- Q3: Analyzed in writeup

---

## Troubleshooting

### ❌ "CUDA out of memory"
Edit [main.py:189](main.py#L189), change batch size:
```python
parser.add_argument("--batch_size", type=int, default=4)  # was 8
```

### ❌ "Can't find module transformers"
```bash
pip install transformers datasets torch evaluate tqdm
```

### ❌ Q1 accuracy < 91%
- Check training completed all 3 epochs
- Verify GPU is being used
- Check logs for errors: `cat logs_*/q1_train_eval_original.log`

### ❌ Q2 accuracy drop < 4%
Your transformation is too weak. Edit [utils.py:50](utils.py#L50):
```python
typo_probability = 0.20  # Increase from 0.15
```

### ❌ Script hangs or stops
```bash
# Find the process
ps aux | grep run_all_experiments

# Check GPU usage
nvidia-smi

# Check logs for errors
tail -f logs_*/main_log.txt
```

---

## Submission Checklist

- [ ] Run experiments: `python3 run_all_experiments.py`
- [ ] Verify completion: `cat experiment_completed.txt`
- [ ] Check accuracies: `cat logs_*/results_summary.txt`
- [ ] Q1 accuracy >91%? ✓
- [ ] Q2 drop >4%? ✓
- [ ] Find submission files: `ls submission_package_*/`
- [ ] Download submission package
- [ ] Upload to Gradescope:
  - [ ] Q1: `out_original.txt`
  - [ ] Q2: `out_transformed.txt`
  - [ ] Q3: `out_augmented_original.txt`
  - [ ] Q3: `out_augmented_transformed.txt`
- [ ] Write analysis using `results_summary.txt`
- [ ] Include GitHub link and Google Drive link in writeup

---

## Additional Resources

📖 **For detailed explanations**:
- [HPC_QUICK_START.md](HPC_QUICK_START.md) - HPC-specific instructions
- [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) - Visual experiment flow

💡 **For debugging**:
- Check `logs_*/main_log.txt` - Overall progress
- Check `logs_*/q*.log` - Individual experiment details
- Check `slurm_*.log` (if using SLURM)

---

## Notes

- ⚠️ **Random seed is fixed** (seed=0) for reproducibility
- 📊 All experiments use identical hyperparameters
- 🔄 The script is **idempotent** - safe to re-run
- 💾 Model checkpoints are saved automatically
- 📝 All outputs are timestamped
- 🎯 Submission files are auto-organized

---

## Need Help?

1. Check the logs in `logs_*/`
2. Review [HPC_QUICK_START.md](HPC_QUICK_START.md)
3. Review [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)
4. Check assignment PDF for requirements

---

## Summary

**Single command** → **Complete experiments** → **Ready for submission**

```bash
python3 run_all_experiments.py
```

Then when you log back in:

```bash
cat logs_*/results_summary.txt
ls submission_package_*/
```

**That's it!** 🎉

Good luck with your assignment!
