# NLP HW4 Part-1: Experiment Summary

## Overview
This document summarizes what the automated script will do for Part-1 of the assignment.

## Experiments Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     START EXPERIMENTS                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Q1: Train Original Model                                         │
│ ─────────────────────────────────────────────────────────────   │
│ • Load IMDB dataset (25k train, 25k test)                       │
│ • Train BERT-base-cased for 3 epochs                            │
│ • Evaluate on original test set                                 │
│ • Save model to ./out/                                          │
│                                                                  │
│ Command: python3 main.py --train --eval                         │
│ Output: out_original.txt                                        │
│ Expected Accuracy: >91%                                         │
│ Time: ~45 minutes                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Q2: Evaluate on Transformed Data                                │
│ ─────────────────────────────────────────────────────────────   │
│ • Load model from ./out/                                        │
│ • Transform test set (introduce typos)                          │
│ • Evaluate on transformed test set                              │
│                                                                  │
│ Command: python3 main.py --eval_transformed --model_dir ./out   │
│ Output: out_transformed.txt                                     │
│ Expected: 4-8% accuracy drop                                    │
│ Time: ~5 minutes                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Q3: Train with Data Augmentation                                │
│ ─────────────────────────────────────────────────────────────   │
│ • Original training data: 25k examples                          │
│ • Sample 5k random examples and transform them                  │
│ • Combine: 25k original + 5k transformed = 30k total            │
│ • Train new BERT model for 3 epochs                             │
│ • Evaluate on transformed test set                              │
│ • Save model to ./out_augmented/                                │
│                                                                  │
│ Command: python3 main.py --train_augmented --eval_transformed   │
│ Output: out_augmented_transformed.txt                           │
│ Expected: Better than Q2 on transformed data                    │
│ Time: ~55 minutes                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Q3 (continued): Evaluate Augmented Model on Original Data       │
│ ─────────────────────────────────────────────────────────────   │
│ • Load augmented model from ./out_augmented/                    │
│ • Evaluate on original test set                                 │
│ • Compare with Q1 results                                       │
│                                                                  │
│ Command: python3 main.py --eval --model_dir out_augmented       │
│ Output: out_augmented_original.txt                              │
│ Expected: Slightly lower than Q1                                │
│ Time: ~5 minutes                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Generate Results Summary                                         │
│ ─────────────────────────────────────────────────────────────   │
│ • Extract accuracy from all experiments                         │
│ • Calculate performance changes                                 │
│ • Generate analysis report                                      │
│ • Create submission package                                     │
│                                                                  │
│ Output: results_summary.txt                                     │
│ Time: <1 minute                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTS COMPLETE                          │
│                                                                  │
│ Total Time: ~2 hours                                            │
└─────────────────────────────────────────────────────────────────┘
```

## What Gets Logged

### 1. Main Log (`main_log.txt`)
- Timestamp for each experiment start/end
- Success/failure status
- Extracted accuracy scores
- File operations (copies, moves)

### 2. Experiment Logs (individual `.log` files)
- Full command output
- Training progress bars
- Model loading/saving messages
- Evaluation metrics
- Any errors or warnings

### 3. Results Summary (`results_summary.txt`)
- All accuracy scores in one place
- Performance comparisons:
  - Original vs Transformed
  - Before vs After Augmentation
  - Original Model vs Augmented Model

## File Organization

```
hw4-code/part-1-code/
│
├── main.py                              # Your code
├── utils.py                             # Your transformation
│
├── run_all_experiments.py               # Main runner (Python)
├── run_all_experiments.sh               # Main runner (Bash)
├── submit_job_slurm.sh                  # SLURM submission script
│
├── out/                                 # Q1 model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
│
├── out_augmented/                       # Q3 model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
│
├── submission_package_YYYYMMDD_HHMMSS/  # Ready for Gradescope
│   ├── README.txt
│   ├── results_summary.txt
│   ├── out_original.txt                 # Q1
│   ├── out_transformed.txt              # Q2
│   ├── out_augmented_original.txt       # Q3
│   └── out_augmented_transformed.txt    # Q3
│
├── logs_YYYYMMDD_HHMMSS/                # All logs
│   ├── main_log.txt                     # Master log
│   ├── q1_train_eval_original.log       # Q1 output
│   ├── q2_eval_transformed.log          # Q2 output
│   ├── q3_train_augmented.log           # Q3 training
│   ├── q3_eval_augmented_original.log   # Q3 eval
│   └── results_summary.txt              # Summary
│
└── outputs_YYYYMMDD_HHMMSS/             # Intermediate outputs
    ├── q1_out_original.txt
    ├── q2_out_transformed.txt
    ├── q3_out_augmented_original.txt
    └── q3_out_augmented_transformed.txt
```

## Expected Results

### Q1: Baseline Performance
```
Original Model on Original Test Set
  Accuracy: 0.92XX (92.XX%)
```
**Required**: >91% for full points

### Q2: Robustness Test
```
Original Model on Transformed Test Set
  Accuracy: 0.86XX (86.XX%)

Accuracy drop: 5-6 percentage points
```
**Required**: >4% drop for full points (15/15)

### Q3: Data Augmentation Effect
```
Augmented Model on Transformed Test Set
  Accuracy: 0.88XX (88.XX%)
  Improvement: +2-3 percentage points over Q2

Augmented Model on Original Test Set
  Accuracy: 0.91XX (91.XX%)
  Change: -0.5 to -1.5 percentage points vs Q1
```

## Analysis Questions (for writeup)

The results will help you answer:

1. **Q2**: How much does the transformation degrade performance?
   - Check difference between Q1 and Q2 accuracies

2. **Q3**: Does data augmentation help on transformed data?
   - Compare Q2 vs Q3 on transformed test set

3. **Q3**: What's the trade-off on original data?
   - Compare Q1 vs Q3 on original test set

4. **Q3**: Why does augmentation work/not work?
   - Model sees similar patterns during training
   - Trade-off between robustness and specialization

5. **Q3**: Limitation of this approach?
   - Only helps with seen transformations
   - Doesn't generalize to other OOD patterns
   - Requires knowing what transformations to expect

## Transformation Details

Your implemented transformation ([utils.py:37-86](utils.py#L37-L86)):
- **Type**: Character-level typos
- **Probability**: 15% of words
- **Methods**:
  - Swap adjacent characters
  - Delete a character
  - Duplicate a character
- **Target**: Words >3 characters

## Quick Commands

### Start experiments
```bash
python3 run_all_experiments.py
```

### Submit to SLURM
```bash
sbatch submit_job_slurm.sh
```

### Check progress
```bash
tail -f logs_*/main_log.txt
```

### View results
```bash
cat logs_*/results_summary.txt
```

### Download results
```bash
tar -czf submission.tar.gz submission_package_*/
scp user@hpc:~/path/submission.tar.gz .
```

## Troubleshooting

### Low Q1 Accuracy (<91%)
- Check if training completed all epochs
- Verify GPU is being used
- Check for errors in training loop

### Small accuracy drop in Q2 (<4%)
- Transformation might be too weak
- Increase typo probability in utils.py
- Or make transformations more aggressive

### No improvement in Q3
- Check that augmented data was created correctly
- Verify 5k examples were added
- Check if transformation is same in train/test

### Out of memory
- Reduce batch size (default: 8 → 4)
- Request more GPU memory
- Use smaller subset for debugging

## Next Steps After Completion

1. ✅ Check `experiment_completed.txt`
2. ✅ Review `results_summary.txt`
3. ✅ Download `submission_package_*/` folder
4. ✅ Upload 4 .txt files to Gradescope
5. ✅ Write analysis for Q3 using the results
6. ✅ Save model checkpoints for your records

Good luck! 🎓
