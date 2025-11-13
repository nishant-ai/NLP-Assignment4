# Part 2: T5 Fine-tuning - Simple Run Guide

## Setup

```bash
cd "/Users/nishant/Desktop/NLP A-4/hw4-code/part-2-code"
pip install -r requirements.txt
```

## Quick Start - Run One Experiment

```bash
python train_t5.py --finetune --max_n_epochs 10 --learning_rate 1e-4 \
    --batch_size 16 --patience_epochs 3 --experiment_name my_first_run
```

## Hyperparameter Tuning

### Try Different Learning Rates

```bash
# Small learning rate (more stable)
python train_t5.py --finetune --max_n_epochs 10 --learning_rate 5e-5 \
    --batch_size 16 --patience_epochs 3 --experiment_name lr_5e5

# Medium learning rate (recommended baseline)
python train_t5.py --finetune --max_n_epochs 10 --learning_rate 1e-4 \
    --batch_size 16 --patience_epochs 3 --experiment_name lr_1e4

# Large learning rate (faster but risky)
python train_t5.py --finetune --max_n_epochs 10 --learning_rate 1e-3 \
    --batch_size 16 --patience_epochs 3 --experiment_name lr_1e3
```

### Try Different Batch Sizes

```bash
# Small batch (slower, more memory efficient)
python train_t5.py --finetune --max_n_epochs 10 --learning_rate 1e-4 \
    --batch_size 8 --patience_epochs 3 --experiment_name bs_8

# Medium batch (recommended)
python train_t5.py --finetune --max_n_epochs 10 --learning_rate 1e-4 \
    --batch_size 16 --patience_epochs 3 --experiment_name bs_16

# Large batch (faster, needs more memory)
python train_t5.py --finetune --max_n_epochs 10 --learning_rate 1e-4 \
    --batch_size 32 --patience_epochs 3 --experiment_name bs_32
```

## What to Report

After each run, note down:
- **Record F1** (main metric)
- **Record EM**
- **SQL EM**
- **Error Rate**

Pick the experiment with the highest Record F1.

## Output Files

Results saved in:
- `checkpoints/ft_experiments/[experiment_name]/` - Model checkpoints
- `results/t5_ft_ft_experiment_dev.sql` - Generated SQL for dev set
- `results/t5_ft_ft_experiment_test.sql` - Generated SQL for test set
- `records/` - Database execution results

## Tips

- Start with the baseline (LR=1e-4, BS=16)
- Try 3-4 different configurations
- Pick the best one based on dev set Record F1
- The test results are automatically generated at the end

That's it!
