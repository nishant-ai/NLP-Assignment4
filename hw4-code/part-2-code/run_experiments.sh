#!/bin/bash

# ============================================================
# T5 Fine-tuning Baseline Experiments
# ============================================================
# This script runs 5 baseline experiments with different
# hyperparameters to find the optimal configuration.
#
# Goal: Achieve ≥65 F1 on test set for full credit
# ============================================================

# Create experiment directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="baseline_experiments_${TIMESTAMP}"
mkdir -p $EXP_DIR

LOG_FILE="$EXP_DIR/experiments_log.txt"
RESULTS_CSV="$EXP_DIR/results_summary.csv"

# ============================================================
# SETUP LOGGING
# ============================================================

echo "===========================================================" | tee $LOG_FILE
echo "T5 BASELINE FINE-TUNING EXPERIMENTS" | tee -a $LOG_FILE
echo "===========================================================" | tee -a $LOG_FILE
echo "Started: $(date)" | tee -a $LOG_FILE
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')" | tee -a $LOG_FILE
echo "===========================================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# CSV header
echo "Experiment,LR,BatchSize,Epochs,Patience,RecordF1,RecordEM,SQLEM,DevLoss,ErrorRate,Runtime_mins,Status" > $RESULTS_CSV

# ============================================================
# EXPERIMENT RUNNER FUNCTION
# ============================================================

run_experiment() {
    local exp_name=$1
    local learning_rate=$2
    local batch_size=$3
    local max_epochs=$4
    local patience=$5
    local scheduler=$6
    local warmup=$7

    START_TIME=$(date +%s)

    echo "" | tee -a $LOG_FILE
    echo "==========================================================" | tee -a $LOG_FILE
    echo "EXPERIMENT: $exp_name" | tee -a $LOG_FILE
    echo "==========================================================" | tee -a $LOG_FILE
    echo "Configuration:" | tee -a $LOG_FILE
    echo "  Learning Rate: $learning_rate" | tee -a $LOG_FILE
    echo "  Batch Size: $batch_size" | tee -a $LOG_FILE
    echo "  Max Epochs: $max_epochs" | tee -a $LOG_FILE
    echo "  Patience: $patience" | tee -a $LOG_FILE
    echo "  Scheduler: $scheduler" | tee -a $LOG_FILE
    echo "  Warmup Epochs: $warmup" | tee -a $LOG_FILE
    echo "  Started: $(date)" | tee -a $LOG_FILE
    echo "==========================================================" | tee -a $LOG_FILE

    # Run training
    python train_t5.py \
        --finetune \
        --learning_rate $learning_rate \
        --weight_decay 0.01 \
        --batch_size $batch_size \
        --test_batch_size 32 \
        --max_n_epochs $max_epochs \
        --patience_epochs $patience \
        --scheduler_type $scheduler \
        --num_warmup_epochs $warmup \
        --optimizer_type AdamW \
        --experiment_name $exp_name \
        2>&1 | tee -a $LOG_FILE

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    RUNTIME=$(( ($END_TIME - $START_TIME) / 60 ))

    # Extract results from log
    if [ $EXIT_CODE -eq 0 ]; then
        # Get the final dev results (last occurrence)
        local results_line=$(grep "Record F1:" $LOG_FILE | grep "Record EM:" | tail -1)

        if [ ! -z "$results_line" ]; then
            # Extract metrics using grep/sed
            local record_f1=$(echo "$results_line" | grep -oP 'Record F1: \K[0-9.]+' || echo "N/A")
            local record_em=$(echo "$results_line" | grep -oP 'Record EM: \K[0-9.]+' || echo "N/A")
            local sql_em=$(echo "$results_line" | grep -oP 'SQL EM: \K[0-9.]+' || echo "N/A")
            local dev_loss=$(echo "$results_line" | grep -oP 'Loss: \K[0-9.]+' || echo "N/A")

            # Get error rate
            local error_line=$(grep "% of the generated outputs led to SQL errors" $LOG_FILE | tail -1)
            local error_rate=$(echo "$error_line" | grep -oP '[0-9.]+(?=%)' || echo "N/A")

            local status="SUCCESS"
        else
            local record_f1="N/A"
            local record_em="N/A"
            local sql_em="N/A"
            local dev_loss="N/A"
            local error_rate="N/A"
            local status="NO_RESULTS"
        fi
    else
        local record_f1="ERROR"
        local record_em="ERROR"
        local sql_em="ERROR"
        local dev_loss="ERROR"
        local error_rate="ERROR"
        local status="FAILED"
    fi

    # Save to CSV
    echo "$exp_name,$learning_rate,$batch_size,$max_epochs,$patience,$record_f1,$record_em,$sql_em,$dev_loss,$error_rate,$RUNTIME,$status" >> $RESULTS_CSV

    # Print summary
    echo "" | tee -a $LOG_FILE
    echo "-----------------------------------------------------------" | tee -a $LOG_FILE
    echo "EXPERIMENT COMPLETED: $exp_name" | tee -a $LOG_FILE
    echo "Runtime: ${RUNTIME} minutes" | tee -a $LOG_FILE
    echo "Record F1: $record_f1 | Record EM: $record_em | SQL EM: $sql_em" | tee -a $LOG_FILE
    echo "Dev Loss: $dev_loss | Error Rate: $error_rate%" | tee -a $LOG_FILE
    echo "Status: $status" | tee -a $LOG_FILE
    echo "-----------------------------------------------------------" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

# ============================================================
# BASELINE EXPERIMENTS
# ============================================================
#
# Strategy: Test different learning rates and batch sizes
# Based on the PDF recommendation to vary hyperparameters
#
# Expected best range: LR between 1e-4 and 5e-4
# ============================================================

echo "Starting 5 Baseline Experiments..." | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Experiment 1: Standard baseline (most likely to work)
# WHY: 3e-4 is the sweet spot for T5-small fine-tuning
run_experiment "baseline_lr3e4" 3e-4 16 15 3 "cosine" 1

# Experiment 2: Higher LR for faster convergence
# WHY: May reach good performance quicker
run_experiment "baseline_lr5e4" 5e-4 16 15 3 "cosine" 1

# Experiment 3: Even higher LR (aggressive but might work)
# WHY: Test if faster learning helps with this task
run_experiment "baseline_lr1e3" 1e-3 16 15 3 "cosine" 1

# Experiment 4: Conservative with longer training
# WHY: Lower LR but more patience for stable convergence
run_experiment "baseline_lr1e4_long" 1e-4 16 20 5 "cosine" 2

# Experiment 5: Larger batch with medium LR
# WHY: Batch 32 with 3e-4 balances stability and speed
run_experiment "baseline_bs32_lr3e4" 3e-4 32 15 3 "cosine" 1

# ============================================================
# GENERATE FINAL SUMMARY
# ============================================================

echo "" | tee -a $LOG_FILE
echo "===========================================================" | tee -a $LOG_FILE
echo "ALL EXPERIMENTS COMPLETED" | tee -a $LOG_FILE
echo "===========================================================" | tee -a $LOG_FILE
echo "Completed: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "RESULTS SUMMARY:" | tee -a $LOG_FILE
echo "-----------------------------------------------------------" | tee -a $LOG_FILE
column -t -s',' $RESULTS_CSV | tee -a $LOG_FILE
echo "-----------------------------------------------------------" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Find best experiment by Record F1
echo "BEST EXPERIMENT BY RECORD F1:" | tee -a $LOG_FILE
tail -n +2 $RESULTS_CSV | sort -t',' -k6 -nr | head -1 | column -t -s',' | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Check if any experiment achieved target F1
echo "EXPERIMENTS ACHIEVING F1 ≥ 0.65 (full credit threshold):" | tee -a $LOG_FILE
tail -n +2 $RESULTS_CSV | awk -F',' '$6 >= 0.65 {print}' | column -t -s',' | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ============================================================
# SAVE RESULTS
# ============================================================

# Copy to easy-access location
cp $RESULTS_CSV baseline_results_latest.csv
cp $LOG_FILE baseline_log_latest.txt

echo "Results saved to:" | tee -a $LOG_FILE
echo "  Main: $EXP_DIR/" | tee -a $LOG_FILE
echo "  Latest CSV: baseline_results_latest.csv" | tee -a $LOG_FILE
echo "  Latest Log: baseline_log_latest.txt" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "To view results:" | tee -a $LOG_FILE
echo "  cat baseline_results_latest.csv" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "===========================================================" | tee -a $LOG_FILE
echo "EXPERIMENTS COMPLETED SUCCESSFULLY" | tee -a $LOG_FILE
echo "===========================================================" | tee -a $LOG_FILE
