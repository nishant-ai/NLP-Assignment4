#!/bin/bash

#SBATCH --job-name=t5_sql_exp
#SBATCH --account=ds_ga_1011_001-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ns6287@nyu.edu

# Run experiments inside Singularity container
singularity exec --bind /scratch --nv --overlay /scratch/ns6287/overlay-25GB-500K.ext3:rw /scratch/ns6287/ubuntu-20.04.3.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate hw4-part-2-nlp
cd /scratch/ns6287/NLP-A-4/hw4-code/part-2-code

# Create results directory
RESULTS_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

LOG_FILE="$RESULTS_DIR/experiments_log.txt"
RESULTS_FILE="$RESULTS_DIR/experiment_results.csv"

# Start logging
echo "==================================================" > $LOG_FILE
echo "T5 Fine-tuning Experiments - Started at $(date)" >> $LOG_FILE
echo "Job ID: $SLURM_JOB_ID" >> $LOG_FILE
echo "Node: $SLURM_NODELIST" >> $LOG_FILE
echo "GPU: $CUDA_VISIBLE_DEVICES" >> $LOG_FILE
echo "==================================================" >> $LOG_FILE
echo "" >> $LOG_FILE

# CSV header
echo "Experiment,Record_F1,Record_EM,SQL_EM,Dev_Loss,Error_Rate,Runtime_mins" > $RESULTS_FILE

# Function to run an experiment
run_experiment() {
    local exp_name=$1
    local lr=$2
    local bs=$3
    local epochs=$4
    local patience=$5

    START_TIME=$(date +%s)

    echo "" | tee -a $LOG_FILE
    echo "================================================" | tee -a $LOG_FILE
    echo "Experiment: $exp_name" | tee -a $LOG_FILE
    echo "LR=$lr, BS=$bs, Epochs=$epochs, Patience=$patience" | tee -a $LOG_FILE
    echo "Started: $(date)" | tee -a $LOG_FILE
    echo "================================================" | tee -a $LOG_FILE

    # Run training
    python train_t5.py --finetune \
        --max_n_epochs $epochs \
        --learning_rate $lr \
        --batch_size $bs \
        --patience_epochs $patience \
        --experiment_name $exp_name \
        2>&1 | tee -a $LOG_FILE

    END_TIME=$(date +%s)
    RUNTIME=$(( ($END_TIME - $START_TIME) / 60 ))

    # Extract results from the last "Dev set results" line
    local last_result=$(grep "Dev set results: Loss:" $LOG_FILE | tail -1)

    if [ ! -z "$last_result" ]; then
        local f1=$(echo "$last_result" | grep -oP 'Record F1: \K[0-9.]+' || echo "N/A")
        local em=$(echo "$last_result" | grep -oP 'Record EM: \K[0-9.]+' || echo "N/A")
        local sql_em=$(echo "$last_result" | grep -oP 'SQL EM: \K[0-9.]+' || echo "N/A")
        local loss=$(echo "$last_result" | grep -oP 'Loss: \K[0-9.]+' || echo "N/A")

        local error_line=$(grep "Dev set results:.*% of the generated outputs led to SQL errors" $LOG_FILE | tail -1)
        local error_rate=$(echo "$error_line" | grep -oP '[0-9.]+(?=% of the generated)' || echo "N/A")

        echo "$exp_name,$f1,$em,$sql_em,$loss,$error_rate,$RUNTIME" >> $RESULTS_FILE

        echo "" | tee -a $LOG_FILE
        echo "✓ Completed: $exp_name (${RUNTIME}m)" | tee -a $LOG_FILE
        echo "  F1=$f1, EM=$em, SQL_EM=$sql_em, Loss=$loss, Error=$error_rate%" | tee -a $LOG_FILE
    else
        echo "$exp_name,ERROR,ERROR,ERROR,ERROR,ERROR,$RUNTIME" >> $RESULTS_FILE
        echo "✗ ERROR: Could not extract results for $exp_name" | tee -a $LOG_FILE
    fi
}

# ========================================
# EXPERIMENT CONFIGURATIONS
# ========================================

echo ""
echo "Starting 6 experiments..."
echo ""

# Experiment 1: Lower learning rate
run_experiment "exp1_lr5e5" 5e-5 16 20 5

# Experiment 2: Even lower learning rate
run_experiment "exp2_lr2e5" 2e-5 16 20 5

# Experiment 3: Very low learning rate
run_experiment "exp3_lr1e5" 1e-5 16 20 5

# Experiment 4: Lower LR with smaller batch
run_experiment "exp4_lr5e5_bs8" 5e-5 8 20 5

# Experiment 5: Original LR with more patience
run_experiment "exp5_lr1e4_patience8" 1e-4 16 30 8

# Experiment 6: Slightly higher LR
run_experiment "exp6_lr5e4" 5e-4 16 20 5

# ========================================
# FINAL SUMMARY
# ========================================

echo "" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE
echo "ALL EXPERIMENTS COMPLETED at $(date)" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "Results Summary:" | tee -a $LOG_FILE
column -t -s',' $RESULTS_FILE | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "Best experiment by Record F1:" | tee -a $LOG_FILE
tail -n +2 $RESULTS_FILE | sort -t',' -k2 -nr | head -1 | column -t -s',' | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "Results saved to: $RESULTS_DIR" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Copy results to main directory for easy access
cp $RESULTS_FILE experiment_results_latest.csv
cp $LOG_FILE experiments_log_latest.txt

echo 'Latest results also copied to:'
echo '  - experiment_results_latest.csv'
echo '  - experiments_log_latest.txt'
"
