#!/bin/bash

################################################################################
# NLP Assignment 4 - Part 1: Complete Experiment Runner
# This script runs all required experiments for Part-1 and saves logs
################################################################################

# Exit on error
set -e

# Create directories for logs and outputs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_${TIMESTAMP}"
OUTPUT_DIR="outputs_${TIMESTAMP}"

mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}

echo "========================================="
echo "NLP HW4 Part-1 Experiment Runner"
echo "Started at: $(date)"
echo "Log directory: ${LOG_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================="
echo ""

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${LOG_DIR}/main_log.txt
}

# Function to run experiment and capture output
run_experiment() {
    local exp_name=$1
    local command=$2
    local log_file="${LOG_DIR}/${exp_name}.log"

    log_message "Starting: ${exp_name}"
    echo "Command: ${command}" >> ${log_file}
    echo "========================================" >> ${log_file}

    # Run and capture both stdout and stderr
    if eval ${command} 2>&1 | tee -a ${log_file}; then
        log_message "Completed: ${exp_name} ✓"
        echo "SUCCESS" >> ${log_file}
    else
        log_message "Failed: ${exp_name} ✗"
        echo "FAILED" >> ${log_file}
        return 1
    fi
    echo ""
}

################################################################################
# Q1: Train and evaluate on original data
################################################################################
log_message "=== Q1: Training BERT on original data ==="
run_experiment \
    "q1_train_eval_original" \
    "python3 main.py --train --eval"

# Copy output file
cp out_original.txt ${OUTPUT_DIR}/q1_out_original.txt
log_message "Saved Q1 output to: ${OUTPUT_DIR}/q1_out_original.txt"

################################################################################
# Q2: Evaluate on transformed data
################################################################################
log_message "=== Q2: Evaluating on transformed data ==="
run_experiment \
    "q2_eval_transformed" \
    "python3 main.py --eval_transformed --model_dir ./out"

# Copy output file
cp out_transformed.txt ${OUTPUT_DIR}/q2_out_transformed.txt
log_message "Saved Q2 output to: ${OUTPUT_DIR}/q2_out_transformed.txt"

################################################################################
# Q3: Train with augmented data and evaluate on both datasets
################################################################################
log_message "=== Q3: Training with augmented data ==="
run_experiment \
    "q3_train_augmented" \
    "python3 main.py --train_augmented --eval_transformed"

# Copy augmented model's transformed evaluation output
cp out_augmented_transformed.txt ${OUTPUT_DIR}/q3_out_augmented_transformed.txt
log_message "Saved Q3 transformed output to: ${OUTPUT_DIR}/q3_out_augmented_transformed.txt"

log_message "=== Q3: Evaluating augmented model on original data ==="
run_experiment \
    "q3_eval_augmented_original" \
    "python3 main.py --eval --model_dir out_augmented"

# Copy augmented model's original evaluation output
cp out_augmented_original.txt ${OUTPUT_DIR}/q3_out_augmented_original.txt
log_message "Saved Q3 original output to: ${OUTPUT_DIR}/q3_out_augmented_original.txt"

################################################################################
# Extract and summarize results
################################################################################
log_message "=== Extracting Results Summary ==="

RESULTS_FILE="${LOG_DIR}/results_summary.txt"

echo "=========================================" > ${RESULTS_FILE}
echo "NLP HW4 Part-1 Results Summary" >> ${RESULTS_FILE}
echo "Generated at: $(date)" >> ${RESULTS_FILE}
echo "=========================================" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

# Extract accuracy from logs
echo "Q1: Original Model on Original Test Set" >> ${RESULTS_FILE}
grep -A 1 "Score:" ${LOG_DIR}/q1_train_eval_original.log | tail -1 >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

echo "Q2: Original Model on Transformed Test Set" >> ${RESULTS_FILE}
grep -A 1 "Score:" ${LOG_DIR}/q2_eval_transformed.log | tail -1 >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

echo "Q3: Augmented Model on Transformed Test Set" >> ${RESULTS_FILE}
grep -A 1 "Score:" ${LOG_DIR}/q3_train_augmented.log | tail -1 >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

echo "Q3: Augmented Model on Original Test Set" >> ${RESULTS_FILE}
grep -A 1 "Score:" ${LOG_DIR}/q3_eval_augmented_original.log | tail -1 >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

echo "=========================================" >> ${RESULTS_FILE}
echo "Accuracy Comparison:" >> ${RESULTS_FILE}
echo "=========================================" >> ${RESULTS_FILE}

# Display results summary
cat ${RESULTS_FILE}
log_message "Results summary saved to: ${RESULTS_FILE}"

################################################################################
# Create submission package
################################################################################
log_message "=== Creating submission package ==="

SUBMISSION_DIR="submission_package_${TIMESTAMP}"
mkdir -p ${SUBMISSION_DIR}

# Copy required files
cp ${OUTPUT_DIR}/q1_out_original.txt ${SUBMISSION_DIR}/out_original.txt
cp ${OUTPUT_DIR}/q2_out_transformed.txt ${SUBMISSION_DIR}/out_transformed.txt
cp ${OUTPUT_DIR}/q3_out_augmented_original.txt ${SUBMISSION_DIR}/out_augmented_original.txt
cp ${OUTPUT_DIR}/q3_out_augmented_transformed.txt ${SUBMISSION_DIR}/out_augmented_transformed.txt

# Copy results summary
cp ${RESULTS_FILE} ${SUBMISSION_DIR}/

# Create README
cat > ${SUBMISSION_DIR}/README.txt << EOF
NLP Assignment 4 - Part 1 Submission Package
Generated at: $(date)

This package contains all required output files for Part-1:

Q1 Submission:
  - out_original.txt

Q2 Submission:
  - out_transformed.txt

Q3 Submission:
  - out_augmented_original.txt
  - out_augmented_transformed.txt

Results Summary:
  - results_summary.txt (contains all accuracy scores)

Full logs are available in the parent logs directory.

Model Checkpoints:
  - ./out/ (original model from Q1)
  - ./out_augmented/ (augmented model from Q3)
EOF

log_message "Submission package created: ${SUBMISSION_DIR}/"

################################################################################
# Final Summary
################################################################################
echo ""
echo "========================================="
echo "All experiments completed successfully!"
echo "========================================="
echo ""
echo "Directories created:"
echo "  - Logs: ${LOG_DIR}/"
echo "  - Outputs: ${OUTPUT_DIR}/"
echo "  - Submission: ${SUBMISSION_DIR}/"
echo ""
echo "Model checkpoints:"
echo "  - ./out/ (Q1 original model)"
echo "  - ./out_augmented/ (Q3 augmented model)"
echo ""
echo "Key files for submission (in ${SUBMISSION_DIR}/):"
echo "  - out_original.txt"
echo "  - out_transformed.txt"
echo "  - out_augmented_original.txt"
echo "  - out_augmented_transformed.txt"
echo ""
echo "Results summary available in:"
echo "  - ${RESULTS_FILE}"
echo ""
echo "Finished at: $(date)"
echo "========================================="

# Create a timestamp file for easy reference
echo "Experiment completed at: $(date)" > experiment_completed.txt
echo "Results in: ${SUBMISSION_DIR}/" >> experiment_completed.txt
