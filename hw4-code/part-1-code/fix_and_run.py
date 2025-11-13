#!/usr/bin/env python3
"""
Fix dependencies and run experiments
"""
import subprocess
import sys

print("Fixing pyarrow compatibility issue...")
print("This may take a minute...\n")

# Uninstall old pyarrow
print("Step 1: Removing old pyarrow...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "pyarrow", "-y"],
               capture_output=True)

# Install compatible version
print("Step 2: Installing compatible pyarrow...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow==14.0.1"],
                       capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error installing pyarrow: {result.stderr}")
    sys.exit(1)

print("Step 3: Installing NLTK...")
subprocess.run([sys.executable, "-m", "pip", "install", "nltk"],
               capture_output=True)

print("Step 4: Downloading NLTK data...")
subprocess.run([sys.executable, "-c",
                "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"],
               capture_output=True)

print("\n✓ Dependencies fixed!\n")
print("="*80)
print("Now running experiments...")
print("="*80)
print()

# Now run the actual experiments by importing
sys.path.insert(0, '.')

# Import fresh without the dependency check
import os
import re
from datetime import datetime
import shutil

class ExperimentRunner:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs_{self.timestamp}"
        self.output_dir = f"outputs_{self.timestamp}"
        self.submission_dir = f"submission_package_{self.timestamp}"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.main_log = os.path.join(self.log_dir, "main_log.txt")
        self.results = {}

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.main_log, 'a') as f:
            f.write(log_msg + '\n')

    def run_command(self, name, command):
        log_file = os.path.join(self.log_dir, f"{name}.log")
        self.log(f"Starting: {name}")
        self.log(f"Command: {command}")

        try:
            with open(log_file, 'w') as f:
                f.write(f"Command: {command}\n")
                f.write("=" * 80 + "\n\n")

                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                for line in process.stdout:
                    print(line, end='')
                    f.write(line)

                process.wait()

                if process.returncode == 0:
                    f.write("\nSUCCESS\n")
                    self.log(f"Completed: {name} ✓")

                    with open(log_file, 'r') as rf:
                        content = rf.read()
                        match = re.search(r"Score:\s*\{'accuracy':\s*([\d.]+)\}", content)
                        if match:
                            accuracy = float(match.group(1))
                            self.results[name] = accuracy
                            self.log(f"Accuracy for {name}: {accuracy:.4f}")
                    return True
                else:
                    f.write("\nFAILED\n")
                    self.log(f"Failed: {name} ✗")
                    return False

        except Exception as e:
            self.log(f"Error running {name}: {str(e)}")
            return False

    def copy_output(self, src, dst):
        try:
            shutil.copy2(src, dst)
            self.log(f"Copied: {src} -> {dst}")
            return True
        except Exception as e:
            self.log(f"Error copying {src}: {str(e)}")
            return False

    def run_all_experiments(self):
        print("=" * 80)
        print("NLP HW4 Part-1 Experiment Runner")
        print(f"Started at: {datetime.now()}")
        print(f"Log directory: {self.log_dir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        print()

        # Q1
        self.log("=== Q1: Training BERT on original data ===")
        if not self.run_command("q1_train_eval_original", "python3 main.py --train --eval"):
            self.log("Q1 failed, stopping...")
            return False
        self.copy_output("out_original.txt", os.path.join(self.output_dir, "q1_out_original.txt"))

        # Q2
        self.log("=== Q2: Evaluating on transformed data ===")
        if not self.run_command("q2_eval_transformed", "python3 main.py --eval_transformed --model_dir ./out"):
            self.log("Q2 failed, stopping...")
            return False
        self.copy_output("out_transformed.txt", os.path.join(self.output_dir, "q2_out_transformed.txt"))

        # Q3
        self.log("=== Q3: Training with augmented data ===")
        if not self.run_command("q3_train_augmented", "python3 main.py --train_augmented --eval_transformed"):
            self.log("Q3 training failed, stopping...")
            return False
        self.copy_output("out_augmented_transformed.txt", os.path.join(self.output_dir, "q3_out_augmented_transformed.txt"))

        self.log("=== Q3: Evaluating augmented model on original data ===")
        if not self.run_command("q3_eval_augmented_original", "python3 main.py --eval --model_dir out_augmented"):
            self.log("Q3 evaluation failed, stopping...")
            return False
        self.copy_output("out_augmented_original.txt", os.path.join(self.output_dir, "q3_out_augmented_original.txt"))

        return True

    def generate_summary(self):
        self.log("=== Generating Results Summary ===")
        results_file = os.path.join(self.log_dir, "results_summary.txt")

        with open(results_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NLP HW4 Part-1 Results Summary\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

            experiments = [
                ("Q1: Original Model on Original Test Set", "q1_train_eval_original"),
                ("Q2: Original Model on Transformed Test Set", "q2_eval_transformed"),
                ("Q3: Augmented Model on Transformed Test Set", "q3_train_augmented"),
                ("Q3: Augmented Model on Original Test Set", "q3_eval_augmented_original"),
            ]

            for title, key in experiments:
                f.write(f"{title}\n")
                if key in self.results:
                    f.write(f"  Accuracy: {self.results[key]:.4f} ({self.results[key]*100:.2f}%)\n")
                else:
                    f.write("  Accuracy: Not available\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("Performance Analysis:\n")
            f.write("=" * 80 + "\n\n")

            if "q1_train_eval_original" in self.results and "q2_eval_transformed" in self.results:
                orig_acc = self.results["q1_train_eval_original"]
                trans_acc = self.results["q2_eval_transformed"]
                drop = (orig_acc - trans_acc) * 100
                f.write(f"Accuracy drop from transformation: {drop:.2f} percentage points\n")
                f.write(f"  Original: {orig_acc*100:.2f}%\n")
                f.write(f"  Transformed: {trans_acc*100:.2f}%\n\n")

            if "q3_train_augmented" in self.results and "q2_eval_transformed" in self.results:
                before_aug = self.results["q2_eval_transformed"]
                after_aug = self.results["q3_train_augmented"]
                improvement = (after_aug - before_aug) * 100
                f.write(f"Improvement from augmentation on transformed data: {improvement:.2f} percentage points\n")
                f.write(f"  Before augmentation: {before_aug*100:.2f}%\n")
                f.write(f"  After augmentation: {after_aug*100:.2f}%\n\n")

            if "q1_train_eval_original" in self.results and "q3_eval_augmented_original" in self.results:
                orig_model = self.results["q1_train_eval_original"]
                aug_model = self.results["q3_eval_augmented_original"]
                diff = (aug_model - orig_model) * 100
                f.write(f"Effect of augmentation on original data: {diff:+.2f} percentage points\n")
                f.write(f"  Original model: {orig_model*100:.2f}%\n")
                f.write(f"  Augmented model: {aug_model*100:.2f}%\n\n")

        with open(results_file, 'r') as f:
            print("\n" + f.read())

        self.log(f"Results summary saved to: {results_file}")
        return results_file

    def create_submission_package(self, results_file):
        self.log("=== Creating submission package ===")
        os.makedirs(self.submission_dir, exist_ok=True)

        files_to_copy = [
            ("q1_out_original.txt", "out_original.txt"),
            ("q2_out_transformed.txt", "out_transformed.txt"),
            ("q3_out_augmented_original.txt", "out_augmented_original.txt"),
            ("q3_out_augmented_transformed.txt", "out_augmented_transformed.txt"),
        ]

        for src_name, dst_name in files_to_copy:
            src = os.path.join(self.output_dir, src_name)
            dst = os.path.join(self.submission_dir, dst_name)
            self.copy_output(src, dst)

        shutil.copy2(results_file, self.submission_dir)

        readme_path = os.path.join(self.submission_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("NLP Assignment 4 - Part 1 Submission Package\n")
            f.write(f"Generated at: {datetime.now()}\n\n")
            f.write("Files for Gradescope:\n")
            f.write("  - out_original.txt (Q1)\n")
            f.write("  - out_transformed.txt (Q2)\n")
            f.write("  - out_augmented_original.txt (Q3)\n")
            f.write("  - out_augmented_transformed.txt (Q3)\n\n")
            f.write("Results summary: results_summary.txt\n")

        self.log(f"Submission package created: {self.submission_dir}/")

    def print_final_summary(self):
        print()
        print("=" * 80)
        print("All experiments completed successfully!")
        print("=" * 80)
        print(f"Results in: {self.submission_dir}/")
        print(f"Finished at: {datetime.now()}")
        print("=" * 80)

# Run experiments
runner = ExperimentRunner()

try:
    if not runner.run_all_experiments():
        print("Experiments failed. Check logs for details.")
        sys.exit(1)

    results_file = runner.generate_summary()
    runner.create_submission_package(results_file)
    runner.print_final_summary()

    with open("experiment_completed.txt", 'w') as f:
        f.write(f"Experiment completed at: {datetime.now()}\n")
        f.write(f"Results in: {runner.submission_dir}/\n")

except KeyboardInterrupt:
    print("\n\nExperiment interrupted by user.")
    sys.exit(1)
except Exception as e:
    print(f"\n\nError: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
