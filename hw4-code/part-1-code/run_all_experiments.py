#!/usr/bin/env python3
"""
NLP Assignment 4 - Part 1: Complete Experiment Runner (Python version)
This script runs all required experiments for Part-1 and saves detailed logs
"""

import os
import sys
import subprocess
import re
from datetime import datetime
import shutil

def check_and_fix_dependencies():
    """Check and fix package dependencies before running experiments"""
    print("Checking dependencies...")

    needs_restart = False

    # Check pyarrow
    try:
        import pyarrow as pa
        if not hasattr(pa, 'PyExtensionType'):
            print("⚠️  PyArrow version is incompatible. Upgrading...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "pyarrow", "-y"],
                capture_output=True, text=True
            )
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pyarrow>=12.0.0"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✓ PyArrow upgraded - please run the script again")
                needs_restart = True
            else:
                print(f"✗ Failed to upgrade pyarrow: {result.stderr}")
                print("\nPlease manually run:")
                print("  pip uninstall pyarrow -y && pip install 'pyarrow>=12.0.0'")
                sys.exit(1)
    except ImportError:
        print("Installing pyarrow...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyarrow>=12.0.0"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✓ PyArrow installed - please run the script again")
            needs_restart = True
        else:
            print(f"✗ Failed to install pyarrow: {result.stderr}")
            sys.exit(1)

    # Check NLTK data
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print("✓ NLTK data downloaded")
    except ImportError:
        print("Installing nltk...")
        subprocess.run([sys.executable, "-m", "pip", "install", "nltk"],
                     check=False, capture_output=True)
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✓ NLTK installed")

    if needs_restart:
        print("\n" + "="*80)
        print("Dependencies have been updated. Please run the script again:")
        print("  python3 run_all_experiments.py")
        print("="*80)
        sys.exit(0)

    print("✓ All dependencies OK\n")

class ExperimentRunner:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs_{self.timestamp}"
        self.output_dir = f"outputs_{self.timestamp}"
        self.submission_dir = f"submission_package_{self.timestamp}"

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Main log file
        self.main_log = os.path.join(self.log_dir, "main_log.txt")

        # Results storage
        self.results = {}

    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.main_log, 'a') as f:
            f.write(log_msg + '\n')

    def run_command(self, name, command):
        """Run a command and capture output"""
        log_file = os.path.join(self.log_dir, f"{name}.log")

        self.log(f"Starting: {name}")
        self.log(f"Command: {command}")

        try:
            # Run command and capture output
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

                # Stream output to both console and file
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)

                process.wait()

                if process.returncode == 0:
                    f.write("\nSUCCESS\n")
                    self.log(f"Completed: {name} ✓")

                    # Extract accuracy if present
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
        """Copy output file and log it"""
        try:
            shutil.copy2(src, dst)
            self.log(f"Copied: {src} -> {dst}")
            return True
        except Exception as e:
            self.log(f"Error copying {src}: {str(e)}")
            return False

    def run_all_experiments(self):
        """Run all Part-1 experiments"""

        print("=" * 80)
        print("NLP HW4 Part-1 Experiment Runner")
        print(f"Started at: {datetime.now()}")
        print(f"Log directory: {self.log_dir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        print()

        # Q1: Train and evaluate on original data
        self.log("=== Q1: Training BERT on original data ===")
        if not self.run_command(
            "q1_train_eval_original",
            "python3 main.py --train --eval"
        ):
            self.log("Q1 failed, stopping...")
            return False

        self.copy_output(
            "out_original.txt",
            os.path.join(self.output_dir, "q1_out_original.txt")
        )

        # Q2: Evaluate on transformed data
        self.log("=== Q2: Evaluating on transformed data ===")
        if not self.run_command(
            "q2_eval_transformed",
            "python3 main.py --eval_transformed --model_dir ./out"
        ):
            self.log("Q2 failed, stopping...")
            return False

        self.copy_output(
            "out_transformed.txt",
            os.path.join(self.output_dir, "q2_out_transformed.txt")
        )

        # Q3: Train with augmented data
        self.log("=== Q3: Training with augmented data ===")
        if not self.run_command(
            "q3_train_augmented",
            "python3 main.py --train_augmented --eval_transformed"
        ):
            self.log("Q3 training failed, stopping...")
            return False

        self.copy_output(
            "out_augmented_transformed.txt",
            os.path.join(self.output_dir, "q3_out_augmented_transformed.txt")
        )

        # Q3: Evaluate augmented model on original data
        self.log("=== Q3: Evaluating augmented model on original data ===")
        if not self.run_command(
            "q3_eval_augmented_original",
            "python3 main.py --eval --model_dir out_augmented"
        ):
            self.log("Q3 evaluation failed, stopping...")
            return False

        self.copy_output(
            "out_augmented_original.txt",
            os.path.join(self.output_dir, "q3_out_augmented_original.txt")
        )

        return True

    def generate_summary(self):
        """Generate results summary"""
        self.log("=== Generating Results Summary ===")

        results_file = os.path.join(self.log_dir, "results_summary.txt")

        with open(results_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NLP HW4 Part-1 Results Summary\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

            # Display results
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

            # Calculate performance changes
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

        # Display summary
        with open(results_file, 'r') as f:
            print("\n" + f.read())

        self.log(f"Results summary saved to: {results_file}")
        return results_file

    def create_submission_package(self, results_file):
        """Create submission package"""
        self.log("=== Creating submission package ===")

        os.makedirs(self.submission_dir, exist_ok=True)

        # Copy required output files
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

        # Copy results summary
        shutil.copy2(results_file, self.submission_dir)

        # Create README
        readme_path = os.path.join(self.submission_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("NLP Assignment 4 - Part 1 Submission Package\n")
            f.write(f"Generated at: {datetime.now()}\n\n")
            f.write("=" * 80 + "\n\n")
            f.write("This package contains all required output files for Part-1:\n\n")
            f.write("Q1 Submission:\n")
            f.write("  - out_original.txt\n\n")
            f.write("Q2 Submission:\n")
            f.write("  - out_transformed.txt\n\n")
            f.write("Q3 Submission:\n")
            f.write("  - out_augmented_original.txt\n")
            f.write("  - out_augmented_transformed.txt\n\n")
            f.write("Results Summary:\n")
            f.write("  - results_summary.txt (contains all accuracy scores and analysis)\n\n")
            f.write("Full logs are available in the parent logs directory.\n\n")
            f.write("Model Checkpoints:\n")
            f.write("  - ./out/ (original model from Q1)\n")
            f.write("  - ./out_augmented/ (augmented model from Q3)\n")

        self.log(f"Submission package created: {self.submission_dir}/")

    def print_final_summary(self):
        """Print final summary"""
        print()
        print("=" * 80)
        print("All experiments completed successfully!")
        print("=" * 80)
        print()
        print("Directories created:")
        print(f"  - Logs: {self.log_dir}/")
        print(f"  - Outputs: {self.output_dir}/")
        print(f"  - Submission: {self.submission_dir}/")
        print()
        print("Model checkpoints:")
        print("  - ./out/ (Q1 original model)")
        print("  - ./out_augmented/ (Q3 augmented model)")
        print()
        print(f"Key files for submission (in {self.submission_dir}/):")
        print("  - out_original.txt")
        print("  - out_transformed.txt")
        print("  - out_augmented_original.txt")
        print("  - out_augmented_transformed.txt")
        print("  - results_summary.txt")
        print()
        print(f"Finished at: {datetime.now()}")
        print("=" * 80)

def main():
    # Fix dependencies first
    check_and_fix_dependencies()

    runner = ExperimentRunner()

    try:
        # Run all experiments
        if not runner.run_all_experiments():
            print("Experiments failed. Check logs for details.")
            sys.exit(1)

        # Generate summary
        results_file = runner.generate_summary()

        # Create submission package
        runner.create_submission_package(results_file)

        # Print final summary
        runner.print_final_summary()

        # Create completion marker
        with open("experiment_completed.txt", 'w') as f:
            f.write(f"Experiment completed at: {datetime.now()}\n")
            f.write(f"Results in: {runner.submission_dir}/\n")

        print("\nAll done! You can now log out and check back later.")
        print(f"Check {runner.submission_dir}/ for all submission files.")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
