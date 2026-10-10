import subprocess
import sys
import os
import argparse

def run_script(script_name, args=None):
    """Run a Python script and stream its output."""
    if args is None:
        args = []
        
    print(f"==================================================")
    print(f"Running: {script_name} {' '.join(args)}")
    print(f"==================================================")
    
    cmd = [sys.executable, script_name] + args
    
    try:
        # Run process and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] {script_name} failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
        print(f"\n[SUCCESS] {script_name} completed successfully.\n")
        
    except KeyboardInterrupt:
        print(f"\n[WARNING] Interrupted by user while running {script_name}")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to run {script_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the complete sensitivity analysis pipeline.")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs for run_sweep.py (default: 4)")
    args = parser.parse_args()

    # Change to the script's directory to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Starting sensitivity analysis pipeline...\n")
    
    # 1. Run the parameter sweep
    run_script("run_sweep.py", ["--jobs", str(args.jobs)])
    
    # 2. Make figures
    run_script("make_figures.py")
    
    # 3. Run verifications
    run_script("verify.py")
    
    print("==================================================")
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
