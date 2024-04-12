import subprocess


def run_script(script_path):
    try:
        # Execute the script
        completed_process = subprocess.run(
            ["python", script_path], check=True, text=True, capture_output=True
        )

        # Improved output formatting
        if completed_process.stdout:
            print("=== Process Output ===")
            print(completed_process.stdout.strip())
        if completed_process.stderr:
            print("=== Process Errors ===")
            print(completed_process.stderr.strip())

    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_path}: {e}\n{e.stderr}")


if __name__ == "__main__":
    script_path = "./src/evaluation/model_plots.py"
    run_script(script_path)
