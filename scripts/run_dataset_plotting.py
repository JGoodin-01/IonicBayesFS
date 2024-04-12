import subprocess


def run_script(script_path):
    try:
        with subprocess.Popen(
            ["python", script_path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as proc:
            # Read line by line as it's being output
            while True:
                output = proc.stdout.readline()
                if output == "" and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())
            rc = proc.poll()
            return rc

    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_path}: {e}")


if __name__ == "__main__":
    script_path = "./src/evaluation/dataset_plots.py"
    run_script(script_path)
