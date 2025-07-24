import subprocess
from pathlib import Path


def run_test(log_path, out_file):
    """"""
    cmd = [
        "python", "analyze_logs.py",
        log_path,
        "--out", out_file,
    ]
    # print(cmd)
    subprocess.run(cmd, check=True)


def main():
    log_files = [Path("")] # List of .json log files to analyze

    for log_file in log_files:
        out_file = log_file.parent / Path("loss_plot.png")
        run_test(str(log_file), str(out_file))

if __name__ == "__main__":
    main()

