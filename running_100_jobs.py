import subprocess

for i in range(1,101):
    subprocess.run(f"python run_job_j30.py {i}", shell=True)