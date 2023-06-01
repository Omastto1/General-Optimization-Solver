import subprocess

for i in range(1, 101):
    print(i)
    subprocess.run(f"sbatch run_job_c15.sh {i}", shell=True)