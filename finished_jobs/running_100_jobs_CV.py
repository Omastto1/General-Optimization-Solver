import subprocess

for i in range(1, 101):
    print(i)
    subprocess.run(f"sbatch run_job_CV.sh {i}", shell=True)