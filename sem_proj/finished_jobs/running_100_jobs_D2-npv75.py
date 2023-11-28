import subprocess

for i in range(46, 101):
    print(i)
    subprocess.run(f"sbatch run_job_D2-npv75.sh {i}", shell=True)