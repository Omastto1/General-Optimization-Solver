#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
# SBATCH --job-name=jobshop

# Define, how many nodes you need. Here, we ask for 1 node.
# 
# See 'System configuration' part of this manual for information about
# available cores.
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=14
# You can further define the number of tasks with --ntasks-per-*
# See "man sbatch" for details. e.g. --ntasks=4 will ask for 4 cpus.

# Define, how long the job will run in real time. This is a hard cap meaning
# that if the job runs longer than what is written here, it will be
# force-stopped by the server. If you make the expected time too long, it will
# take longer for the job to start. Here, we say the job will take 5 minutes.
#              d-hh:mm:ss
# SBATCH --time=0-03:00:10

# Define the partition on which the job shall run. May be omitted.
# See 'System configuration' for info about available partitionss

# How much memory you need.
# --mem will define memory per node and
# --mem-per-cpu will define memory per CPU/core. Choose one of those.
## SBATCH --mem-per-cpu=1500MB
#SBATCH --mem=4GB    # this one is not in effect, due to the double hash

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
# SBATCH --mail-type=BEGIN,END,FAIL

# You may not place any commands before the last SBATCH directive

# Define and create a unique scratch directory for this job
# /lscratch is local ssd disk on particular node which is faster
# than your network home dir
# SCRATCH_DIRECTORY=/lscratch/${USER}/${SLURM_JOBID}.stallo-adm.uit.no
# mkdir -p ${SCRATCH_DIRECTORY}
# cd ${SCRATCH_DIRECTORY}

# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was
# submitted from (usually in your network home dir)
# cp ${SLURM_SUBMIT_DIR}/myfiles*.txt ${SCRATCH_DIRECTORY}

# This is where the actual work is done. In this case, the script only waits.
# The time command is optional, but it may give you a hint on how long the
# command worked
# time sleep 10
#sleep 10
python run_jobshop_all_instances.py

# After the job is done we copy our output back to $SLURM_SUBMIT_DIR
# cp ${SCRATCH_DIRECTORY}/my_output ${SLURM_SUBMIT_DIR}

# In addition to the copied files, you will also find a file called
# slurm-1234.out in the submit directory. This file will contain all output that
# was produced during runtime, i.e. stdout and stderr.

# After everything is saved to the home directory, delete the work directory to
# save space on /lscratch
# old files in /lscratch will be deleted automatically after some time
# cd ${SLURM_SUBMIT_DIR}
# rm -rf ${SCRATCH_DIRECTORY}

# Finish the script
exit 0