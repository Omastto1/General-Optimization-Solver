#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=j30_ga_test_${SLURM_ARRAY_JOB_ID}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=0-00:05:00
#SBATCH --partition gpu
#SBATCH --mem=16G
##SBATCH --mail-user=tomom@email.cz
#SBATCH --mail-type=END,FAIL

# Define and create a unique scratch directory for this job
# /lscratch is local ssd disk on particular node which is faster
# than your network home dir
SCRATCH_DIRECTORY=/lscratch/${USER}/${SLURM_JOBID}.stallo-adm.uit.no
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was
# submitted from (usually in your network home dir)
cp ${SLURM_SUBMIT_DIR}/myfiles*.txt ${SCRATCH_DIRECTORY}

python runner_RCPSP_j30_ga_comparison.py

# After the job is done we copy our output back to $SLURM_SUBMIT_DIR
cp ${SCRATCH_DIRECTORY}/my_output ${SLURM_SUBMIT_DIR}

# In addition to the copied files, you will also find a file called
# slurm-1234.out in the submit directory. This file will contain all output that
# was produced during runtime, i.e. stdout and stderr.

# After everything is saved to the home directory, delete the work directory to
# save space on /lscratch
# old files in /lscratch will be deleted automatically after some time
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}

# Finish the script
exit 0