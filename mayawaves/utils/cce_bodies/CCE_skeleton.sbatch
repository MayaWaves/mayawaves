#!/bin/bash

#SBATCH -A {allocation}
#SBATCH -J CCE                               # Job name
#SBATCH -o {out}                           # Name of stdout output file (%j expands to jobId)
#SBATCH -e {err}                           # Name of stderr err file (%j expands to jobId)
#SBATCH -p {queue}                       # Queue name
#SBATCH -N {nodes}                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n {cores}                                # Total number of mpi tasks requested
#SBATCH -t {walltime}                          # Run time (hh:mm:ss)
{email_slurm}
CCE_DIR={spectre_dir}/CceExecutables
PREPROCESS_DIR=$CCE_DIR/PreprocessCceWorldtube
INPUT_DIR={target_directory}

echo 'Starting CCE sbatch script:'
date

module load tacc-apptainer
cd $SCRATCH
apptainer pull docker://sxscollaboration/spectre:dev

echo "Converting worldtube H5 file into a corresponding Bondi-Sachs worldtube H5 file that can be read in by CCE . . ."
echo
cd $PREPROCESS_DIR
apptainer exec $SCRATCH/spectre_dev.sif ./PreprocessCceWorldtube --input-file $INPUT_DIR/PreprocessCceWorldtube.yaml

echo "Starting CCE executable . . ."
echo
cd $CCE_DIR
apptainer exec $SCRATCH/spectre_dev.sif ./CharacteristicExtract --input-file $INPUT_DIR/CharacteristicExtract.yaml
