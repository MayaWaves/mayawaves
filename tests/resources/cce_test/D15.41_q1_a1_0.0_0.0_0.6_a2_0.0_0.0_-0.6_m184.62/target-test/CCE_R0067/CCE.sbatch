#!/bin/bash

#SBATCH -A PHY00000
#SBATCH -J CCE                               # Job name
#SBATCH -o /path/to/target/dir/on/cluster/CCE_R0067/CCE_R67.00.out                           # Name of stdout output file (%j expands to jobId)
#SBATCH -e /path/to/target/dir/on/cluster/CCE_R0067/CCE_R67.00.err                           # Name of stderr err file (%j expands to jobId)
#SBATCH -p development                       # Queue name
#SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n 56                                # Total number of mpi tasks requested
#SBATCH -t 02:00:00                          # Run time (hh:mm:ss)
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=all    # Send email at begin and end of job

CCE_DIR=/path/to/spectre_cce/CceExecutables/CceExecutables
PREPROCESS_DIR=$CCE_DIR/PreprocessCceWorldtube
INPUT_DIR=/path/to/target/dir/on/cluster/CCE_R0067

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
