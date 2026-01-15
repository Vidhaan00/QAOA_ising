#PBS -N ising_x_n10_l100_block_25_s50
#PBS -lselect=1:ncpus=8:mem=40gb
#######walltime=10:00:00
#PBS -q workq
#PBS -m abe
#PBS -o output_x_n10_l100_block_25_s50.log
#PBS -e error.log

cd ${PBS_O_WORKDIR}

# Initialize Conda
source /apps/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate deeplearning

# Print time
date

# Run your Python script
python3 ising_x_block_wise.py

# Print time again
date

