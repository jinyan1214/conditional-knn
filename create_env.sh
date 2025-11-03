#!/bin/bash

HPC=caltech # caltech or stampede3
# # load cuda and install pytorch with cuda support
if [ "$HPC" = "caltech" ]; then
    module load cuda/12.2.1-gcc-11.3.1-sdqrj2e
    module load cudnn/8.9.7.29-12-gcc-11.3.1-v7mrdbz
    conda_path=/home/jyzhao/miniconda3/bin/conda
    # This is openmpi 
    module load mpi/latest
elif [ "$HPC" = "stampede3" ]; then
    # See module spider cuda/12.8
    # module load python/3.12.11
    module load nvidia/25.3
    module load cuda/12.8
    module load openmpi/5.0.7
    conda_path=/home1/07059/jyzhao/miniconda3/bin/conda
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Create conda environment required for the conditional point selection
$conda_path env create -f environment.yml -p ./.venv

# # activate conda environment
$conda_path activate ./.venv

python setup.py build_ext --inplace


if [ "$HPC" = "caltech" ]; then
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    conda install -c conda-forge openmpi mpi4py
elif [ "$HPC" = "stampede3" ]; then
    # See module spider cuda/12.8
    pip3 install torch torchvision
    # For stampede3, you may need to install mpi4py with impi-rt
    python -m pip install mpi4py impi-rt
fi

# Install additional python packages
pip3 install gpytorch
pip3 install pandas
pip3 install tqdm
pip3 install h5py

pip3 install geopandas
pip3 install folium
pip3 install mapclassify
pip3 install contextily
pip3 install pygmm
