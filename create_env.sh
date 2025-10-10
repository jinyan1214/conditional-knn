

# Create conda environment required for the conditional point selection
conda env create -f environment.yml -p ./.venv

# # activate conda environment
conda activate ./.venv

# # load cuda and install pytorch with cuda support
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e 
module load cudnn/8.9.7.29-12-gcc-11.3.1-v7mrdbz

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install additional python packages
pip3 install gpytorch
pip3 install pandas
pip3 install tqdm
pip3 install h5py

module load mpi/latest
python -m pip install mpi4py openmpi
pip3 install geopandas
pip3 install folium
pip3 install mapclassify
pip3 install contextily
pip3 install pygmm
