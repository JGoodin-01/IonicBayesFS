# IonicBayesFS
 
## Usage:

For running this with and without CUDA, [Anaconda](https://www.anaconda.com/download) is required to some degree. For this purpose, Miniconda works fine.

### Base Usage:
This is to be ran through conda for compatibility with the packages.
Simple installation and run is as follows:
```
conda env create -f environment.yml
conda activate myenv
py ./main.py
```

### CUDA:
For usage with [RAPIDS](https://docs.rapids.ai/) library for GPU accelerated performance, an addition env file has been supplied to facilitate this.
This requires:
- NVIDIA GPU (Supportive of CUDAv12.0)
- For Windows, WSL2 running Ubuntu
  - [Installation of CUDA on a WSL-Ubuntu](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) 

Bash commands for usage are:
```bash
conda env create -f cuda_environment.yml
conda activate cuda_env
py ./main.py
```