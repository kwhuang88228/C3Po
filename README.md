
## Get Started

### Installation

1. Clone C3Po.
```bash
git clone --recursive git@github.com:kwhuang88228/C3Po.git
cd C3Po
# if you have already cloned dust3r:
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n c3po python=3.11 cmake=3.14.0
conda activate c3po 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121  # use the correct version for you
pip install -r requirements_optional.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Checkpoints

Pre-trained mode weight:
[`ckpt.pth`](https://1drv.ms/f/s!AgWLhKhRf9v1jM0-2B3MmvMfxHC2fA?e=zrXGkr)

### Demo

Run demo.ipynb
