* Create an environment and install dependencies
```sh
conda create --prefix /work/piyush/conda_envs/timechat python=3.9
conda activate /work/piyush/conda_envs/timechat
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements_specific.txt
pip install --no-deps peft==0.5.0
pip install --no-deps accelerate==0.23.0
pip install jupyterlab ipdb
```

Download checkpoints
```sh
# See README
```
