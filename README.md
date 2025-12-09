# Activate Conda environment for python 3.8

```
conda create -n food-recog-venv python=3.8 -y
conda activate food-recog-venv
```

# Install torch cpu version

```
# Cài PyTorch bản mới nhất (Hỗ trợ tốt Python 3.12)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài timm và các thư viện hỗ trợ
pip install timm scipy
```

# Install other dependencies

move to lsnet folder

```
pip install -r requirements.txt
```"# Intro2ML-Lab03-FoodRecLSNet" 
