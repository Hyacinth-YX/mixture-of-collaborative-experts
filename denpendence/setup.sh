conda create -n moce python=3.8
conda activate moce
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-2.0.0%2Bcu118.html
pip install -r requirements.txt