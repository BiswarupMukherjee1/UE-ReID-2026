# Environment Setup

## Step 1: Open terminal
code
source /media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main/setup_ue_reid.sh

## Step 2: Create conda environment (do this fresh each time if needed)
conda create -n pat python=3.10 -y
conda activate pat
cd /media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4
pip install einops timm scikit-image opencv-python tensorboard yacs

## Step 3: Verify CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

## Monitoring Training
# Check process is alive
ps aux | grep train.py
# Watch log live
tail -f /media/DiscoLocal/IPCV/UE-ReID/experiments/run_80ep.log

## Notes
- numpy must be 1.26.4 (not higher)
- If CUDA fails without full install, try activating pat env then running directly
