export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python train/train_classi_w_recon.py
