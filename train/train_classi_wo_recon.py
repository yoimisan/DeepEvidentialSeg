"""
Docstring for src.train.train_classi_wo_recon

Training classification only, without reconstruction loss.
"""
import sys
import os

# 获取当前脚本的绝对路径 (DEEPEVIDENTIALSEG/train/train_classi_w_recon.py)
current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在目录 (DEEPEVIDENTIALSEG/train)
current_dir = os.path.dirname(current_file_path)
# 获取项目根目录 (DEEPEVIDENTIALSEG)
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到系统路径中
sys.path.append(project_root)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src import RUGDH5Dataset, DeepEvidentialSegModelConfig
import wandb
from tqdm import tqdm

# 设置设备和混合精度训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler(device=device)

# 设置超参数
epochs = 50 # 别太多，太多会过拟合
lr = 1e-5
batch_size = 16

# 加载数据集
dataset = RUGDH5Dataset('./data/rugd_train.h5')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 模型设置部分
## 初始化模型
model_config = DeepEvidentialSegModelConfig()
model = model_config.make_model(device=device)

## 设置优化器
optim_grouped_parameters = [
    {
        'params': model.feature_extractor.encoder.parameters(),
        'lr': lr
    },
    {
        'params': model.feature_extractor.decoder.parameters(), 
        'lr': lr * 10
    },
    {
        'params': model.classification_head.parameters(),
        'lr': lr * 10
    },
]
optim = torch.optim.AdamW(optim_grouped_parameters, weight_decay=1e-2)

## 设置学习率调度器
# warmup_epochs = 3 # 预热3个epoch，可以不要的（这个操作纯纯负优化）
# warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#     optim,
#     start_factor=0.01,
#     total_iters=warmup_epochs, # 预热 $warmup_epochs 个epoch
# )
# main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optim,
#     T_0=10,
#     T_mult=2, # 每次重启后周期倍增, 即10,20,40,...
#     eta_min=1e-6, # 最小学习率
# )
# scheduler = torch.optim.lr_scheduler.SequentialLR(
#     optim,
#     schedulers=[warmup_scheduler, main_scheduler],
#     milestones=[warmup_epochs],
# )

# 初始化WandB
wandb.init(
    project='DeepEvidentialSeg',
    name='classification_only_train',
    group='classification_only',
    notes='Training Deep Evidential Segmentation Model - Classification Only',
    tags=['different_lr'],
    config={
        'epochs': epochs,
        'lr': lr,
    }
)

# 训练循环
for epoch in range(epochs):
    tbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', dynamic_ncols=True, leave=False)
    for images, labels in tbar:
        with torch.amp.autocast(device_type=device.type):
            loss, logits = model.classify(images.to(device), labels.to(device))

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if wandb.run is not None:
            wandb.log({
                'loss': loss.item(),
                'epoch': epoch,
                'lr': optim.param_groups[0]['lr'],
            })
        tbar.set_postfix({'loss': loss.item()})
    # scheduler.step()
    tbar.close()
    
# 保存模型
torch.save(model.state_dict(), './checkpoints/deep_evidential_seg_model_different_lr.pth')
