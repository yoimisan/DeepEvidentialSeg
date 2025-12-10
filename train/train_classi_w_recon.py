"""
Docstring for src.train.train_classi_w_recon

Training classification only, with reconstruction loss.
"""
from matplotlib.pylab import f
import torch
from torch.utils.data import DataLoader
from src import RUGDH5Dataset
from src import DeepEvidentialSegModelConfig, FPNFeatureExtractorConfig, PatchDecoderConfig
import wandb
from rich import print as rprint
from tqdm import tqdm

# 设置设备和混合精度训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler(device=device)

# 设置超参数
epochs = 120 # 过拟合看看最好能到什么程度（实验之后大概 60 epoch 就差不多收敛了）
lr = 1e-5
batch_size = 16

# 加载数据集
dataset = RUGDH5Dataset('./data/rugd_train.h5')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 模型设置部分
## 初始化模型
feature_extractor_config = FPNFeatureExtractorConfig()
patch_decoder_config = PatchDecoderConfig(
    num_samples=1024,
)
model_config = DeepEvidentialSegModelConfig(
    feature_extractor_config=feature_extractor_config,
    patch_decoder_config=patch_decoder_config,
    num_classes=20, # RUGD 人为将后 5 类别放到了 validate 里面，所以训练集只有 20 类
)
rprint('[bold green]模型配置：')
rprint(model_config)
model = model_config.make_model(device=device)

## 设置优化器
optim_grouped_parameters = [
    {
        'params': model.feature_extractor.encoder.parameters(),
        'lr': lr
    },
    {
        'params': model.feature_extractor.decoder.parameters(), 
        'lr': lr * 10 # 因为没有加载预训练权重，需要用更大学习率
    },
    {
        'params': model.classification_head.parameters(),
        'lr': lr * 10 # 同理
    },
    {
        'params': model.path_decoder.parameters(),
        'lr': lr * 10 # 同理
    }
]
optim = torch.optim.AdamW(optim_grouped_parameters, weight_decay=1e-2)

# 学习率调度器（这里不用了，用了没用）
scheduler = None
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim,
    T_max=epochs,
    eta_min=1e-7, # 最小学习率
)

# 初始化 WandB
wandb.init(
    project='DeepEvidentialSeg',
    name='train_classi_w_recon',
    group='classification with reconstruction',
    notes='Training classification only, with reconstruction loss. CosineAnnealingLR scheduler.',
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
            loss, logits, info = model.classify(images.to(device), labels.to(device))

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if wandb.run is not None:
            wandb.log({
                'loss': info['total_loss'],
                'classification_loss': info['class_loss'],
                'reconstruction_loss': info['recon_loss'],
                'epoch': epoch,
                'lr': optim.param_groups[0]['lr'],
            })
        tbar.set_postfix(
            **{k: f"{v:.2f}" for k, v in info.items()}
        )
    if scheduler is not None:
        scheduler.step()
    tbar.close()
    
# 保存模型
torch.save(model.state_dict(), './checkpoints/classi_w_recon/deep_evidential_seg_model_ver3.pth')
