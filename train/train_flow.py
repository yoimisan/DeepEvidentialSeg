import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb # 导入 wandb
from tqdm import tqdm

# ==========================================
# 1. 修复路径与导入
# ==========================================
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上跳一级到项目根目录，以便找到 src
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# 按照文件夹结构正确导入
from src.dataset import RUGDH5Dataset # 数据集在 dataset.py
from src.model import (
    DeepEvidentialSegModelConfig, 
    TRAIN_STAGE
)

# ==========================================
# 2. 配置与超参数
# ==========================================
config = {
    "lr": 1e-4,
    "batch_size": 16,
    "epochs": 50,
    "gmm_components": 20,
    "num_classes": 20,
    "checkpoint_s1": "../checkpoints/classi_w_recon/deep_evidential_seg_model_final.pth",
    "save_dir": "../checkpoints/stage2_evidential"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(config["save_dir"], exist_ok=True)

# ==========================================
# 3. 初始化 WandB
# ==========================================
wandb.init(
    project="DeepEvidentialSeg",
    name="Stage2_NormalizingFlow",
    config=config
)

# ==========================================
# 4. 加载数据与模型
# ==========================================
dataset = RUGDH5Dataset('../data/rugd_train.h5')
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

model_config = DeepEvidentialSegModelConfig(
    train_stage=TRAIN_STAGE.NORMALIZING_FLOW,
    num_classes=config["num_classes"],
    gmm_components=config["gmm_components"]
)
model = model_config.make_model(device=device)

# 加载 Stage 1 权重
if os.path.exists(config["checkpoint_s1"]):
    print(f"Loading Stage 1 weights from {config['checkpoint_s1']}")
    ckpt = torch.load(config["checkpoint_s1"], map_location=device)
    # 使用 strict=False 因为 Stage 2 有新的 NF 参数
    model.load_state_dict(ckpt, strict=False)
else:
    print("Warning: Stage 1 checkpoint not found. Starting from scratch.")

# ==========================================
# 5. 【核心】拟合 GMM
# ==========================================
print("Step 1: Fitting GMM base distribution...")
model.fit_gmm(dataloader)

# ==========================================
# 6. 设置优化器
# ==========================================
# 在 Stage 2，我们冻结 Encoder，只训练 NF 和分类头
for param in model.feature_extractor.parameters():
    param.requires_grad = False

# 收集需要训练的参数
params_to_train = (
    list(model.density_estimator.parameters()) + 
    list(model.classification_head.parameters())
)
optimizer = optim.Adam(params_to_train, lr=config["lr"])
scaler = torch.amp.GradScaler(device=device)

# ==========================================
# 7. 训练循环
# ==========================================
print("Step 2: Training Normalizing Flow...")
model.train()

for epoch in range(config["epochs"]):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    epoch_metrics = {"loss": 0, "likelihood": 0, "reg": 0, "log_density": 0}
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type):
            # 内部会自动进行像素采样以防 OOM
            loss, logits, info = model.classify(images, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        metrics = {
            "batch_loss": loss.item(),
            "loss_likelihood": info['loss_likelihood'],
            "loss_reg": info['loss_reg'],
            "mean_log_density": info['mean_log_density']
        }
        # 记录到 WandB
        wandb.log(metrics)
        
        pbar.set_postfix({
            'L': f"{metrics['batch_loss']:.3f}",
            'Dens': f"{metrics['mean_log_density']:.1f}"
        })

    # 保存模型
    save_path = f"{config['save_dir']}/stage2_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), save_path)
    # 也可以把模型上传到 wandb 存档
    # wandb.save(save_path)

wandb.finish()
print("Training Finished!")