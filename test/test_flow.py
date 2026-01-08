import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os
import argparse

# ==========================================
# 1. 路径设置 (适配你的目录结构)
# ==========================================
# 获取当前脚本所在目录 (DEEPEVIDENTIALSEG/test)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (DEEPEVIDENTIALSEG)
project_root = os.path.dirname(current_dir)

# 将 src 添加到系统路径，以便导入模块
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.model import DeepEvidentialSegModelConfig, FPNFeatureExtractorConfig, PatchDecoderConfig, DeepEvidentialSegModel, TRAIN_STAGE
from src.dataset import RUGDH5Dataset 
from src.utils import image_reverse_transforms, label2name, visualize_with_legend

def parse_args():
    parser = argparse.ArgumentParser(description="Test Normalizing Flow and Uncertainty Estimation")
    
    # 默认路径设置，根据你的截图调整
    parser.add_argument('--train_data', type=str, 
                        default=os.path.join(project_root, 'data', 'rugd_train.h5'),
                        help='Path to training data for GMM fitting')
    parser.add_argument('--test_data', type=str, 
                        default=os.path.join(project_root, 'data', 'rugd_test.h5'),
                        help='Path to test data for inference')
    parser.add_argument('--checkpoint', type=str, 
                        # 假设权重在 checkponts 文件夹下
                        default=os.path.join(project_root, 'checkpoints', 'classi_w_recon', 'deep_evidential_seg_model_ver3.pth'),
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--gmm_components', type=int, default=20, help='Number of GMM components')
    return parser.parse_args()

def compute_uncertainty(model, image_tensor):
    """
    执行前向推理并计算 Aleatoric 和 Epistemic 不确定性
    """
    model.eval()
    with torch.no_grad():
        # 1. 特征提取
        # 注意：这里我们手动调用 extractor，方便 reshape
        latent = model.feature_extractor(image_tensor)
        features = model.feature_extractor.upsample(latent, size=image_tensor.shape[-2:]) # [1, C, H, W]
        
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # 2. 分类 Logits
        logits_flat = model.classification_head(features_flat)
        logits = logits_flat.view(B, H, W, -1).permute(0, 3, 1, 2) # [1, NumClass, H, W]
        
        # 3. 密度估计 (Epistemic Source)
        # 确保 density_estimator 已经初始化
        if model.density_estimator is None:
            raise RuntimeError("Density Estimator is None. Did you run fit_gmm?")
            
        log_density_flat = model.density_estimator(features_flat) # [N]
        log_density = log_density_flat.view(H, W)
        
        # --- 计算指标 ---
        
        # A. 预测结果
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
        
        # B. Aleatoric Uncertainty (Entropy)
        # H = - sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        aleatoric = entropy.squeeze(0).cpu().numpy()
        
        # C. Epistemic Uncertainty (-LogDensity)
        epistemic = -log_density.cpu().numpy()
        
        return prediction, aleatoric, epistemic

def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Project Root: {project_root}")

    # ==========================================
    # 2. 模型初始化 (Flow Mode)
    # ==========================================
    feature_config = FPNFeatureExtractorConfig(
        encoder_name="resnet50", 
        decoder_segmentation_channels=128
    )
    # 保持 Config 结构完整
    patch_config = PatchDecoderConfig(num_samples=0) 

    model_config = DeepEvidentialSegModelConfig(
        feature_extractor_config=feature_config,
        patch_decoder_config=patch_config,
        num_classes=24, # 请根据 utils.py 确认你的类别数
        train_stage=TRAIN_STAGE.NORMALIZING_FLOW, # <--- 关键：开启 Flow 模式
        gmm_components=args.gmm_components
    )

    model = model_config.make_model(device=device)

    # ==========================================
    # 3. 加载 Stage 1 权重
    # ==========================================
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        
        # strict=False 允许忽略 Flow 和 GMM 的缺失权重
        msg = model.load_state_dict(state_dict, strict=False)
        print("Weights loaded.")
        print(f"  Missing keys (Expected): {len(msg.missing_keys)} (Density Estimator parameters)")
        print(f"  Unexpected keys (Expected): {len(msg.unexpected_keys)} (Patch Decoder parameters)")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # ==========================================
    # 4. 拟合 GMM (初始化 Base Distribution)
    # ==========================================
    if os.path.exists(args.train_data):
        print(f"Loading training data from {args.train_data} for GMM fitting...")
        train_dataset = RUGDH5Dataset(args.train_data)
        # 只需要部分数据，Batch Size 可以大一点
        gmm_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        
        print("Fitting GMM... (This creates the base distribution for the Flow)")
        model.fit_gmm(gmm_loader)
    else:
        print(f"Error: Training data not found at {args.train_data}")
        return

    # ==========================================
    # 5. 推理与可视化
    # ==========================================
    if not os.path.exists(args.test_data):
        print(f"Error: Test data not found at {args.test_data}")
        return

    test_dataset = RUGDH5Dataset(args.test_data)
    # 随机取一张
    idx = torch.randint(0, len(test_dataset), (1,)).item()
    print(f"Visualizing test image index: {idx}")
    
    image, label = test_dataset[idx]
    image_tensor = image.unsqueeze(0).to(device) # [1, C, H, W]

    # 计算结果
    pred, unc_aleatoric, unc_epistemic = compute_uncertainty(model, image_tensor)
    
    # 准备可视化原图
    raw_img = image_reverse_transforms(image.cpu()).permute(1, 2, 0).numpy()

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # A. Input
    axes[0, 0].imshow(raw_img)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')

    # B. Prediction
    visualize_with_legend(axes[0, 1], pred, "Prediction", label2name)
    axes[0, 1].axis('off')

    # C. Aleatoric
    im_al = axes[1, 0].imshow(unc_aleatoric, cmap='inferno')
    axes[1, 0].set_title("Aleatoric Uncertainty (Entropy)")
    axes[1, 0].axis('off')
    plt.colorbar(im_al, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # D. Epistemic
    im_ep = axes[1, 1].imshow(unc_epistemic, cmap='viridis')
    axes[1, 1].set_title("Epistemic Uncertainty (-LogDensity)")
    axes[1, 1].axis('off')
    plt.colorbar(im_ep, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 保存结果到 test 目录
    save_path = os.path.join(current_dir, 'result_flow_vis.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()