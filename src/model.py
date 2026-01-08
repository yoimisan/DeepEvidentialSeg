from enum import Enum
from math import prod
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable
from .utils import sample_patches
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# ==========================================
# 1. Normalizing Flow 基础模块 (RealNVP 简化版)
# ==========================================
# 论文中使用的是 Residual Flows，但实现复杂且需要谱归一化。
# 这里使用 RealNVP (Affine Coupling Layer) 作为替代，效果近似且更稳定。

class AffineCouplingLayer(nn.Module):
    def __init__(self, num_channels, hidden_dim=64):
        super().__init__()
        self.num_channels = num_channels
        # 将输入分为两半
        self.split_len1 = num_channels // 2
        self.split_len2 = num_channels - self.split_len1

        # 变换网络 s (scale) 和 t (translation)
        self.net = nn.Sequential(
            nn.Linear(self.split_len1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_len2 * 2) # 输出 s 和 t
        )

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.split_len1], x[:, self.split_len1:]

        if not reverse:
            # Forward: z -> x (生成方向，但在密度估计中通常计算 x -> z)
            # 在 NF 密度估计中，我们通常做 x -> z (归一化过程)
            # 论文公式 (4): z = f(x). log p(x) = log p(z) + log |det J|
            
            params = self.net(x1)
            s, t = params.chunk(2, dim=1)
            s = torch.tanh(s) # 限制缩放范围，增加稳定性
            
            # z2 = x2 * exp(s) + t
            z2 = x2 * torch.exp(s) + t
            z1 = x1 # 一半保持不变
            
            z = torch.cat([z1, z2], dim=1)
            log_det = torch.sum(s, dim=1)
            return z, log_det
        else:
            # Reverse: z -> x
            params = self.net(x1)
            s, t = params.chunk(2, dim=1)
            s = torch.tanh(s)
            
            x2 = (x2 - t) * torch.exp(-s)
            x1 = x1
            return torch.cat([x1, x2], dim=1)

class NormalizingFlow(nn.Module):
    def __init__(self, in_channels, num_layers=4, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AffineCouplingLayer(in_channels, hidden_dim))
            # 在层之间交换通道，混合信息
            self.layers.append(PermutationLayer(in_channels))

    def forward(self, x):
        """
        计算 x -> z 的映射以及 Log Determinant Jacobian
        """
        log_det_total = 0
        z = x
        for layer in self.layers:
            if isinstance(layer, AffineCouplingLayer):
                z, log_det = layer(z)
                log_det_total += log_det
            else:
                z = layer(z)
        return z, log_det_total

class PermutationLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        # 固定随机置换
        perm = torch.randperm(num_channels)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', torch.argsort(perm))

    def forward(self, x):
        return x[:, self.perm]

# ==========================================
# 2. 可微 GMM Base Distribution
# ==========================================

class DifferentiableGMM(nn.Module):
    """
    将 sklearn 训练好的 GMM 参数加载进来，实现可微的 log_prob 计算。
    论文 Section IV: Normalizing Flow 的 Base Density 是 p_GMM(z)
    """
    def __init__(self, means, covariances, weights):
        super().__init__()
        self.n_components, self.n_features = means.shape
        
        # 注册为 buffer，不参与梯度更新 (GMM 是预训练固定的)
        self.register_buffer('means', torch.from_numpy(means).float())
        # sklearn 的 covariance 通常是 covariance matrix，这里简化假设为对角阵或全阵
        # 为了数值稳定性，我们通常使用精度矩阵 (precision = inv(cov))
        # 这里假设输入的是 covariance matrices (K, D, D) 或 (K, D)
        if covariances.ndim == 2: # diag
            covs = torch.from_numpy(covariances).float()
            self.register_buffer('precs', 1.0 / (covs + 1e-6))
            self.cov_type = 'diag'
        else: # full
            covs = torch.from_numpy(covariances).float()
            self.register_buffer('precs', torch.inverse(covs + 1e-6 * torch.eye(self.n_features)))
            self.cov_type = 'full'

        self.register_buffer('weights', torch.from_numpy(weights).float())
        
        # 计算 log det (常量)
        if self.cov_type == 'diag':
            self.log_det_covs = torch.sum(torch.log(covs), dim=1) 
        else:
            self.log_det_covs = torch.logdet(covs)

    def log_prob(self, z):
        # z: [Batch, D]
        # 计算 log p(z | k) + log w_k
        # 输出: [Batch] (logsumexp over components)
        
        B, D = z.shape
        K = self.n_components
        
        # 扩展维度以便广播: z -> [B, 1, D], means -> [1, K, D]
        diff = z.unsqueeze(1) - self.means.unsqueeze(0) # [B, K, D]
        
        if self.cov_type == 'diag':
            # (x-u)^T S^-1 (x-u) -> sum((x-u)^2 * prec)
            mahalanobis = torch.sum(diff**2 * self.precs.unsqueeze(0), dim=2) # [B, K]
        else:
            # [B, K, 1, D] @ [1, K, D, D] @ [B, K, D, 1]
            diff_expanded = diff.unsqueeze(-1)
            precs_expanded = self.precs.unsqueeze(0)
            mahalanobis = torch.matmul(torch.matmul(diff.unsqueeze(2), precs_expanded), diff_expanded).squeeze()
            
        # Log Gaussian constant: -0.5 * (D * log(2pi) + log|Sigma| + Mahalanobis)
        log_2pi = D * np.log(2 * np.pi)
        log_probs_k = -0.5 * (log_2pi + self.log_det_covs.unsqueeze(0) + mahalanobis)
        
        # Add weights: log(w_k * N(...)) = log(w_k) + log(N(...))
        total_log_probs = log_probs_k + torch.log(self.weights.unsqueeze(0) + 1e-8)
        
        # LogSumExp over components
        return torch.logsumexp(total_log_probs, dim=1)

# ==========================================
# 3. 集成 Density Estimator
# ==========================================

class DensityEstimator(nn.Module):
    def __init__(self, in_channels, gmm_model=None):
        super().__init__()
        self.flow = NormalizingFlow(in_channels)
        self.base_dist = gmm_model # DifferentiableGMM instance
    
    def forward(self, x):
        # 1. 通过 Flow 将 x 变换到 z
        z, log_det_jacobian = self.flow(x)
        
        # 2. 计算 Base Distribution (GMM) 的 log_prob(z)
        if self.base_dist is not None:
            log_prob_z = self.base_dist.log_prob(z)
        else:
            # 默认为标准正态分布 N(0, I)
            log_prob_z = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
            
        # 3. Change of variable formula: log p(x) = log p(z) + log |det J|
        log_prob_x = log_prob_z + log_det_jacobian
        return log_prob_x

@runtime_checkable
class BaseConfig(Protocol):
    def make_model(self) -> nn.Module:
        ...


@dataclass(frozen=True)
class FPNFeatureExtractorConfig(BaseConfig):
    encoder_name: str = field(default="resnet50")
    encoder_weights: str = field(default="imagenet")
    in_channels: int = 3
    decoder_segmentation_channels: int = 128  # 输出特征的通道数

    def make_model(self, *, device='cpu') -> 'FPNFeatureExtractor':
        extractor = FPNFeatureExtractor(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=1,  # 此参数在提取特征时不再重要，但在初始化父类时是必须的
            decoder_segmentation_channels=self.decoder_segmentation_channels,
        ) 
        return extractor.to(device)

class FPNFeatureExtractor(smp.FPN):
    def forward(self, x):
        """
        前向传播，提取多尺度特征并融合
        """
        self.check_input_shape(x)

        # 1. 编码器 (Encoder) 提取多尺度特征
        features = self.encoder(x)

        # 2. 解码器 (Decoder) 融合特征
        # FPN Decoder 输出的是融合后的特征金字塔
        decoder_output = self.decoder(features) # 大小是原图的 1/4
        # 如果你需要原始尺寸，可能还需要根据 output_stride 进行上采样
        return decoder_output
    
    def upsample(self, x, size):
        """
        上采样到指定大小
        """
        return nn.functional.interpolate(
            x,
            size=size,
            mode='bilinear',
            align_corners=False
        )
    
    def get_pixel_wise_features(self, x):
        """
        获取逐像素特征向量
        输入:
            x: [Batch, C, H, W]
        输出:
            features: [Batch, decoder_segmentation_channels, H, W]
        """
        image_size = x.shape[-2:]
        decoder_output = self.forward(x)
        # 调整维度顺序以符合逐像素特征的习惯表示
        return self.upsample(decoder_output, size=image_size)


@dataclass(frozen=True)
class PatchDecoderConfig(BaseConfig):
    in_channels: int = 128
    patch_size: int = 60
    num_samples: int = 8

    def make_model(self, *, device='cpu') -> 'PatchDecoder':
        model = PatchDecoder(
            in_channels=self.in_channels,
            patch_size=self.patch_size
        )
        return model.to(device)

class PatchDecoder(nn.Module):
    def __init__(self, in_channels: int=128, patch_size: int = 60):
        super().__init__()
        self.patch_size = patch_size
        
        # 1. 投影层：将 1x1 的特征向量投影成 5x5 的特征图
        # 假设中间通道数设为 64，这一步参数量有点大，但对于辅助任务可以接受
        self.base_size = 5
        self.mid_channels = 64
        # (*, in_channels) -> (*, mid_channels * 5 * 5)
        self.projection = nn.Linear(in_channels, self.mid_channels * self.base_size * self.base_size)
        
        # 2. 上采样网络
        self.decoder = nn.Sequential(
            # 5x5 -> 10x10
            # (*, mid_channels, 5, 5) -> (*, 64, 10, 10)
            nn.ConvTranspose2d(self.mid_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SELU(inplace=True),
            
            # 10x10 -> 20x20
            # (*, 64, 10, 10) -> (*, 32, 20, 20)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SELU(inplace=True),
            
            # 20x20 -> 60x60 (关键步：stride=3)
            # (*, 32, 20, 20) -> (*, 3, 60, 60)
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=3, padding=1),
            
            # 最后一层用 Sigmoid (如果你的图片归一化是 0~1) 或者 Tanh (如果是 -1~1)
            # 假设你做了 imagenet normalize，这里最好不加激活，让 loss 去拟合数值
            # 或者输出层不加激活，直接算 Loss
        )

    def forward(self, x):
        # x shape: [*, C]
        *bs, _ = x.shape
        x = self.projection(x)
        x = x.view(prod(bs), self.mid_channels, self.base_size, self.base_size)
        output = self.decoder(x)
        return output.view(*bs, 3, self.patch_size, self.patch_size)




class TRAIN_STAGE(Enum):
    NO_TRAIN = 0
    RECONSTRUCTION = 1
    NORMALIZING_FLOW = 2

@dataclass(frozen=True)
class DeepEvidentialSegModelConfig(BaseConfig):
    # 论文中提到的训练阶段
    # 0：不训练
    # 1：训练分类头和 reconstruction
    # 2：训练分类头和 normalizing flow
    train_stage: TRAIN_STAGE = field(default=TRAIN_STAGE.RECONSTRUCTION)
    gmm_components: int = 20 

    # 特征提取函数的 config
    feature_extractor_config: FPNFeatureExtractorConfig = field(
        default_factory=FPNFeatureExtractorConfig
    )
    # classification head
    classification_head_channels: tuple[int, ...] = field(default=(128, 64, 32))
    num_classes: int = 25 # 在 data/RUGD/RUGD_annotations/RUGD_annotation-colormap.txt 中有说明
    # Patch Decoder
    patch_decoder_config: PatchDecoderConfig = field(
        default_factory=PatchDecoderConfig
    )
    # NOTE: 其他模型参数可以在这里添加


    def make_model(self, *, device='cpu') -> 'DeepEvidentialSegModel':
        model = DeepEvidentialSegModel(config=self)
        return model.to(device)

class DeepEvidentialSegModel(nn.Module):
    """Overall Model
    """
    def __init__(self, config: DeepEvidentialSegModelConfig):
        super().__init__()
        self.config = config
        self.train_stage = config.train_stage
        feat_dim = config.feature_extractor_config.decoder_segmentation_channels

        # 特征提取器
        self.feature_extractor = config.feature_extractor_config.make_model()
        # 分类头
        self.classification_head = self._build_classification_head(
            in_channels=config.feature_extractor_config.decoder_segmentation_channels,
            hidden_channels=config.classification_head_channels,
            num_classes=config.num_classes
        )
        # Patch Decoder
        if self.train_stage == TRAIN_STAGE.RECONSTRUCTION:
            self.patch_decoder = config.patch_decoder_config.make_model()
        elif self.train_stage == TRAIN_STAGE.NORMALIZING_FLOW:
            self.patch_decoder = None
            pass # TODO: 添加 Normalizing Flow 模块
        else:
            self.patch_decoder = None
        if self.train_stage == TRAIN_STAGE.NORMALIZING_FLOW:
            # 注意：实际使用时，你需要先加载 Stage 1 训练好的权重
            self.density_estimator = DensityEstimator(in_channels=feat_dim, gmm_model=None)
        else:
            self.density_estimator = None
    def fit_gmm(self, dataloader):
        """
        在 Stage 2 开始前运行。提取所有训练数据的特征，拟合 GMM，然后初始化 Base Dist。
        """
        print("Extracting features for GMM fitting...")
        self.eval()
        all_features = []
        device = self.get_current_device()
        
        # 随机采样一部分像素以节省内存 (例如取 1%)
        with torch.no_grad():
            for images, _ in tqdm(dataloader):
                images = images.to(device)
                features = self.feature_extractor.get_pixel_wise_features(images) # [B, C, H, W]
                # 展平并采样
                features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
                indices = torch.randperm(features.shape[0])[:1000] # 每张图采样 1000 个点
                all_features.append(features[indices].cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        print(f"Fitting GMM with {self.config.gmm_components} components on {all_features.shape[0]} samples...")
        
        gmm = GaussianMixture(n_components=self.config.gmm_components, covariance_type='full', verbose=1)
        gmm.fit(all_features)
        
        # 初始化 DifferentiableGMM 并赋值给 DensityEstimator
        diff_gmm = DifferentiableGMM(gmm.means_, gmm.covariances_, gmm.weights_)
        self.density_estimator.base_dist = diff_gmm.to(device)
        print("GMM fitted and loaded.")

    def evidential_loss(self, logits, labels, log_prob_x):
        """
        实现论文公式 (7): Bayesian Loss
        Dir(p | alpha) where alpha = 1 + N * p(x) * softmax(logits)
        """
        probs = torch.softmax(logits, dim=-1) # beta_phi(x)
        density = torch.exp(log_prob_x).unsqueeze(1) # p_theta(x)
        
        # 证据 N (scale factor)，论文没给具体数值，通常需要调节或者设为 dataset size 的 scaling
        N = 10.0 
        
        # 计算 Alpha 参数: alpha_c = 1 + N * p(x) * prob_c
        alphas = 1 + N * density * probs
        
        # 下面是 Dirichlet 损失的常见实现 (Type 2 Maximum Likelihood / Expected Cross Entropy)
        # Loss = E_Dir[-log(p_y)] + lambda * KL(Dir || Uniform)
        
        S = torch.sum(alphas, dim=1, keepdim=True)
        
        # 1. Expected Cross Entropy term (Likelihood)
        # E[log(p_y)] = psi(alpha_y) - psi(S)
        one_hot = F.one_hot(labels, num_classes=self.config.num_classes).float()
        loss_ce = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alphas)), dim=1)
        
        # 2. Regularization term (Entropy / KL)
        # 论文提出 lambda = 1 / (N * p(x))
        lam = 1.0 / (N * density.squeeze(1) + 1e-6)
        
        # KL Divergence between Dir(alpha) and Dir(1,1,...1)
        # 这是一个简化的 KL，用于惩罚证据过高（Confidence Penalty）
        # 具体的 Dirichlet KL 公式比较长，这里用常见的近似：
        # 尽量让 alpha 接近 1 (即不确定) 当预测错误时
        # 为了简化复现，这里使用 NatPN 论文中的 KL term 或简单的 Entropy 正则
        
        # 使用论文公式 (7) 的描述: H(Dir(n))
        # 这里用 PyTorch 的 Dirichlet 分布计算熵
        dir_dist = torch.distributions.Dirichlet(alphas)
        entropy = dir_dist.entropy()
        
        loss_total = loss_ce - lam * entropy
        return (
            loss_total.mean(),           # 用于反向传播的 Tensor
            loss_ce.mean().item(),        # 似然损失项 (数值)
            (lam * entropy).mean().item() # 正则项 (数值)
        )
    
    def _build_classification_head(self, in_channels, hidden_channels, num_classes):
        """ 构建分类头
        """
        layers = []
        current_channels = in_channels
        for hidden_channel in hidden_channels:
            layers.append(nn.Linear(current_channels, hidden_channel))
            layers.append(nn.SELU(inplace=True))
            current_channels = hidden_channel
        layers.append(nn.Linear(current_channels, num_classes))
        # layers.append(nn.Softmax(dim=-1))  # 假设我们需要概率输出
        return nn.Sequential(*layers)
    
    def get_current_device(self):
        return next(self.parameters()).device

    def classify(self, images, labels=None, reconstruction_weight=0.5):
        """
        完整的分类与训练前向过程
        """
        device = self.get_current_device()
        images = images.to(device)
        if labels is not None:
            labels = labels.to(device)

        loss = None
        info = {} 
        
        # =================================================
        # 1. 特征提取
        # =================================================
        # [B, C, H, W]
        # 在 Stage 2，为了稳定密度估计，通常冻结 encoder 或将其 detach
        if self.train_stage == TRAIN_STAGE.NORMALIZING_FLOW:
            with torch.no_grad():
                latent_features = self.feature_extractor.forward(images)
                features_upsampled = self.feature_extractor.upsample(latent_features, size=images.shape[-2:])
        else:
            latent_features = self.feature_extractor.forward(images)
            features_upsampled = self.feature_extractor.upsample(latent_features, size=images.shape[-2:])
        
        # =================================================
        # 2. 准备数据形状 [B, C, H, W] -> [N, C]
        # =================================================
        B, C, H, W = features_upsampled.shape
        # Permute to [B, H, W, C] then flatten to [B*H*W, C]
        # 这对于全连接层和 Flow 都是必要的
        features_flat = features_upsampled.permute(0, 2, 3, 1).reshape(-1, C)
        
        # =================================================
        # 3. 分类头预测
        # =================================================
        logits_flat = self.classification_head(features_flat) # [N, Num_Classes]
        
        # 为了输出，reshape 回图片格式
        logits_img = logits_flat.view(B, H, W, -1).permute(0, 3, 1, 2) # [B, Classes, H, W]

        # =================================================
        # 4. 训练逻辑分支
        # =================================================
        if labels is not None:
            labels_flat = labels.flatten().long()
            
            # --- Stage 1: Reconstruction + Standard Classification ---
            if self.train_stage == TRAIN_STAGE.RECONSTRUCTION:
                # A. 分类损失 (Standard Cross Entropy)
                # 使用 PyTorch 内置的 CrossEntropyLoss (包含 log_softmax)
                cls_loss = F.cross_entropy(logits_flat, labels_flat)
                
                # B. 重建损失 (Patch Reconstruction)
                # 计算步幅 stride (原图/特征图)
                stride = H // latent_features.shape[-2] 
                
                # 采样 Patch (注意：这里传的是未上采样的 latent_features 以节省计算，
                # 但 sample_patches 内部需要处理尺寸对应关系，或者这里直接传 features_upsampled 也可以，
                # 但原论文 PatchDecoder 是设计给 latent features 的)
                image_patchs, feature_patchs = sample_patches(
                    images=images,
                    features=latent_features, # [B, C, h, w]
                    num_samples=self.config.patch_decoder_config.num_samples,
                    patch_size=self.config.patch_decoder_config.patch_size,
                    stride=stride
                )
                
                # [B, Samples, C] -> [B*Samples, C]
                # PatchDecoder 输入是 [*, C]
                features_for_recon = feature_patchs.flatten(0, 1) 
                target_patches = image_patchs.flatten(0, 1)
                
                reconstructed_patches = self.patch_decoder(features_for_recon)
                
                recon_loss = F.smooth_l1_loss(
                    reconstructed_patches, 
                    target_patches,
                    beta=1.0
                )
                
                loss = cls_loss + reconstruction_weight * recon_loss
                
                info['class_loss'] = cls_loss.item()
                info['recon_loss'] = recon_loss.item()

            # --- Stage 2: Normalizing Flow + Evidential Loss ---
            elif self.train_stage == TRAIN_STAGE.NORMALIZING_FLOW:
                if self.density_estimator is None:
                    raise RuntimeError("Density Estimator not initialized. Call fit_gmm first.")

                # =================================================
                # 像素采样逻辑：防止 OOM 且提高训练效率
                # =================================================
                # 设定每张图片采样的像素点数。通常 1024 或 2048 足够了。
                num_samples = 1024 
                feat_flat = features_upsampled.permute(0, 2, 3, 1).reshape(-1, C)
                
                # 随机采样 1024 个点
                indices = torch.randperm(feat_flat.size(0))[:num_samples]
                sampled_features = feat_flat[indices].detach()
                
                # 只对这 1024 个点跑密度估计
                log_prob_x = self.density_estimator(sampled_features) 
                
                # 对应的 logits 也采样
                sampled_logits = logits_flat[indices]
                sampled_labels = labels_flat[indices] # 如果有 labels 的话
                
                # B. Evidential Loss
                # 联合优化 Flow 和 Classifier (Fine-tune)
                loss, loss_lik, loss_reg = self.evidential_loss(
                    sampled_logits, 
                    sampled_labels, 
                    log_prob_x
                )
                
                info['loss_likelihood'] = loss_lik
                info['loss_reg'] = loss_reg
                info['mean_log_density'] = log_prob_x.mean().item()
            
            else:
                # 仅推断或仅分类训练
                loss = F.cross_entropy(logits_flat, labels_flat)

            info['total_loss'] = loss.item()

        return loss, logits_img, info

if __name__ == "__main__":
    model = FPNFeatureExtractor(
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,                 # 此参数在提取特征时不再重要，但在初始化父类时是必须的
        decoder_segmentation_channels=128, # 这是你得到的特征向量的维度
    )