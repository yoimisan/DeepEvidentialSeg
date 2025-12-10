from math import prod
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable
from utils import sample_patches

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






@dataclass(frozen=True)
class DeepEvidentialSegModelConfig(BaseConfig):
    # 论文中提到的训练阶段
    # 0：不训练
    # 1：训练分类头和 reconstruction
    # 2：训练分类头和 normalizing flow
    train_stage: Literal[0, 1, 2] = field(default=1)
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
        # 特征提取器
        self.feature_extractor = config.feature_extractor_config.make_model()
        # 分类头
        self.classification_head = self._build_classification_head(
            in_channels=config.feature_extractor_config.decoder_segmentation_channels,
            hidden_channels=config.classification_head_channels,
            num_classes=config.num_classes
        )
        # Patch Decoder
        if self.train_stage == 1:
            self.patch_decoder = config.patch_decoder_config.make_model()
        elif self.train_stage == 2:
            pass # TODO: 添加 Normalizing Flow 模块
        else:
            self.patch_decoder = None
        # NOTE: 这里可以添加更多的模块，例如 Normaliz flow, feature reconstruction 等
    
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
        layers.append(nn.Softmax(dim=-1))  # 假设我们需要概率输出
        return nn.Sequential(*layers)
    
    def get_current_device(self):
        return next(self.parameters()).device

    def classify(self, images, labels=None, reconstruction_weight=0.5):
        """ 对输入图像进行分类，同时计算 loss
        """
        device= self.get_current_device()
        images = images.to(device)
        if labels is not None:
            labels = labels.to(device)

        loss = None
        info = {} # 额外信息存储

        *_, C, H, W = images.shape
        latent_features = self.feature_extractor.forward(images)
        features = self.feature_extractor.upsample(latent_features, size=images.shape[-2:])  # *CHW
        features = features.permute(*range(len(_)), -2, -1, -3) # *CHW -> *HWC
        class_probs = self.classification_head(features)
        logits = torch.log(class_probs + 1e-8)  # 避免 log(0)

        if labels is not None:
            # 训练部分，先计算交叉熵损失，再计算 reconstruction loss
            # 计算交叉熵损失
            one_hot_labels = nn.functional.one_hot(labels.long(), num_classes=self.config.num_classes)
            class_loss = one_hot_labels * logits
            class_loss = -class_loss.sum(dim=-1).mean()  # 平均损失

            info['class_loss'] = class_loss.item()
            loss = class_loss

            # reconstruction loss
            if self.train_stage == 1:
                stride = H // latent_features.shape[-2]  # 根据特征图大小计算步幅 (应该是 4)
                image_patchs, feature_patchs = sample_patches(
                    images=images,
                    features=latent_features,
                    num_samples=self.config.patch_decoder_config.num_samples,
                    patch_size=self.config.patch_decoder_config.patch_size,
                    stride=stride
                )
                reconstructed_patches = self.patch_decoder(feature_patchs)  # *3phpw
                recon_loss = nn.functional.smooth_l1_loss(
                    reconstructed_patches, 
                    image_patchs,
                    reduction='mean', # 对所有像素求平均
                    beta=1.0, # 使用默认的 beta 值
                ) # 按照论文中的描述，使用 Smooth L1 Loss

                # 总损失
                info['recon_loss'] = recon_loss.item()
                loss += reconstruction_weight * recon_loss

            # 记录信息
            info['total_loss'] = loss.item()

        return loss, logits, info

if __name__ == "__main__":
    model = FPNFeatureExtractor(
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,                 # 此参数在提取特征时不再重要，但在初始化父类时是必须的
        decoder_segmentation_channels=128, # 这是你得到的特征向量的维度
    )