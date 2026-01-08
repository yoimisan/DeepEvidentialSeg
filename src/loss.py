import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.log_beta_prior = 0.0 # log(1) = 0, Dirichlet prior beta=1 (Uniform)

    def forward(self, probs, log_density, targets, evidence_scale_N=1.0):
        """
        参数:
            probs: 分类器输出的概率分布 (Batch, C) -> beta_phi
            log_density: Flow 输出的对数密度 (Batch,) -> log p_theta
            targets: 真实标签 (Batch,)
            evidence_scale_N: 论文中的常数 N，用于缩放密度
        """
        # 1. 计算 Dirichlet 参数 alpha
        # alpha = 1 + N * p(x) * probs
        # 为了数值稳定性，我们在对数域操作一部分，但在加 1 时需要回到线性域
        density = torch.exp(log_density)
        evidence = evidence_scale_N * density.unsqueeze(-1) * probs
        alpha = evidence + 1.0 
        
        # S = sum(alpha) = N * density + K (因为 sum(probs)=1, sum(1)=K)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 2. 第一项: Expected Cross Entropy (ECE)
        # E[log(p_c)] = psi(alpha_c) - psi(S)
        # Loss_ECE = - sum( y_one_hot * (psi(alpha) - psi(S)) )
        
        # 获取对应真实标签的 alpha_c
        # creating one_hot: (Batch, C)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # E[log(p_y)]
        expected_log_prob = torch.digamma(alpha) - torch.digamma(S)
        loss_ece = -torch.sum(targets_one_hot * expected_log_prob, dim=1)
        
        # 3. 第二项: Entropy Regularization (KL Divergence form)
        # 论文 Eq 7 中的 entropy regularization term 实际上等价于 KL(Dir(alpha) || Dir(prior=1))
        # 这是一个正则项，防止分布过于尖锐（overconfident）
        
        # 计算 Dir(alpha) 的熵是不容易优化的，通常做法是最小化 KL 散度到均匀分布
        # 或者直接按照论文公式写: - lambda * H(Dir)
        # 这里我们使用 NatPN 和论文中推荐的 KL 散度形式，效果一致且更稳定
        # KL(Dir(alpha) || Dir(1))
        
        # beta_prior = 1
        # KL = log_gamma(S) - sum(log_gamma(alpha)) + sum((alpha - 1) * (digamma(alpha) - digamma(S)))
        
        kl_term = torch.lgamma(S).squeeze(-1) - torch.sum(torch.lgamma(alpha), dim=1) \
                  + torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
        
        # 4. 计算自适应系数 lambda
        # lambda = 1 / (N * p(x))
        # 加上 epsilon 防止除零
        inv_evidence = 1.0 / (evidence_scale_N * density + 1e-8)
        
        # 5. 总损失
        # Loss = ECE + lambda * KL
        total_loss = loss_ece + inv_evidence * kl_term
        
        return total_loss.mean()

class PatchReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=1.0) # 论文提及使用 Smooth-L1

    def forward(self, reconstructed_patches, real_patches):
        return self.loss_fn(reconstructed_patches, real_patches)