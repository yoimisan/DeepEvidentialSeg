import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
# 1. 谱归一化线性层 (Spectral Norm Linear)
# ------------------------------------------------------------------
class SpectralNormLinear(nn.Module):
    def __init__(self, in_features, out_features, coeff=0.97, n_power_iterations=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff # Lipschitz constant upper bound (< 1)
        self.n_power_iterations = n_power_iterations
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 注册 buffer 用于存储 power iteration 的向量 u
        self.register_buffer("u", torch.randn(1, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def _compute_sigma(self):
        # Power iteration to estimate spectral norm (largest singular value)
        weight_mat = self.weight
        u = self.u
        v = None
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # v = u @ W
                v = torch.nn.functional.normalize(torch.matmul(u, weight_mat), dim=1, eps=1e-3)
                # u = v @ W.T
                u = torch.nn.functional.normalize(torch.matmul(v, weight_mat.t()), dim=1, eps=1e-3)
            self.u.copy_(u) # update buffer
            
        # sigma = u @ W @ v.T
        sigma = torch.matmul(torch.matmul(u, weight_mat), v.t())
        return sigma

    def forward(self, x):
        # 每次前向传播都重新计算 sigma 并归一化权重
        sigma = self._compute_sigma()
        
        # 这里的关键是让权重矩阵 W 的谱范数 <= coeff
        # 如果 sigma > coeff, 我们需要缩放 W
        # factor = coeff / sigma
        
        # 使用 soft-scaling 可能会更稳定，但这里使用直接缩放
        # W_sn = W * (coeff / sigma)
        
        scale = self.coeff / sigma
        weight_sn = self.weight * scale
        
        return F.linear(x, weight_sn, self.bias)

# ------------------------------------------------------------------
# 2. Residual Block
# f(x) = x + g(x)
# ------------------------------------------------------------------
class ResidualFlowBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, coeff=0.97):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2
            
        # g(x) 是一个简单的 MLP
        self.g = nn.Sequential(
            SpectralNormLinear(dim, hidden_dim, coeff=coeff),
            nn.ELU(), # ELU is Lipschitz continuous with K=1
            SpectralNormLinear(hidden_dim, dim, coeff=coeff)
        )

    def forward(self, x):
        return x + self.g(x)

    def g_forward(self, x):
        return self.g(x)

    def log_det_estimate(self, x, n_power_series=5):
        """
        使用幂级数估算 log|det(I + J_g)|
        log(det(I + Jg)) = Tr(ln(I + Jg)) 
                         = Sum_{k=1...inf} (-1)^(k+1) * Tr(Jg^k) / k
        利用 Hutchinson Trace Estimator: Tr(A) = E[v^T A v]
        """
        # 生成随机向量 v (Hutchinson estimator)
        v = torch.randn_like(x)
        
        # Neumann series sum
        log_det = 0
        w = v
        
        for k in range(1, n_power_series + 1):
            # 计算 Jg * w using vector-Jacobian product (autograd)
            # 这里的计算比较 tricky，需要用 autograd.grad
            
            with torch.enable_grad():
                # 必须开启梯度以计算雅可比积
                if not x.requires_grad:
                    x_in = x.detach().requires_grad_(True)
                else:
                    x_in = x
                
                g_val = self.g(x_in)
                
                # 计算 vector-Jacobian product: w^T * J
                # torch.autograd.grad(outputs, inputs, grad_outputs=w) computes J^T * w
                # 但我们需要 J * w。
                # 在 iResNet 的推导中，通常不需要显式区分 J 和 J^T 的 trace，因为 Tr(A) = Tr(A^T)
                
                # 计算 w_next = J^T * w
                w = torch.autograd.grad(
                    outputs=g_val, 
                    inputs=x_in, 
                    grad_outputs=w, 
                    create_graph=self.training, # 训练时需要梯度穿透
                    retain_graph=True
                )[0]
            
            term = torch.sum(w * v, dim=1) # v^T * (J^k * v) approximation
            
            if k % 2 == 1:
                log_det += term / k
            else:
                log_det -= term / k
                
        return log_det


# ------------------------------------------------------------------
# 3. 完整的 Normalizing Flow 模型
# ------------------------------------------------------------------
class GMMResidualFlow(nn.Module):
    def __init__(self, dim, n_gmm_components, n_layers=4):
        super().__init__()
        from .gmm import GaussianMixtureModel # 假设 gmm.py 在同一级
        
        self.dim = dim
        self.layers = nn.ModuleList([
            ResidualFlowBlock(dim, hidden_dim=dim*2, coeff=0.90) # coeff < 1 是严格必须的
            for _ in range(n_layers)
        ])
        
        self.base_dist = GaussianMixtureModel(n_gmm_components, dim)

    def forward(self, x):
        """
        计算 Log Density: log p(x)
        log p(x) = log p_base(z) + sum(log_det)
        其中 z = f(x)  (注意：这篇论文的定义方向)
        Wait: 论文公式 (4) 采样方向是 z -> x。
        那么密度估计方向 (Density Estimation, Eq 6) 是 z_t = f^{-1}(x)。
        
        但是 iResNet 的结构 F(x) = x + g(x) 容易计算 F(x) 和 log_det(J_F)。
        如果论文定义 x = F(z) = z + g(z)，那么求逆 x -> z 是不动点迭代，比较慢。
        
        **仔细看论文公式 (5) 和 (6)**:
        Eq (6): z_t = f^{-1}(x) ... 
        这意味着 x 是复杂分布，z 是简单分布。
        
        如果我们要高效计算 log_density，我们希望 x -> z 的方向是显式的 ResNet。
        即定义网络为: z = x + g(x)。这样计算 log_det 就很容易。
        如果这样定义，那么采样 (z->x) 就需要逆变换（不动点迭代）。
        
        鉴于我们的任务主要是 Density Estimation (不确定性估计)，而不是生成图片。
        **我们应该把网络建模为 x -> z 的方向为显式 Residual Block。**
        即: z = layer(x) = x + g(x)
        此时: log p(x) = log p(z) + log |det J|
        """
        
        z = x
        log_det_sum = 0
        
        for layer in self.layers:
            # 累加每一层的 log determinant
            # J = I + Jg
            log_det = layer.log_det_estimate(z)
            log_det_sum += log_det
            
            # 变换特征
            z = layer(z)
            
        # 计算 Base Distribution 的 Log Probability
        log_prob_base = self.base_dist(z)
        
        # Total log prob
        log_prob_x = log_prob_base + log_det_sum
        
        return log_prob_x