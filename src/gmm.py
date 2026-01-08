import torch
import torch.nn as nn
import numpy as np

class GaussianMixtureModel(nn.Module):
    """
    一个固定的 GMM 分布模块，用于 Normalizing Flow 的 Base Distribution。
    它不进行训练（通常由 sklearn 预训练好），但在 PyTorch 中用于计算 log_prob。
    """
    def __init__(self, n_components: int, n_features: int):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        # 注册为 buffer，意味着它们是模型的一部分但不会在反向传播中更新（除非手动开启 requires_grad）
        self.register_buffer("means", torch.zeros(n_components, n_features))
        # 我们存储精度的 Cholesky 分解形式 (L)，因为计算马氏距离需要精度矩阵
        # Precision = L @ L.T
        self.register_buffer("precisions_cholesky", torch.eye(n_features).unsqueeze(0).repeat(n_components, 1, 1))
        self.register_buffer("log_weights", torch.zeros(n_components))
        
        # 常数项: -D/2 * log(2pi)
        self.register_buffer("log_2pi_factor", torch.tensor(-0.5 * n_features * np.log(2 * np.pi)))

    def load_from_sklearn(self, gmm_sklearn):
        """
        从 sklearn.mixture.GaussianMixture 加载参数
        """
        assert gmm_sklearn.n_components == self.n_components
        assert gmm_sklearn.n_features_in_ == self.n_features

        # 1. Weights
        # sklearn 存储的是 weights_, 我们存 log_weights
        weights = torch.from_numpy(gmm_sklearn.weights_).float()
        self.log_weights.copy_(torch.log(weights + 1e-6))

        # 2. Means
        self.means.copy_(torch.from_numpy(gmm_sklearn.means_).float())

        # 3. Precisions Cholesky
        # sklearn 可能存储 covariances_ 或 precisions_cholesky_
        if hasattr(gmm_sklearn, 'precisions_cholesky_'):
            prec_chol = torch.from_numpy(gmm_sklearn.precisions_cholesky_).float()
        else:
            # 如果没有直接提供 cholesky，手动计算
            # precisions = inverse(covariance)
            cov = torch.from_numpy(gmm_sklearn.covariances_).float()
            # 针对 'full' covariance type
            prec = torch.linalg.inv(cov) 
            prec_chol = torch.linalg.cholesky(prec)
        
        self.precisions_cholesky.copy_(prec_chol)
        print(f"GMM parameters loaded from sklearn. Components: {self.n_components}, Dim: {self.n_features}")

    def forward(self, x):
        """
        计算 log p(x)
        x: [Batch, D]
        Return: [Batch]
        """
        # x: (B, D) -> (B, 1, D)
        # means: (K, D) -> (1, K, D)
        # diff: (B, K, D)
        bs = x.shape[0]
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)

        # 计算马氏距离 (Mahalanobis distance) 的一部分
        # 我们需要计算 (x-mu)^T * Sigma^{-1} * (x-mu)
        # Sigma^{-1} = L @ L.T
        # 所以项为: (L.T @ (x-mu))^2
        
        # precisions_cholesky shape: (K, D, D) -> (1, K, D, D)
        # diff.unsqueeze(-1) shape: (B, K, D, 1)
        # y = L.T @ diff
        # torch.matmul 对最后两维进行操作
        # 注意：sklearn 的 precisions_cholesky 存储的是下三角矩阵 L，满足 Precision = L @ L.T
        # 但在计算 log_prob 时，通常实现是 y = diff @ L (如果是行向量x) 或者 L.T @ diff (如果是列向量)
        # 这里我们直接利用 PyTorch 的 broadcast
        
        # (1, K, D, D) @ (B, K, D, 1) -> (B, K, D, 1)
        y = torch.matmul(self.precisions_cholesky.unsqueeze(0), diff.unsqueeze(-1))
        y = y.squeeze(-1) # (B, K, D)
        
        # log_prob_det = sum(log(diag(L)))
        # L 的对角线元素乘积等于 sqrt(|Precision|)
        # log_det = sum(log(diag))
        log_det_precision = torch.sum(torch.log(torch.diagonal(self.precisions_cholesky, dim1=1, dim2=2)), dim=1) # (K,)
        
        # log_prob_exp = -0.5 * y^T * y
        log_prob_exp = -0.5 * torch.sum(y ** 2, dim=2) # (B, K)

        # Component log probs
        # log p(x|k) = const + 0.5 * log|P| - 0.5 * (x-u)'P(x-u)
        log_prob_component = self.log_2pi_factor + log_det_precision.unsqueeze(0) + log_prob_exp # (B, K)

        # Mixture log prob
        # log p(x) = log sum_k ( w_k * p(x|k) )
        #          = log sum_k ( exp(log_w_k + log_p_k) )
        total_log_prob = torch.logsumexp(self.log_weights.unsqueeze(0) + log_prob_component, dim=1) # (B,)

        return total_log_prob