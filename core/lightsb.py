import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GMMPotential(nn.Module):
    """
    Parameterize Schrödinger potential using Gaussian Mixture Model.
    Formula: v_theta(x) = Sum alpha_k * N(x | mu_k, S_k)
    """
    def __init__(self, dim, n_components=10, epsilon=1.0):
        super().__init__()
        self.dim = dim
        self.K = n_components
        self.epsilon = epsilon

        # Learnable parameters
        self.logits = nn.Parameter(torch.randn(self.K)) # For alpha via softmax
        self.means = nn.Parameter(torch.randn(self.K, dim))
        # Covariance factor (Cholesky) to ensure positive definiteness
        self.cov_factors = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(self.K, 1, 1))

    def get_distribution_params(self):
        alphas = torch.softmax(self.logits, dim=0)
        
        # S_k = L * L^T + epsilon * I
        covs = torch.bmm(self.cov_factors, self.cov_factors.transpose(1, 2))
        covs = covs + (torch.eye(self.dim, device=covs.device) * 1e-6).unsqueeze(0)
        
        return alphas, self.means, covs

    def log_prob(self, x):
        """Compute log(v_theta(x))"""
        alphas, means, covs = self.get_distribution_params()
        
        # x: (Batch, Dim) -> (Batch, K, Dim)
        x_expanded = x.unsqueeze(1).expand(-1, self.K, -1)
        
        # Multivariate Gaussian Log PDF manually or via Dist
        # Using PyTorch distributions for stability
        mix = torch.distributions.Categorical(probs=alphas)
        comp = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covs)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        
        return gmm.log_prob(x)

    def compute_normalization_constant(self):
        """
        c_theta(x0) analytically. 
        Note: The PDF formulation suggests simulation-free training using analytic formulas.
        For standard LightSB, c is often related to the integral.
        Here we return a placeholder for the integral of the unnormalized potential if needed,
        but LightSB objective usually involves E[log c] - E[log v].
        """
        # In many LightSB formulations, c_theta is the dual potential. 
        # For this specific prompt, we assume the Single-Stage Non-Minimax Loss.
        # L(theta) = E_P0 [log c(x0)] - E_P1 [log v(x1)]
        # If c and v are conjugate, or if c is computed analytically.
        pass

class LightSBTrainer:
    def __init__(self, dim, n_components=20, lr=1e-3):
        self.model = GMMPotential(dim, n_components)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, x0_batch, x1_batch):
        """
        Single stage non-minimax loss:
        L(theta) = E_{x0}[log c_theta(x0)] - E_{x1}[log v_theta(x1)]
        
        *Assumption*: The PDF implies c_theta is the normalization constant.
        For a GMM potential, the normalization constant over a Gaussian reference measure
        can often be computed analytically (convolution of Gaussians).
        """
        self.optimizer.zero_grad()
        
        # 1. Calculate log v_theta(x1)
        log_v_x1 = self.model.log_prob(x1_batch)
        
        # 2. Calculate log c_theta(x0) (Analytic Convolution)
        # If reference is Brownian Motion, P(x1|x0) is Gaussian.
        # c(x0) = Integral P(x1|x0) v(x1) dx1
        # Convolution of Gaussian (transition) and GMM (potential) is a GMM.
        log_c_x0 = self._analytic_convolution(x0_batch)
        
        loss = log_c_x0.mean() - log_v_x1.mean()
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _analytic_convolution(self, x0):
        """
        Computes log( Integral N(x1 | x0, t) * Sum alpha N(x1 | mu, S) dx1 )
        Result is log of a new GMM.
        """
        alphas, mus, Sigmas = self.model.get_distribution_params()
        
        # Assuming T=1 for specific step, or passed as param. Let's assume unit time for bridge step.
        # Transition kernel N(x1 | x0, I) (simplified for standard BM)
        
        # New Means: mu_k
        # New Covs: Sigma_k + I
        
        x0_expanded = x0.unsqueeze(1).expand(-1, self.model.K, -1)
        
        new_covs = Sigmas + torch.eye(self.model.dim, device=x0.device).unsqueeze(0)
        
        # Evaluate Mix Gaussian at x0
        mix = torch.distributions.Categorical(probs=alphas)
        comp = torch.distributions.MultivariateNormal(loc=mus, covariance_matrix=new_covs)
        gmm_conv = torch.distributions.MixtureSameFamily(mix, comp)
        
        return gmm_conv.log_prob(x0)

    def get_drift(self, t, x):
        """
        Analytical Drift: alpha*(t, x) = grad_x log(v_theta(x)) + path_dependent_term
        """
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        log_v = self.model.log_prob(x_tensor.unsqueeze(0))
        grad = torch.autograd.grad(log_v, x_tensor)[0]
        return grad.detach().numpy()