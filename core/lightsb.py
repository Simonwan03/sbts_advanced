import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GMMPotential(nn.Module):
    """
    Paramétrisation du potentiel de Schrödinger via un Modèle de Mélange Gaussien (GMM).
    Formule: v_theta(x) = Sum alpha_k * N(x | mu_k, S_k)
    
    [Update]: Ajout de contraintes de covariance minimale pour éviter l'explosion du gradient.
    """
    def __init__(self, dim, n_components=20, min_cov=1e-3):
        super().__init__()
        self.dim = dim
        self.K = n_components
        self.min_cov = min_cov # Seuil plancher pour la variance

        # Paramètres apprenables
        self.logits = nn.Parameter(torch.randn(self.K)) # Pour alpha via softmax
        self.means = nn.Parameter(torch.randn(self.K, dim))
        # Facteur de Covariance (Cholesky)
        self.cov_factors = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(self.K, 1, 1))

    def get_distribution_params(self):
        alphas = torch.softmax(self.logits, dim=0)
        
        # S_k = L * L^T + epsilon * I
        # Construction robuste de la matrice de covariance
        covs = torch.bmm(self.cov_factors, self.cov_factors.transpose(1, 2))
        
        # Ajout d'un "Jitter" robuste pour empêcher la singularité
        eye = torch.eye(self.dim, device=covs.device).unsqueeze(0)
        covs = covs + eye * self.min_cov
        
        return alphas, self.means, covs

    def log_prob(self, x):
        """Calcule log(v_theta(x))"""
        alphas, means, covs = self.get_distribution_params()
        
        # x: (Batch, Dim) -> (Batch, K, Dim) (Gestion via broadcasting interne de MixSameFamily)
        
        # Utilisation des distributions PyTorch pour la stabilité numérique
        mix = torch.distributions.Categorical(probs=alphas)
        comp = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covs)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        
        return gmm.log_prob(x)

class LightSBTrainer:
    def __init__(self, dim, n_components=20, lr=1e-3, min_cov=1e-2):
        self.model = GMMPotential(dim, n_components, min_cov=min_cov)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.dim = dim

    def train_step(self, x0_batch, x1_batch):
        """
        Objectif: L(theta) = E_{x0}[log c(x0)] - E_{x1}[log v(x1)]
        """
        self.optimizer.zero_grad()
        
        # 1. log v_theta(x1)
        log_v_x1 = self.model.log_prob(x1_batch)
        
        # 2. log c_theta(x0) (Convolution Analytique)
        log_c_x0 = self._analytic_convolution(x0_batch)
        
        # Maximiser la vraisemblance => Minimiser le négatif
        loss = log_c_x0.mean() - log_v_x1.mean()
        
        loss.backward()
        # Clipping de gradient sur les paramètres du réseau eux-mêmes
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def _analytic_convolution(self, x0):
        """
        Calcule log( Intégrale P(x1|x0) * v(x1) dx1 )
        Hypothèse: P(x1|x0) est Brownien standard (ou proche).
        La convolution d'une Gaussienne et d'un GMM est un nouveau GMM.
        """
        alphas, mus, Sigmas = self.model.get_distribution_params()
        
        # Nouvelle Covariance: Sigma_k + I (pour dt=1, simplifié)
        # Dans un cadre complet, ce serait Sigma_k + dt * I
        new_covs = Sigmas + torch.eye(self.dim, device=x0.device).unsqueeze(0)
        
        mix = torch.distributions.Categorical(probs=alphas)
        comp = torch.distributions.MultivariateNormal(loc=mus, covariance_matrix=new_covs)
        gmm_conv = torch.distributions.MixtureSameFamily(mix, comp)
        
        return gmm_conv.log_prob(x0)

    def get_drift(self, t, x, clip_val=3.0):
        """
        Dérive Analytique: alpha*(t, x) = grad_x log(v_theta(x))
        [Update]: Ajout de 'clip_val' pour éviter l'explosion lors de la génération.
        """
        # Conversion en tenseur
        if not torch.is_tensor(x):
            x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        else:
            x_tensor = x.clone().detach().requires_grad_(True)
            
        # Forward pass
        log_v = self.model.log_prob(x_tensor)
        
        # Calcul du gradient
        # create_graph=False pour l'inférence, True si on voulait entraîner dessus
        grad = torch.autograd.grad(log_v.sum(), x_tensor, create_graph=False)[0]
        
        # --- FIX: Gradient Clipping ---
        # Empêche la dérive de devenir infinie dans les zones de faible probabilité
        grad = torch.clamp(grad, -clip_val, clip_val)
        
        return grad.detach().cpu().numpy()