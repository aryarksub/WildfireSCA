import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# ============================================================
# Neighbor Burning Count
# ============================================================

def compute_neighbor_burning(state, radius=1):
    """
    state: (B,1,H,W) with values {0,1,2}
    """
    print('Computing neighbor burning count...', file=sys.__stdout__)
    burning = (state == 1).float()

    kernel = torch.ones(1, 1, 2*radius+1, 2*radius+1, device=state.device)
    kernel[0, 0, 1, 1] = 0  # exclude center cell

    neighbor_count = F.conv2d(burning, kernel, padding=radius)

    return neighbor_count


# ============================================================
# Logistic SCA Model
# ============================================================

class DirectLogisticSCA(nn.Module):

    def __init__(self, n_covariates, num_states=3, radius=1):
        super().__init__()

        self.n_covariates = n_covariates
        self.num_states = num_states
        self.radius = radius

        # --- Shared ignition (U -> *) ---
        self.alpha0 = nn.Parameter(torch.tensor([-1.0]))
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(n_covariates))

        if num_states == 3:
            # --- Burnout (B -> E) ---
            self.gamma0 = nn.Parameter(torch.zeros(1))
            self.gamma = nn.Parameter(torch.zeros(n_covariates))

            # --- Direct extinguish (U -> E) ---
            self.delta0 = nn.Parameter(torch.tensor([0.0]))
            self.delta1 = nn.Parameter(torch.zeros(1))
            self.eta = nn.Parameter(torch.zeros(n_covariates))

    def forward(self, x):

        state = x[:, 0:1]
        covariates = x[:, 1:]

        # Neighbor feature
        num_neighbors = (2 * self.radius + 1) ** 2 - 1
        N_B = compute_neighbor_burning(state, radius=self.radius) / num_neighbors

        # Shared linear term helper
        def linear_term(weights):
            return (covariates * weights.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

        # =========================================================
        # TWO-STATE CASE
        # =========================================================
        if self.num_states == 2:

            ignition_logit = (
                self.alpha0
                + self.alpha1 * N_B
                + linear_term(self.beta)
            )

            p_UB = torch.sigmoid(ignition_logit)

            return p_UB

        # =========================================================
        # THREE-STATE CASE
        # =========================================================
        elif self.num_states == 3:

            # U transitions
            z_UB = (
                self.alpha0
                + self.alpha1 * N_B
                + linear_term(self.beta)
            )

            z_UE = (
                self.delta0
                + self.delta1 * N_B
                + linear_term(self.eta)
            )

            z_UU = torch.zeros_like(z_UB)

            U_logits = torch.cat([z_UU, z_UB, z_UE], dim=1)
            U_probs = torch.softmax(U_logits, dim=1)

            p_UU = U_probs[:, 0:1]
            p_UB = U_probs[:, 1:2]
            p_UE = U_probs[:, 2:3]

            # B -> E
            burnout_logit = (
                self.gamma0
                + linear_term(self.gamma)
            )

            p_BE = torch.sigmoid(burnout_logit)

            return p_UU, p_UB, p_UE, p_BE

        else:
            raise ValueError(f"Unsupported num_states={self.num_states}")

class HazardLogisticSCA(nn.Module):
    """
    Hazard-based SCA model with hourly hazards aggregated over 12 hours.

    Matches theoretical formulation:
        p_ign = 1 - ∏ (1 - λ_k)
        p_burn = 1 - ∏ (1 - μ_k)

    Supports 3-state model:
        U -> {U, B, E}
        B -> {B, E}
        E -> E
    """

    def __init__(self, n_covariates, num_static, num_states=3, T=12, radius=1):
        super().__init__()

        self.num_states = num_states
        self.T = T  # number of hourly steps
        self.num_static = num_static # number of static variables/layers
        self.radius = radius
        
        # infer hourly dimension
        self.n_hourly_total = n_covariates - num_static
        assert self.n_hourly_total % T == 0, f"Hourly covariates must be divisible by T={T}"
        self.C_per_hour = self.n_hourly_total // T
        self.C = self.C_per_hour + num_static

        # -----------------------------
        # Ignition hazard (U -> B)
        # -----------------------------
        self.alpha0 = nn.Parameter(torch.tensor([-2.0]))  # low base hazard
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(self.C))

        if num_states == 3:
            # -----------------------------
            # Burnout hazard (B -> E)
            # -----------------------------
            self.gamma0 = nn.Parameter(torch.tensor([-1.0]))
            self.gamma = nn.Parameter(torch.zeros(self.C))

            # -----------------------------
            # Direct extinguish hazard (U -> E)
            # -----------------------------
            self.delta0 = nn.Parameter(torch.tensor([-3.0]))  # very rare
            self.delta1 = nn.Parameter(torch.zeros(1))
            self.eta = nn.Parameter(torch.zeros(self.C))

    # ============================================================
    # Helper: split covariates
    # ============================================================

    def split_covariates(self, covariates):
        """
        covariates: (B, C_total, H, W)

        returns:
            cov_seq: (B, T, C_per_hour + num_static, H, W)
        """

        B, C, H, W = covariates.shape

        # -----------------------------
        # Split hourly vs static
        # -----------------------------
        hourly = covariates[:, :self.n_hourly_total]   # (B, C_hourly_total, H, W)
        static = covariates[:, self.n_hourly_total:]   # (B, num_static, H, W)

        # -----------------------------
        # Reshape hourly → (B, T, C_per_hour, H, W)
        # -----------------------------
        hourly = hourly.view(B, self.T, self.C_per_hour, H, W)

        # -----------------------------
        # Broadcast static → (B, T, num_static, H, W)
        # -----------------------------
        static = static.unsqueeze(1)                   # (B,1,num_static,H,W)
        static = static.expand(-1, self.T, -1, -1, -1)

        # -----------------------------
        # Concatenate
        # -----------------------------
        cov_seq = torch.cat([hourly, static], dim=2)

        return cov_seq

    # ============================================================
    # Helper: stable hazard aggregation
    # ============================================================

    def aggregate_hazard(self, hazard):
        """
        hazard: (B, T, 1, H, W)

        returns:
            p = 1 - prod(1 - hazard)
        computed in log-space for stability
        """
        log_survival = torch.sum(torch.log(1 - hazard + 1e-8), dim=1)
        return 1 - torch.exp(log_survival)

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x):

        state = x[:, 0:1]          # (B,1,H,W)
        covariates = x[:, 1:]      # (B,C,H,W)

        # Neighborhood feature (constant over 12h)
        num_neighbors = (2 * self.radius + 1) ** 2 - 1
        N_B = compute_neighbor_burning(state, radius=self.radius) / num_neighbors

        # Split into hourly covariates
        cov_seq = self.split_covariates(covariates)  # (B,T,C,H,W)

        B, T, C, H, W = cov_seq.shape

        # =========================================================
        # Ignition hazard λ_k
        # =========================================================

        # (B,T,1,H,W)
        linear_ign = (cov_seq * self.beta.view(1,1,-1,1,1)).sum(dim=2, keepdim=True)

        z_ign = (
            self.alpha0
            + self.alpha1 * N_B.unsqueeze(1)
            + linear_ign
        )

        lambda_k = torch.sigmoid(z_ign)

        p_UB = self.aggregate_hazard(lambda_k)

        # =========================================================
        # TWO-STATE CASE
        # =========================================================
        if self.num_states == 2:
            return p_UB

        # =========================================================
        # Burnout hazard μ_k
        # =========================================================

        linear_burn = (cov_seq * self.gamma.view(1,1,-1,1,1)).sum(dim=2, keepdim=True)

        z_burn = self.gamma0 + linear_burn
        mu_k = torch.sigmoid(z_burn)

        p_BE = self.aggregate_hazard(mu_k)

        # =========================================================
        # Direct extinguish hazard η_k
        # =========================================================

        linear_ext = (cov_seq * self.eta.view(1,1,-1,1,1)).sum(dim=2, keepdim=True)

        z_ext = (
            self.delta0
            + self.delta1 * N_B.unsqueeze(1)
            + linear_ext
        )

        eta_k = torch.sigmoid(z_ext)

        p_UE = self.aggregate_hazard(eta_k)

        # =========================================================
        # Normalize U transitions
        # =========================================================

        total = p_UB + p_UE

        p_UU = torch.clamp(1 - total, min=1e-6)

        # Optional normalization (keeps strict simplex)
        Z = p_UU + p_UB + p_UE
        p_UU = p_UU / Z
        p_UB = p_UB / Z
        p_UE = p_UE / Z

        return p_UU, p_UB, p_UE, p_BE


# ============================================================
# MLP SCA Model
# ============================================================

class MLPSCA(nn.Module):
    def __init__(self, n_covariates, hidden_dim=32, num_states=3, radius=1):
        super().__init__()
        self.num_states = num_states
        self.radius = radius

        in_dim = n_covariates + 1  # covariates + neighbor burning fraction

        # Shared ignition backbone
        self.ignite_backbone = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
        )

        if self.num_states == 3:
            # U -> {U, B, E}
            self.ignite_head = nn.Conv2d(hidden_dim, 3, 1) # logits for UU, UB, UE
            # Initial bias towards remaining unburned
            self.ignite_head.bias.data = torch.tensor([5.0, -3.0, -4.0])

            # B -> E
            self.burnout_mlp = nn.Sequential(
                nn.Conv2d(n_covariates, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, 1),
            )
            # Initial bias towards extinguishing (BE)
            self.burnout_mlp[-1].bias.data.fill_(-0.5)

        else:  # 2-state
            # U -> B
            self.ignite_head = nn.Conv2d(hidden_dim, 1, 1)
            # Initial bias towards remaining unburned
            self.ignite_head.bias.data.fill_(-4.0)

    def forward(self, x):
        """
        x: (B, C_total, H, W)

        Channel 0 = burned_state at time t
        Channels 1: = covariates at time t
        """
        state = x[:, 0:1]
        covariates = x[:, 1:]

        # Neighbor burning fraction
        num_neighbors = (2 * self.radius + 1) ** 2 - 1
        N_B = compute_neighbor_burning(state, radius=self.radius) / num_neighbors

        features = torch.cat([covariates, N_B], dim=1)

        # Shared backbone
        h = self.ignite_backbone(features)

        logits = self.ignite_head(h)

        if self.num_states == 2:
            # U -> B
            p_ignite = torch.sigmoid(logits)
            return p_ignite

        else:
            # U -> {U, B, E}
            U_probs = torch.softmax(logits, dim=1)
            p_UU = U_probs[:, 0:1]
            p_UB = U_probs[:, 1:2]
            p_UE = U_probs[:, 2:3]

            # B -> E
            burnout_logit = self.burnout_mlp(covariates)
            p_BE = torch.sigmoid(burnout_logit)

            return p_UU, p_UB, p_UE, p_BE
        
class HazardMLPSCA(nn.Module):
    """
    MLP-based hazard SCA model.

    - Applies MLP per hour
    - Aggregates hazards over T=12
    """

    def __init__(self, n_covariates, num_static, hidden_dim=32, num_states=3, T=12, radius=1):
        super().__init__()

        self.num_states = num_states
        self.T = T
        self.num_static = num_static
        self.radius = radius

        # -----------------------------
        # Infer per-hour structure
        # -----------------------------
        self.n_hourly_total = n_covariates - num_static
        assert self.n_hourly_total % T == 0, f"Hourly covariates must be divisible by T={T}"

        self.C_per_hour = self.n_hourly_total // T
        self.C = self.C_per_hour + num_static

        # +1 for neighbor feature
        in_dim_ignite = self.C + 1
        in_dim_burn = self.C

        # =========================================================
        # Ignition hazard MLP (shared across time)
        # =========================================================
        self.ignite_mlp = nn.Sequential(
            nn.Conv2d(in_dim_ignite, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

        # Bias → low ignition initially
        self.ignite_mlp[-1].bias.data.fill_(-2.0)

        if num_states == 3:

            # =====================================================
            # Burnout hazard MLP
            # =====================================================
            self.burnout_mlp = nn.Sequential(
                nn.Conv2d(in_dim_burn, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, 1),
            )
            self.burnout_mlp[-1].bias.data.fill_(-1.0)

            # =====================================================
            # Direct extinguish (U -> E)
            # =====================================================
            self.extinguish_mlp = nn.Sequential(
                nn.Conv2d(in_dim_ignite, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, 1),
            )
            self.extinguish_mlp[-1].bias.data.fill_(-3.0)

    # =========================================================
    # Split covariates
    # =========================================================

    def split_covariates(self, covariates):
        B, C, H, W = covariates.shape

        hourly = covariates[:, :self.n_hourly_total]
        static = covariates[:, self.n_hourly_total:]

        hourly = hourly.view(B, self.T, self.C_per_hour, H, W)

        static = static.unsqueeze(1).expand(-1, self.T, -1, -1, -1)

        cov_seq = torch.cat([hourly, static], dim=2)

        return cov_seq  # (B,T,C,H,W)

    # =========================================================
    # Hazard aggregation
    # =========================================================

    def aggregate_hazard(self, hazard):
        log_survival = torch.sum(torch.log(1 - hazard + 1e-8), dim=1)
        return 1 - torch.exp(log_survival)

    # =========================================================
    # Forward
    # =========================================================

    def forward(self, x):

        state = x[:, 0:1]
        covariates = x[:, 1:]

        # Neighborhood (constant across time)
        num_neighbors = (2 * self.radius + 1) ** 2 - 1
        N_B = compute_neighbor_burning(state, radius=self.radius) / num_neighbors

        cov_seq = self.split_covariates(covariates)  # (B,T,C,H,W)

        B, T, C, H, W = cov_seq.shape

        # =====================================================
        # Prepare inputs per hour
        # =====================================================

        N_B_seq = N_B.unsqueeze(1).expand(-1, T, -1, -1, -1)

        ignite_input = torch.cat([cov_seq, N_B_seq], dim=2)  # (B,T,C+1,H,W)

        # Merge batch + time for conv
        def merge_time(x):
            return x.view(B*T, x.shape[2], H, W)

        def unmerge_time(x):
            return x.view(B, T, 1, H, W)

        # =====================================================
        # Ignition hazard λ_k
        # =====================================================

        ignite_flat = merge_time(ignite_input)
        lambda_k = torch.sigmoid(self.ignite_mlp(ignite_flat))
        lambda_k = unmerge_time(lambda_k)

        p_UB = self.aggregate_hazard(lambda_k)

        if self.num_states == 2:
            return p_UB

        # =====================================================
        # Burnout hazard μ_k
        # =====================================================

        burn_flat = merge_time(cov_seq)
        mu_k = torch.sigmoid(self.burnout_mlp(burn_flat))
        mu_k = unmerge_time(mu_k)

        p_BE = self.aggregate_hazard(mu_k)

        # =====================================================
        # Direct extinguish η_k
        # =====================================================

        ext_flat = merge_time(ignite_input)
        eta_k = torch.sigmoid(self.extinguish_mlp(ext_flat))
        eta_k = unmerge_time(eta_k)

        p_UE = self.aggregate_hazard(eta_k)

        # =====================================================
        # Normalize U transitions
        # =====================================================

        total = p_UB + p_UE
        p_UU = torch.clamp(1 - total, min=1e-6)

        Z = p_UU + p_UB + p_UE
        p_UU = p_UU / Z
        p_UB = p_UB / Z
        p_UE = p_UE / Z

        return p_UU, p_UB, p_UE, p_BE