import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# ============================================================
# Neighbor Burning Count
# ============================================================

def compute_neighbor_burning(state):
    """
    state: (B,1,H,W) with values {0,1,2}
    """
    print('Computing neighbor burning count...', file=sys.__stdout__)
    burning = (state == 1).float()

    kernel = torch.ones(1, 1, 3, 3, device=state.device)
    kernel[0, 0, 1, 1] = 0  # exclude center cell

    neighbor_count = F.conv2d(burning, kernel, padding=1)

    return neighbor_count


# ============================================================
# Logistic SCA Model
# ============================================================

class DirectLogisticSCA(nn.Module):

    def __init__(self, n_covariates, num_states=3):
        super().__init__()

        self.num_states = num_states

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
        N_B = compute_neighbor_burning(state) / 8.0

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