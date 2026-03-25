import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# ============================================================
# SCA Log-Likelihood
# ============================================================

def compute_loss_three_state(model, batch, device, weights=None):
    print('Computing loss...', file=sys.__stdout__)

    x = batch["x_all"].to(device)
    y = batch["y"].to(device)          # future burned_state
    # mask = batch["mask"].to(device).float()

    p_UU, p_UB, p_UE, p_BE = model(x)

    state = x[:, 0:1]   # current state

    U = (state == 0)
    BURN = (state == 1)
    E = (state == 2)

    loss = torch.zeros_like(state, dtype=torch.float)

    if weights is not None and len(weights) == 5:
        uu_weight, ub_weight, ue_weight, bb_weight, be_weight = weights
    else:
        print('Given weights are None or not of length 5. Using default weights of 1.0 for all transitions.', file=sys.__stdout__)
        uu_weight = ub_weight = ue_weight = bb_weight = be_weight = 1.0

    # U transitions
    loss[U] = torch.where(
        y[U] == 0,
        uu_weight * -torch.log(p_UU[U] + 1e-8),
        torch.where(
            y[U] == 1,
            ub_weight * -torch.log(p_UB[U] + 1e-8),
            ue_weight * -torch.log(p_UE[U] + 1e-8)
        )
    )

    # B transitions
    loss[BURN] = torch.where(
        y[BURN] == 2,
        be_weight * -torch.log(p_BE[BURN] + 1e-8),
        bb_weight * -torch.log(1 - p_BE[BURN] + 1e-8)
    )

    # E stays E
    loss[E] = 0.0

    return loss.mean()

def compute_loss_two_state(model, batch, device, weights=None):

    x = batch["x_all"].to(device)
    y = batch["y"].to(device)          # future burned_state
    # mask = batch["mask"].to(device).float()

    p_UB = model(x)

    state = x[:, 0:1]   # current state

    U = (state == 0)
    B = (state == 1)

    loss = torch.zeros_like(state, dtype=torch.float)

    if weights is not None and len(weights) == 2:
        uu_weight, ub_weight = weights
    else:
        print('Given weights are None or not of length 2. Using default weights of 1.0 for all transitions.', file=sys.__stdout__)
        uu_weight = ub_weight = 1.0

    # U transitions
    loss[U] = torch.where(
        y[U] == 1,
        ub_weight * -torch.log(p_UB[U] + 1e-8),
        uu_weight * -torch.log(1 - p_UB[U] + 1e-8)
    )

    # B stays B
    loss[B] = 0.0

    return loss.mean()



# ============================================================
# Evaluation
# ============================================================

def build_transition_probs(x, model_outputs, num_states):
    state = x[:, 0:1]

    if num_states == 2:
        (p_UB,) = model_outputs

        P = torch.zeros(x.size(0), 2, *x.shape[2:], device=x.device)

        # U
        mask = (state == 0)
        P[:, 0:1][mask] = 1 - p_UB[mask]
        P[:, 1:2][mask] = p_UB[mask]

        # B
        P[:, 1:2][state == 1] = 1.0

    elif num_states == 3:
        p_UU, p_UB, p_UE, p_BE = model_outputs

        P = torch.zeros(x.size(0), 3, *x.shape[2:], device=x.device)

        # U
        mask = (state == 0)
        P[:, 0:1][mask] = p_UU[mask]
        P[:, 1:2][mask] = p_UB[mask]
        P[:, 2:3][mask] = p_UE[mask]

        # B
        mask = (state == 1)
        P[:, 1:2][mask] = 1 - p_BE[mask]
        P[:, 2:3][mask] = p_BE[mask]

        # E
        P[:, 2:3][state == 2] = 1.0

    else:
        raise ValueError(f"Unsupported num_states={num_states}")

    return P


@torch.no_grad()
def evaluate(model, loader, device, num_states, train_weights=None, cost_matrix=None):
    print('Evaluating...', file=sys.__stdout__)

    model.eval()

    total_loss = 0.0
    total_pixels = 0
    correct = 0
    num_batches = 0

    # Per-state counters
    state_correct = torch.zeros(num_states)
    state_total = torch.zeros(num_states)

    # Confusion matrix
    confusion = torch.zeros(num_states, num_states)

    # Brier score accumulator
    brier_sum = 0.0

    loss_fn = compute_loss_three_state if num_states == 3 else compute_loss_two_state

    for batch in loader:
        x = batch["x_all"].to(device)
        y = batch["y"].to(device).long()
        # mask = batch["mask"].to(device).float()

        # --- model forward ---
        outputs = model(x)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # --- build probabilities ---
        P = build_transition_probs(x, outputs, num_states)

        # Choose most likely next state
        if cost_matrix is not None:
            C = cost_matrix.to(P.device)

            P_exp = P.unsqueeze(2)                 # (B,K,1,H,W)
            C_exp = C.view(1, num_states, num_states, 1, 1)

            expected_cost = (P_exp * C_exp).sum(dim=1)  # (B,K,H,W)

            pred = torch.argmin(expected_cost, dim=1, keepdim=True)
        else:
            pred = torch.argmax(P, dim=1, keepdim=True)

        loss = loss_fn(model, batch, device, weights=train_weights)

        total_loss += loss.item()
        total_pixels += y.numel()

        correct += (pred == y).sum().item()
        num_batches += 1

        # ----- Per-state accuracy -----
        for z in range(num_states):
            mask = (y == z)
            state_total[z] += mask.sum().item()
            state_correct[z] += ((pred == y) & mask).sum().item()

        # ----------------------------------
        # Confusion matrix (vectorized)
        # ----------------------------------

        y_flat = y.view(-1).to(torch.long)
        pred_flat = pred.view(-1).to(torch.long)

        indices = num_states * y_flat + pred_flat
        cm = torch.bincount(indices, minlength=num_states**2).reshape(num_states,num_states).cpu()

        confusion += cm

        # ----------------------------------
        # Brier score
        # ----------------------------------

        y_onehot = F.one_hot(y.squeeze(1), num_classes=num_states).permute(0,3,1,2).float()

        brier_sum += ((P - y_onehot)**2).sum().item()

    # =========================================================
    # Final Metrics
    # =========================================================

    precision = torch.zeros(num_states)
    recall = torch.zeros(num_states)
    iou = torch.zeros(num_states)

    for k in range(num_states):

        TP = confusion[k, k]
        FP = confusion[:, k].sum() - TP
        FN = confusion[k, :].sum() - TP

        precision[k] = TP / (TP + FP + 1e-8)
        recall[k] = TP / (TP + FN + 1e-8)
        iou[k] = TP / (TP + FP + FN + 1e-8)

    brier = brier_sum / total_pixels

    return (
        total_loss / num_batches,
        correct / total_pixels,
        state_correct / state_total,
        precision,
        recall,
        iou,
        brier
    )

@torch.no_grad()
def predict_states(model, loader, device, num_states, cost_matrix=None):

    model.eval()

    predictions = []
    ground_truth = []
    current_states = []

    for batch in loader:

        x = batch["x_all"].to(device)
        y = batch["y"].to(device)

        outputs = model(x)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        P = build_transition_probs(x, outputs, num_states)

        if cost_matrix is not None:
            C = cost_matrix.to(P.device)

            P_exp = P.unsqueeze(2)                 # (B,K,1,H,W)
            C_exp = C.view(1, num_states, num_states, 1, 1)

            expected_cost = (P_exp * C_exp).sum(dim=1)  # (B,K,H,W)

            pred = torch.argmin(expected_cost, dim=1, keepdim=True)
        else:
            pred = torch.argmax(P, dim=1, keepdim=True)

        state = x[:, 0:1]

        predictions.extend(pred.squeeze(1).cpu())
        ground_truth.extend(y.squeeze(1).cpu())
        current_states.extend(state.squeeze(1).cpu())

    return predictions, ground_truth, current_states