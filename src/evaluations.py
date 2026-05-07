from collections import defaultdict

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

    total_same = 0
    total_pixels = 0
    matching_batches = 0
    total_batches = 0
    matching_batches2 = 0
    total_batches2 = 0

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

        same_as_current = (pred == state)
        total_same += same_as_current.sum().item()
        total_pixels += same_as_current.numel()

        if torch.equal(pred, state):
            matching_batches += 1
        total_batches += 1

        per_sample_match = (pred == state).view(pred.size(0), -1).all(dim=1)
        matching_batches2 += per_sample_match.sum().item()
        total_batches2 += pred.size(0)
    
    print(f"Percentage of predictions same as current state: {total_same / total_pixels:.4f}", total_same, total_pixels, file=sys.__stdout__)
    print(f"Percentage of batches where all predictions match current state: {matching_batches / total_batches:.4f}", matching_batches, total_batches, file=sys.__stdout__)
    print(f"Percentage of batches where all samples match current state: {matching_batches2 / total_batches2:.4f}", matching_batches2, total_batches2, file=sys.__stdout__)

    return predictions, ground_truth, current_states

@torch.no_grad()
def evaluate_persistent(loader, device, num_states):
    print('Evaluating persistent baseline...', file=sys.__stdout__)

    total_loss = 0.0   # no training, so default to 0
    total_pixels = 0
    correct = 0
    num_batches = 0

    state_correct = torch.zeros(num_states)
    state_total = torch.zeros(num_states)

    confusion = torch.zeros(num_states, num_states)
    brier_sum = 0.0

    for batch in loader:
        x = batch["x_all"].to(device)
        y = batch["y"].to(device).long()

        # Persistent prediction
        pred = x[:, 0:1].long()  # (B,1,H,W)

        # Build one-hot deterministic "probabilities"
        P = F.one_hot(
            pred.squeeze(1), num_classes=num_states
        ).permute(0, 3, 1, 2).float()       

        total_pixels += y.numel()
        correct += (pred == y).sum().item()
        num_batches += 1

        for z in range(num_states):
            mask = (y == z)
            state_total[z] += mask.sum().item()
            state_correct[z] += ((pred == y) & mask).sum().item()

        # Confusion matrix
        y_flat = y.view(-1).long()
        pred_flat = pred.view(-1).long()

        indices = num_states * y_flat + pred_flat
        cm = torch.bincount(
            indices, minlength=num_states**2
        ).reshape(num_states, num_states).cpu()

        confusion += cm

        # Brier score
        y_onehot = F.one_hot(
            y.squeeze(1), num_classes=num_states
        ).permute(0, 3, 1, 2).float()

        brier_sum += ((P - y_onehot) ** 2).sum().item()

    # Final metrics
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
        total_loss,
        correct / total_pixels,
        state_correct / state_total,
        precision,
        recall,
        iou,
        brier
    )

@torch.no_grad()
def predict_persistent(loader, device):
    predictions = []
    ground_truth = []
    current_states = []

    for batch in loader:
        # print(batch.keys(), file=sys.__stdout__)
        # print(batch['event_name'], file=sys.__stdout__)
        # print(batch['t'], file=sys.__stdout__)

        x = batch["x_all"].to(device)
        y = batch["y"].to(device)

        state = x[:, 0:1]
        pred = state.clone()  # persistence: next state = current state

        predictions.extend(pred.squeeze(1).cpu())
        ground_truth.extend(y.squeeze(1).cpu())
        current_states.extend(state.squeeze(1).cpu())

    return predictions, ground_truth, current_states


def group_by_event(loader):
    events = defaultdict(list)

    for batch in loader:
        B = len(batch["event_name"])

        for i in range(B):
            events[batch["event_name"][i]].append({
                "t": batch["t"][i],
                "x_all": batch["x_all"][i],
                "y": batch["y"][i],
            })

    # Sort each event by time
    for k in events:
        events[k] = sorted(events[k], key=lambda x: x["t"])

    return events

@torch.no_grad()
def simulate(model, loader, device, num_states, cost_matrix=None):

    print("Running autoregressive simulation...", file=sys.__stdout__)

    model.eval()

    events = group_by_event(loader)

    all_preds = []
    all_targets = []

    total_pixels = 0
    correct = 0

    state_correct = torch.zeros(num_states)
    state_total = torch.zeros(num_states)

    confusion = torch.zeros(num_states, num_states)
    brier_sum = 0.0

    for event_name, trajectory in events.items():

        # Initialize with FIRST ground truth state
        first = trajectory[0]
        x = first["x_all"].unsqueeze(0).to(device)

        # autoregressive state
        state = x[:, 0:1].clone()

        for step in trajectory:

            x = step["x_all"].unsqueeze(0).to(device)
            y = step["y"].to(device).long()

            # Replace state with autoregressive prediction
            x[:, 0:1] = state

            # Forward pass
            outputs = model(x)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            P = build_transition_probs(x, outputs, num_states)

            # Prediction
            if cost_matrix is not None:
                C = cost_matrix.to(P.device)
                expected_cost = (P.unsqueeze(2) * C.view(1, num_states, num_states, 1, 1)).sum(dim=1)
                pred = torch.argmin(expected_cost, dim=1, keepdim=True)
            else:
                pred = torch.argmax(P, dim=1, keepdim=True)

            # Move to CPU for metrics
            y_cpu = y.squeeze(0).cpu()
            pred_cpu = pred.squeeze(0).cpu()
            P_cpu = P.squeeze(0).cpu()

            all_preds.append(pred_cpu)
            all_targets.append(y_cpu)

            # Metrics
            total_pixels += y_cpu.numel()
            correct += (pred_cpu == y_cpu).sum().item()

            # Per-state accuracy
            for z in range(num_states):
                mask = (y_cpu == z)
                state_total[z] += mask.sum().item()
                state_correct[z] += ((pred_cpu == y_cpu) & mask).sum().item()

            # Confusion matrix
            y_flat = y_cpu.view(-1).long()
            pred_flat = pred_cpu.view(-1).long()

            indices = num_states * y_flat + pred_flat
            cm = torch.bincount(indices, minlength=num_states**2).reshape(num_states, num_states)

            confusion += cm

            # Brier score
            y_onehot = F.one_hot(
                y_cpu.squeeze(0), num_classes=num_states
            ).permute(2, 0, 1).float()

            brier_sum += ((P_cpu - y_onehot)**2).sum().item()

            # Autoregressive step
            state = pred.clone()

    # Final Metrics
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

    accuracy = correct / total_pixels
    state_acc = state_correct / (state_total + 1e-8)
    brier = brier_sum / total_pixels

    return (
        all_preds,
        all_targets,
        accuracy,
        state_acc,
        precision,
        recall,
        iou,
        brier
    )

@torch.no_grad()
def simulate_probabilistic(model, loader, device, num_states, cost_matrix=None):

    print("Running probabilistic autoregressive simulation...", file=sys.__stdout__)

    model.eval()
    events = group_by_event(loader)

    all_preds = []
    all_targets = []

    total_pixels = 0
    correct = 0

    state_correct = torch.zeros(num_states)
    state_total = torch.zeros(num_states)

    confusion = torch.zeros(num_states, num_states)
    brier_sum = 0.0

    for event_name, trajectory in events.items():

        print(f"Simulating prob for {event_name}", file=sys.__stdout__)

        # --- Initialize belief with first ground truth ---
        start_index = 0 # if len(trajectory) == 1 else 1
        first = trajectory[start_index]
        x = first["x_all"].unsqueeze(0).to(device)

        state0 = x[:, 0:1].long()
        pi = F.one_hot(
            state0.squeeze(1), num_classes=num_states
        ).permute(0, 3, 1, 2).float()  # (B, K, H, W)

        for step in trajectory[start_index:]:

            x = step["x_all"].unsqueeze(0).to(device)
            y = step["y"].to(device).long()

            # print(f'Time {step["t"]}: X/Y sums = {x[:,0:1].sum(), y.sum()}', file=sys.__stdout__)

            # --- Feed expected state into model (minimal-change version) ---
            expected_state = (
                pi * torch.arange(num_states, device=pi.device).view(1, -1, 1, 1)
            ).sum(dim=1, keepdim=True)

            x[:, 0:1] = expected_state

            # print(f'Time {step["t"]}: Exp sum = {x[:,0:1].sum()}', file=sys.__stdout__)

            # --- Forward pass ---
            outputs = model(x)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            # --- Build full transition tensor T ---
            B, _, H, W = pi.shape
            T = torch.zeros(B, num_states, num_states, H, W, device=device)

            if num_states == 2:
                (p_UB,) = outputs

                # From U (0)
                T[:, 0, 0] = 1 - p_UB
                T[:, 0, 1] = p_UB

                # From B (1)
                T[:, 1, 1] = 1.0

            elif num_states == 3:
                p_UU, p_UB, p_UE, p_BE = outputs

                # From U (0)
                T[:, 0, 0] = p_UU
                T[:, 0, 1] = p_UB
                T[:, 0, 2] = p_UE

                # From B (1)
                T[:, 1, 1] = 1 - p_BE
                T[:, 1, 2] = p_BE

                # From E (2)
                T[:, 2, 2] = 1.0

            else:
                raise ValueError(f"Unsupported num_states={num_states}")

            # --- Propagate belief ---
            # pi_next(z') = sum_z pi(z) * P(z' | z)
            pi = (pi.unsqueeze(2) * T).sum(dim=1)

            # --- Prediction for metrics ---
            if cost_matrix is not None:
                C = cost_matrix.to(pi.device)
                expected_cost = (pi.unsqueeze(2) * C.view(1, num_states, num_states, 1, 1)).sum(dim=1)
                pred = torch.argmin(expected_cost, dim=1, keepdim=True)
            else:
                pred = torch.argmax(pi, dim=1, keepdim=True)

            # print(f'Time {step["t"]}: Pred sum = {pred.sum()}', file=sys.__stdout__)

            # --- Move to CPU ---
            y_cpu = y.squeeze(0).cpu()
            pred_cpu = pred.squeeze(0).cpu()
            pi_cpu = pi.squeeze(0).cpu()

            all_preds.append(pred_cpu)
            all_targets.append(y_cpu)

            # --- Metrics ---
            total_pixels += y_cpu.numel()
            correct += (pred_cpu == y_cpu).sum().item()

            for z in range(num_states):
                mask = (y_cpu == z)
                state_total[z] += mask.sum().item()
                state_correct[z] += ((pred_cpu == y_cpu) & mask).sum().item()

            # Confusion matrix
            y_flat = y_cpu.view(-1).long()
            pred_flat = pred_cpu.view(-1).long()

            indices = num_states * y_flat + pred_flat
            cm = torch.bincount(indices, minlength=num_states**2).reshape(num_states, num_states)
            confusion += cm

            # --- Brier score (using pi instead of P) ---
            y_onehot = F.one_hot(
                y_cpu.squeeze(0), num_classes=num_states
            ).permute(2, 0, 1).float()

            brier_sum += ((pi_cpu - y_onehot)**2).sum().item()

    # --- Final metrics ---
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

    accuracy = correct / total_pixels
    state_acc = state_correct / (state_total + 1e-8)
    brier = brier_sum / total_pixels

    return (
        all_preds,
        all_targets,
        accuracy,
        state_acc,
        precision,
        recall,
        iou,
        brier
    )