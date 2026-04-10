import sys
import os
import torch
import argparse
from pathlib import Path
import re

from evaluations import compute_loss_three_state, compute_loss_two_state, evaluate, evaluate_persistent, predict_persistent, predict_states
from plots import plot_metrics, plot_multiple
from training import get_training_objects, load_config

RESULTS_DIR = 'results'
CONFIGS_DIR = 'configs'
PLOTS_DIR = 'plots'

def print_model_params(model):
    print("\nModel parameters:", file=sys.__stdout__)
    for name, param in model.named_parameters():
        # print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
        print(f"{name}: {param.data}", file=sys.__stdout__)

def parse_train_weights(arg, num_states, train_loader=None):
    """
    Returns list of weights for transitions
    """
    # Case 1: manual weights
    if "," in arg:
        return [float(w) for w in arg.split(",")]

    # Case 2: predefined schemes
    if arg.startswith("invtrans"):
        assert train_loader is not None, "Need train_loader to compute frequencies"

        match = re.search(r"invtrans(_noclamp)?([-+]?\d*\.?\d+)?", arg)
        if match:
            clamp = not (match.group(1) is not None)
            alpha = float(match.group(2)) if match.group(2) is not None else 0.5
        else:
            clamp = True
            alpha = 0.5

        freq = torch.zeros(num_states, num_states, dtype=torch.long)

        for batch in train_loader:
            x = batch["x_all"]
            y = batch["y"].long().view(-1)

            state = x[:, 0:1].long().view(-1)

            # Flatten pair indices into 1D
            indices = state * num_states + y

            # Count occurrences
            counts = torch.bincount(indices, minlength=num_states * num_states)

            # Reshape back to matrix and accumulate
            freq += counts.view(num_states, num_states)

        # print(freq, file=sys.__stdout__)

        freq = freq + 1 # add-one smoothing to avoid division by zero
        freq = freq.float()

        # mask for valid transitions
        mask = freq > 1

        # initialize weights
        weights = torch.zeros_like(freq)

        # inverse frequency only where valid
        weights[mask] = 1.0 / (freq[mask] ** alpha)

        # normalize ONLY over valid entries (mean normalization instead of sum normalization for stability/robustness)
        weights[mask] = weights[mask] / weights[mask].mean()

        if clamp:
            weights[mask] = torch.clamp(weights[mask], 0.1, 10.0) # clip extreme weights in between 0.1 and 10
        
        # print(weights, file=sys.__stdout__)

        # flatten into expected format
        if num_states == 3:
            # [UU, UB, UE, BB, BE]
            return [
                weights[0,0].item(),
                weights[0,1].item(),
                weights[0,2].item(),
                weights[1,1].item(),
                weights[1,2].item()
            ]
        elif num_states == 2:
            return [
                weights[0,0].item(),
                weights[0,1].item()
            ]
        
    # Case 3: equal weights
    if arg == "equal":
        return [1,1,1,1,1] if num_states == 3 else [1,1]

    raise ValueError(f"Unknown train_weights option: {arg}")

def parse_cost_matrix(arg, num_states):
    """
    Returns (num_states x num_states) tensor
    """

    # C[i,j] = cost of predicting j when true state is i

    # Case 1: manual matrix
    if "," in arg:
        values = [float(x) for x in arg.split(",")]
        C = torch.tensor(values).view(num_states, num_states)
        return C

    # Case 2: predefined
    if arg == "uniform":
        C = torch.ones(num_states, num_states)
        C.fill_diagonal_(0)
        return C

    if arg == "default" or arg == "wildfire":
        if num_states == 3:
            return torch.tensor([
                [0, 2, 1],  # true U: predicting U is 0 cost, B is worst (2), E is bad (1)
                [50, 0, 35], # true B: predicting B is 0 cost, U is worst (50), E is bad (35)
                [5, 2, 0]   # true E: predicting E is 0 cost, U is worst (5), B is bad (2)
            ]).float()
        elif num_states == 2:
            return torch.tensor([
                [0, 1], # true U: predicting U is 0 cost, B is bad (1)
                [5, 0]  # true B: predicting B is 0 cost, U is bad (5)
            ]).float()

    raise ValueError(f"Unknown cost_matrix option: {arg}")

def driver(model_type, backbone, agg, num_states, train_weights_arg, cost_matrix_arg, epochs, radius, imputed):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if model_type == 'persistent':
        print(f"Running with config: model_type={model_type}, num_states={num_states}, radius={radius}, imputed={imputed}", file=sys.__stdout__)
    else:
        print(f"Running with config: model_type={model_type}, backbone={backbone}, agg={agg}, num_states={num_states}, train_weights={train_weights_arg}, cost_matrix={cost_matrix_arg}, epochs={epochs}, radius={radius}, imputed={imputed}", file=sys.__stdout__)

    # prefix corresponds to model configuration
    prefix = f"{model_type}_{backbone}_{agg}_{num_states}"
    # suffix corresponds to loss weights
    suffix = (
        f"tw_{train_weights_arg or 'none'}_"
        f"cm_{cost_matrix_arg or 'none'}_"
        f"epochs{epochs}_radius{radius}_imputed{imputed}"
    )
    log_file = os.path.join(RESULTS_DIR, f'{prefix}_sca_{suffix}.txt')

    config_file = os.path.join(Path(__file__).resolve().parent.parent, CONFIGS_DIR, f'configs_sca_{num_states}{"_imp" if imputed else ""}.yaml')
    plots_dir = os.path.join(PLOTS_DIR, f'{prefix}_{suffix}')
    os.makedirs(plots_dir, exist_ok=True)
    
    with open(log_file, 'w') as f:
        sys.stdout = f

        cfg = load_config(config_file)
        cfg["data"]["hourly_agg"] = agg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, test_loader, model, optimizer = get_training_objects(
            data_cfg=cfg["data"],
            device=device,
            model_type=model_type,
            model_backbone=backbone,
            num_states=num_states,
            radius=radius,
        )
        print(len(train_loader), len(val_loader), len(test_loader), file=sys.__stdout__)

        train_weights = parse_train_weights(
            train_weights_arg,
            num_states,
            train_loader
        )

        cost_matrix = parse_cost_matrix(
            cost_matrix_arg,
            num_states
        )

        train_losses = []
        val_losses = []
        val_accs = []
        val_state_accs = []
        val_briers = []

        # Skip training loop if using persistent model (it has no learnable parameters)
        if model_type == 'persistent':
            epochs = 0

        for epoch in range(epochs):

            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                print(f"Processing batch {num_batches+1}...", file=sys.__stdout__)
                if num_states == 3:
                    loss = compute_loss_three_state(model, batch, device, weights=train_weights)
                elif num_states == 2:
                    loss = compute_loss_two_state(model, batch, device, weights=train_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                print(f"Epoch {epoch+1:02d} | Batch {num_batches:03d} | Loss: {loss.item():.4f}", file=sys.__stdout__)

            # print_model_params(model)

            avg_train_loss = total_loss / num_batches
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1:02d} | "
                f"Train Loss: {avg_train_loss:.4f}")

            val_loss, val_overall_acc, val_state_acc, val_prec, val_rec, val_iou, val_brier = evaluate(model, val_loader, device, num_states, train_weights, cost_matrix)
            val_losses.append(val_loss)
            val_accs.append(val_overall_acc)
            val_state_accs.append(val_state_acc)
            val_briers.append(val_brier)

            print(f"Epoch {epoch+1:02d} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_overall_acc:.4f} | "
                f"Val State Acc: {val_state_acc.tolist()} | "
                f"Val Prec: {val_prec.tolist()} | "
                f"Val Rec: {val_rec.tolist()} | "
                f"Val IoU: {val_iou.tolist()} | "
                f"Val Brier: {val_brier:.4f}")
            print(f"Epoch {epoch+1:02d} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_overall_acc:.4f} | "
                f"Val State Acc: {val_state_acc.tolist()} | "
                f"Val Prec: {val_prec.tolist()} | "
                f"Val Rec: {val_rec.tolist()} | "
                f"Val IoU: {val_iou.tolist()} | "
                f"Val Brier: {val_brier:.4f}", file=sys.__stdout__)
            
        # Persistent model has no training/validation metrics to plot
        if model_type != 'persistent':
            plot_metrics(
                train_losses, val_losses, val_briers, val_state_accs, 
                save_dir=plots_dir, save_file=f'metrics.png'
            )

        print("\nFinal Test Evaluation")

        if model_type != 'persistent':
            test_loss, test_overall_acc, test_state_acc, test_prec, test_rec, test_iou, test_brier = evaluate(model, test_loader, device, num_states, train_weights, cost_matrix)
        else:
            test_loss, test_overall_acc, test_state_acc, test_prec, test_rec, test_iou, test_brier = evaluate_persistent(test_loader, device, num_states)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_overall_acc:.4f} | "
              f"Test State Acc: {test_state_acc.tolist()} | "
              f"Test Prec: {test_prec.tolist()} | "
              f"Test Rec: {test_rec.tolist()} | "
              f"Test IoU: {test_iou.tolist()} | "
              f"Test Brier: {test_brier:.4f}")

        if model_type != 'persistent':
            preds, gts, states = predict_states(model, test_loader, device, num_states, cost_matrix)
            plot_multiple(states, gts, preds, n=10, save_dir=plots_dir, save_name='pred')
        else:
            preds, gts, states = predict_persistent(test_loader, device)
            plot_multiple(states, gts, preds, n=10, save_dir=plots_dir, save_name='pred_persistent')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Model configuration")

    parser.add_argument(
        "--model_type", "--type", "--mt",
        type=str,
        default="direct",
        help="Type of model (direct, hazard, persistent)"
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="logistic",
        help="Backbone model (logistic, mlp)"
    )

    parser.add_argument(
        "--agg",
        type=str,
        default="concat",
        help="Aggregation method (concat, mean)"
    )

    parser.add_argument(
        "--num_states", "--states", "--ns",
        type=int,
        default=3,
        help="Number of states (2, 3)"
    )

    parser.add_argument(
        "--train_weights", "--tw",
        type=str,
        default="equal",
        help="Training weights: 'invtrans' or comma-separated list"
    )

    parser.add_argument(
        "--cost_matrix", "--cm",
        type=str,
        default="uniform",
        help="Cost matrix: 'wildfire', 'uniform', or comma-separated list"
    )

    parser.add_argument(
        "--epochs", "--ep",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)."
    )

    parser.add_argument(
        "--radius", "--r",
        type=int,
        default=1,
        help="Radius for neighborhood computation (default: 1)."
    )

    parser.add_argument(
        "--imputed", "--impute", "--imp",
        action='store_true',
        default=False,
        help="Whether to use imputed values (default: False)."
    )

    args = parser.parse_args()

    driver(args.model_type, args.backbone, args.agg, args.num_states, args.train_weights, args.cost_matrix, args.epochs, args.radius, args.imputed)