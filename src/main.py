import sys
import os
import torch
import argparse
from pathlib import Path

from evaluations import compute_loss_three_state, compute_loss_two_state, evaluate, predict_states
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

def driver(model_type, backbone, agg, num_states, weights, epochs):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Running with config: model_type={model_type}, backbone={backbone}, agg={agg}, num_states={num_states}, weights={weights}, epochs={epochs}", file=sys.__stdout__)

    # prefix corresponds to model configuration
    prefix = f"{model_type}_{backbone}_{agg}_{num_states}"
    # suffix corresponds to loss weights
    suffix = '-'.join(map(str, weights)) if weights is not None else 'eq_weights'
    log_file = os.path.join(RESULTS_DIR, f'{prefix}_sca_{suffix}.txt')

    config_file = os.path.join(Path(__file__).resolve().parent.parent, CONFIGS_DIR, f'configs_sca_{num_states}.yaml')
    plots_dir = os.path.join(PLOTS_DIR, f'{prefix}_{suffix}')
    
    with open(log_file, 'w') as f:
        sys.stdout = f

        cfg = load_config(config_file)
        cfg["data"]["hourly_agg"] = agg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, test_loader, model, optimizer = get_training_objects(
            data_cfg=cfg["data"],
            device=device,
            model_backbone=backbone,
        )

        train_losses = []
        val_losses = []
        val_accs = []
        val_state_accs = []
        val_briers = []

        for epoch in range(epochs):

            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                print(f"Processing batch {num_batches+1}...", file=sys.__stdout__)
                if num_states == 3:
                    loss = compute_loss_three_state(model, batch, device, weights=weights)
                elif num_states == 2:
                    loss = compute_loss_two_state(model, batch, device, weights=weights)

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

            val_loss, val_overall_acc, val_state_acc, val_prec, val_rec, val_iou, val_brier = evaluate(model, val_loader, device, num_states, weights=weights)
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
            
        plot_metrics(
            train_losses, val_losses, val_briers, val_state_accs, 
            save_dir=plots_dir, save_file=f'metrics.png'
        )

        print("\nFinal Test Evaluation")

        test_loss, test_overall_acc, test_state_acc, test_prec, test_rec, test_iou, test_brier = evaluate(model, test_loader, device, num_states, weights=weights)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_overall_acc:.4f} | "
              f"Test State Acc: {test_state_acc.tolist()} | "
              f"Test Prec: {test_prec.tolist()} | "
              f"Test Rec: {test_rec.tolist()} | "
              f"Test IoU: {test_iou.tolist()} | "
              f"Test Brier: {test_brier:.4f}")

        preds, gts, states = predict_states(model, test_loader, device, num_states)
        plot_multiple(states, gts, preds, n=10, save_dir=plots_dir, save_name='pred')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Model configuration")

    parser.add_argument(
        "--model_type",
        type=str,
        default="direct",
        help="Type of model (direct, hazard)"
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
        "--num_states",
        type=int,
        default=3,
        help="Number of states (2, 3)"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma-separated weights (e.g. '1,1,1,1,1'). For no weights, pass empty string."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)."
    )

    args = parser.parse_args()

    # --- parse weights into list of floats ---
    if args.weights:
        args.weights = [float(w) for w in args.weights.split(",")]
    else:
        args.weights = None

    driver(args.model_type, args.backbone, args.agg, args.num_states, args.weights, args.epochs)