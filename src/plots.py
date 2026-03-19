from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_multiple(states, gts, preds, n=5, save_dir='plots', save_name='pred'):

    cmap = ListedColormap(["green","red","black"])

    n = min(n, len(states))

    for i in range(n):

        state_squeeze = np.squeeze(states[i])
        gt_squeeze = np.squeeze(gts[i])
        pred_squeeze = np.squeeze(preds[i])

        fig, axes = plt.subplots(1,3,figsize=(15,5))

        axes[0].imshow(state_squeeze, cmap=cmap, vmin=0, vmax=2)
        axes[0].set_title("State t")

        axes[1].imshow(gt_squeeze, cmap=cmap, vmin=0, vmax=2)
        axes[1].set_title("Ground Truth")

        axes[2].imshow(pred_squeeze, cmap=cmap, vmin=0, vmax=2)
        axes[2].set_title("Prediction")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{save_name}_{i}.png'))

def plot_metrics(train_losses, val_losses, val_briers, val_state_accs, save_dir='plots', save_file='metrics.png'):
    epochs = range(1, len(train_losses) + 1)

    # split state accuracies
    acc0 = [a[0] for a in val_state_accs]
    acc1 = [a[1] for a in val_state_accs]
    acc2 = [a[2] if len(a) > 2 else 0.0 for a in val_state_accs]

    fig, axes = plt.subplots(2, 3, figsize=(15, 4))

    # Train Loss
    axes[0,0].plot(epochs, train_losses)
    axes[0,0].set_title("Train Loss")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")

    # Validation Loss
    axes[0,1].plot(epochs, val_losses)
    axes[0,1].set_title("Validation Loss")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("Loss")

    # Validation Brier Score
    axes[0,2].plot(epochs, val_briers)
    axes[0,2].set_title("Validation Brier Score")
    axes[0,2].set_xlabel("Epoch")
    axes[0,2].set_ylabel("Brier Score")

    # Validation accuracies per state
    axes[1,0].plot(epochs, acc0)
    axes[1,0].set_title("Accuracy (State 0)")
    axes[1,0].set_xlabel("Epoch")
    axes[1,0].set_ylabel("Accuracy")

    axes[1,1].plot(epochs, acc1)
    axes[1,1].set_title("Accuracy (State 1)")
    axes[1,1].set_xlabel("Epoch")
    axes[1,1].set_ylabel("Accuracy")

    axes[1,2].plot(epochs, acc2)
    axes[1,2].set_title("Accuracy (State 2)")
    axes[1,2].set_xlabel("Epoch")
    axes[1,2].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_file))
