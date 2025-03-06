import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy


def visualize_results(results, save_dir='plots/'):
    """
    Visualize model predictions and intermediate results
    
    Args:
        results: Results dictionary from evaluate_model
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    result = results[0]  # First batch
    
    # Get first sample in batch
    inputs = result['inputs'][0]
    solutions = [sol[0] for sol in result['solutions']]
    target = result['targets'][0]
    rmse_values = result['rmse_values']
    
    # Create colormap
    my_cmap = copy.copy(cm.get_cmap('jet_r'))
    
    # Plot input channels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    channel_names = ['Reflectance', 'Transmittance', 'Distance', 'Radiation Pattern']
    
    for i in range(4):
        im = axes[i].imshow(inputs[i], cmap='viridis')
        axes[i].set_title(channel_names[i])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'input_channels.png'))
    plt.close(fig)
    
    # Plot refinement iterations
    fig, axes = plt.subplots(1, len(solutions), figsize=(16, 4))
    
    for i, solution in enumerate(solutions):
        im = axes[i].imshow(solution[0], cmap=my_cmap)
        if i == 0:
            axes[i].set_title(f'Initial (RMSE: {rmse_values[i]:.4f})')
        else:
            axes[i].set_title(f'Iteration {i} (RMSE: {rmse_values[i]:.4f})')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'refinement_iterations.png'))
    plt.close(fig)
    
    # Plot ground truth vs. final prediction
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    im1 = axes[0].imshow(solutions[-1][0], cmap=my_cmap)
    axes[0].set_title(f'Final Prediction (Iter {len(solutions)-1})')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(target[0], cmap=my_cmap)
    axes[1].set_title('Ground Truth')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference
    diff = target[0] - solutions[-1][0]
    im3 = axes[2].imshow(diff, cmap='coolwarm')
    axes[2].set_title(f'Difference (RMSE: {rmse_values[-1]:.4f})')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_comparison.png'))
    plt.close(fig)
    
    # Plot RMSE improvement
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(rmse_values)), rmse_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('RMSE Improvement Over Iterations')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'rmse_improvement.png'))
    plt.close()

