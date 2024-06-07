from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
import seaborn as sns

def exercise8():
    pylog.info("Ex 8")
    pylog.info("Implement exercise 8")
    log_path = './logs/exercise8/'
    os.makedirs(log_path, exist_ok=True)

    sigma_values = np.linspace(0, 30, num=10)
    w_stretch_values = np.linspace(0, 10, num=10)

    params_list = [
        SimulationParameters(
            controller="firing_rate",
            method = "noise",
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            noise_sigma=sigma,  # Varying noise level
            theta=0.1,
            n_desc_str = 2,
            w_stretch=w_stretch,  # Varying feedback strength
            video_record=False,
            video_name=f"exercise8_simulation_sigma_{sigma}_w_stretch_{w_stretch}",
            video_fps=30
        ) for sigma in sigma_values for w_stretch in w_stretch_values
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list, num_process=1)

    pylog.info("Simulations finished")

    # Initialize matrices to store metrics
    frequency_matrix = np.zeros((len(sigma_values), len(w_stretch_values)))
    wavefrequency_matrix = np.zeros((len(sigma_values), len(w_stretch_values)))
    forward_speed_matrix = np.zeros((len(sigma_values), len(w_stretch_values)))

    for idx, controller in enumerate(controllers):
        sigma = params_list[idx].noise_sigma
        w_stretch = params_list[idx].w_stretch
        i = np.where(sigma_values == sigma)[0][0]
        j = np.where(w_stretch_values == w_stretch)[0][0]

        metrics = controller.metrics
        frequency_matrix[i, j] = metrics['frequency']
        wavefrequency_matrix[i, j] = metrics['wavefrequency']
        forward_speed_matrix[i, j] = metrics['fspeed_PCA']

    # Plotting heatmaps for the metrics
    plt.figure('Heatmap: Frequency vs sigma and w_stretch')
    sns.heatmap(frequency_matrix, xticklabels=w_stretch_values, yticklabels=sigma_values, annot=True, cmap='viridis')
    plt.xlabel('w_stretch')
    plt.ylabel('sigma')
    plt.title('Heatmap: Frequency vs sigma and w_stretch')
    plt.savefig(f'{log_path}/heatmap_frequency_vs_sigma_w_stretch.png')
    plt.close()

    plt.figure('Heatmap: Wave Frequency vs sigma and w_stretch')
    sns.heatmap(wavefrequency_matrix, xticklabels=w_stretch_values, yticklabels=sigma_values, annot=True, cmap='viridis')
    plt.xlabel('w_stretch')
    plt.ylabel('sigma')
    plt.title('Heatmap: Wave Frequency vs sigma and w_stretch')
    plt.savefig(f'{log_path}/heatmap_wavefrequency_vs_sigma_w_stretch.png')
    plt.close()

    plt.figure('Heatmap: Forward Speed vs sigma and w_stretch')
    sns.heatmap(forward_speed_matrix, xticklabels=w_stretch_values, yticklabels=sigma_values, annot=True, cmap='viridis')
    plt.xlabel('w_stretch')
    plt.ylabel('sigma')
    plt.title('Heatmap: Forward Speed vs sigma and w_stretch')
    plt.savefig(f'{log_path}/heatmap_forward_speed_vs_sigma_w_stretch.png')
    plt.close()

    pylog.info("Plots saved successfully")

if __name__ == '__main__':
    exercise8()