from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories

def exercise7():

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    log_path = './logs/exercise7/'
    os.makedirs(log_path, exist_ok=True)

    I_values = np.linspace(0, 30, num=10)
    w_stretch_values = np.linspace(0, 15, num=10)

    params_list = [
        SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            I=I,  # Varying I
            w_stretch=w_stretch,  # Varying feedback weight
            video_record=False,  # Disable video recording
            video_name=f"exercise7_simulation_I_{I}_gss_{w_stretch}",  # Name of the video file
            video_fps=30  # Frames per second
        ) for I in I_values for w_stretch in w_stretch_values
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list, num_process=10)

    pylog.info("Simulations finished")

    # Initialize arrays to store metrics
    frequency_values = np.zeros((len(I_values), len(w_stretch_values)))
    wavefrequency_values = np.zeros((len(I_values), len(w_stretch_values)))
    forward_speed_values = np.zeros((len(I_values), len(w_stretch_values)))

    pylog.info("Plotting the results")

    for idx, controller in enumerate(controllers):
        I = params_list[idx].I  # Get the I value for the current simulation
        w_stretch = params_list[idx].w_stretch  # Get the gss value for the current simulation

        # Retrieve metrics
        metrics = controller.metrics
        i_idx = np.where(I_values == I)[0][0]
        gss_idx = np.where(w_stretch_values == w_stretch)[0][0]

        frequency_values[i_idx, gss_idx] = metrics['frequency']
        wavefrequency_values[i_idx, gss_idx] = metrics['wavefrequency']
        forward_speed_values[i_idx, gss_idx] = metrics['fspeed_PCA']

        # Plot muscle activities
        plt.figure(f'muscle_activities_I_{I}_gss_{w_stretch}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.muscle_l,
            controller.muscle_r,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/muscle_activities_I_{I}_gss_{w_stretch}.png')  # Save the figure
        plt.close()

        # Plot trajectory
        if hasattr(controller, 'links_positions'):
            plt.figure(f"trajectory_I_{I}_gss_{w_stretch}")
            plot_trajectory(controller)
            plt.savefig(f'{log_path}/trajectory_I_{I}_gss_{w_stretch}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning(f"Controller {idx} does not have attribute 'links_positions'. Cannot plot trajectory.")

        # Plot joint positions
        if hasattr(controller, 'joints_positions'):
            plt.figure(f"joint_positions_I_{I}_gss_{w_stretch}")
            plot_time_histories(
                controller.times,
                controller.joints_positions,
                offset=-0.4,
                colors="green",
                ylabel="joint positions",
                lw=1
            )
            plt.savefig(f'{log_path}/joint_positions_I_{I}_gss_{w_stretch}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning(f"Controller {idx} does not have attribute 'joints_positions'. Cannot plot joint positions.")

    # Plotting heatmaps for the metrics
    plt.figure('Heatmap: Frequency vs I and gss')
    sns.heatmap(frequency_values, xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis')
    plt.xlabel('gss')
    plt.ylabel('I')
    plt.title('Heatmap: Frequency vs I and gss')
    plt.savefig(f'{log_path}/heatmap_frequency_vs_I_gss.png')
    plt.close()

    plt.figure('Heatmap: Wave Frequency vs I and gss')
    sns.heatmap(wavefrequency_values, xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis')
    plt.xlabel('gss')
    plt.ylabel('I')
    plt.title('Heatmap: Wave Frequency vs I and gss')
    plt.savefig(f'{log_path}/heatmap_wavefrequency_vs_I_gss.png')
    plt.close()

    plt.figure('Heatmap: Forward Speed vs I and gss')
    sns.heatmap(forward_speed_values, xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis')
    plt.xlabel('gss')
    plt.ylabel('I')
    plt.title('Heatmap: Forward Speed vs I and gss')
    plt.savefig(f'{log_path}/heatmap_forward_speed_vs_I_gss.png')
    plt.close()

    pylog.info("Plots saved successfully")

if __name__ == '__main__':
    exercise7()