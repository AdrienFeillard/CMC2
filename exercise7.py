from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_center_of_mass_trajectory_with_circle

def calculate_total_distance(head_positions):
    """Calculate the total distance traveled by the head of the fish."""
    diffs = np.diff(head_positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(distances)
    return total_distance

def exercise7():

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    log_path = './logs/exercise7/'
    os.makedirs(log_path, exist_ok=True)

    Idiff_values = [0, 1, 2, 3, 4]
    I_values = np.round(np.linspace(0, 30, num=10), 1)
    w_stretch_values = np.round(np.linspace(0, 15, num=10), 1)

    params_list = [
        SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            Idiff=Idiff,  # Varying Idiff
            I=I,  # Varying I
            w_stretch=w_stretch,  # Varying feedback weight
            video_record=False,  # Disable video recording
            video_name=f"exercise7_simulation_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}",  # Name of the video file
            video_fps=30  # Frames per second
        ) for Idiff in Idiff_values for I in I_values for w_stretch in w_stretch_values
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list, num_process=10)

    pylog.info("Simulations finished")

    # Initialize arrays to store metrics
    frequency_values = np.zeros((len(Idiff_values), len(I_values), len(w_stretch_values)))
    wavefrequency_values = np.zeros((len(Idiff_values), len(I_values), len(w_stretch_values)))
    forward_speed_values = np.zeros((len(Idiff_values), len(I_values), len(w_stretch_values)))
    curvature_values = np.zeros((len(Idiff_values), len(I_values), len(w_stretch_values)))
    lateral_speed_values = np.zeros((len(Idiff_values), len(I_values), len(w_stretch_values)))
    radii = np.zeros((len(Idiff_values), len(I_values), len(w_stretch_values)))
    total_distance_values = np.zeros((len(Idiff_values), len(I_values), len(w_stretch_values)))

    pylog.info("Plotting the results")

    for idx, controller in enumerate(controllers):
        Idiff = params_list[idx].Idiff  # Get the Idiff value for the current simulation
        I = params_list[idx].I  # Get the I value for the current simulation
        w_stretch = params_list[idx].w_stretch  # Get the w_stretch value for the current simulation

        # Retrieve metrics
        metrics = controller.metrics
        idiff_idx = Idiff_values.index(Idiff)
        I_idx = np.where(I_values == I)[0][0]
        w_stretch_idx = np.where(w_stretch_values == w_stretch)[0][0]

        frequency_values[idiff_idx, I_idx, w_stretch_idx] = metrics['frequency']
        wavefrequency_values[idiff_idx, I_idx, w_stretch_idx] = metrics['wavefrequency']
        forward_speed_values[idiff_idx, I_idx, w_stretch_idx] = metrics['fspeed_PCA']
        curvature_values[idiff_idx, I_idx, w_stretch_idx] = metrics['curvature']
        lateral_speed_values[idiff_idx, I_idx, w_stretch_idx] = metrics['lspeed_PCA']

        # Calculate turning radius
        radius = 1 / metrics['curvature'] if metrics['curvature'] != 0 else np.inf
        radii[idiff_idx, I_idx, w_stretch_idx] = radius

        # Calculate total distance swum
        if hasattr(controller, 'links_positions'):
            head_positions = np.array(controller.links_positions)[:, 0, :]  # Extract head positions
            total_distance = calculate_total_distance(head_positions)
            total_distance_values[idiff_idx, I_idx, w_stretch_idx] = total_distance
        else:
            pylog.warning(f"Controller {idx} does not have attribute 'links_positions'. Cannot calculate total distance.")
            total_distance_values[idiff_idx, I_idx, w_stretch_idx] = np.nan

        # Plot muscle activities
        plt.figure(f'muscle_activities_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.muscle_l,
            controller.muscle_r,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/muscle_activities_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}.png')  # Save the figure
        plt.close()

        # Plot trajectory
        if hasattr(controller, 'links_positions'):
            plt.figure(f"trajectory_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}")
            plot_trajectory(controller)
            plt.savefig(f'{log_path}/trajectory_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning(f"Controller {idx} does not have attribute 'links_positions'. Cannot plot trajectory.")

        # Plot joint positions
        if hasattr(controller, 'joints_positions'):
            plt.figure(f"joint_positions_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}")
            plot_time_histories(
                controller.times,
                controller.joints_positions,
                offset=-0.4,
                colors="green",
                ylabel="joint positions",
                lw=1
            )
            plt.savefig(f'{log_path}/joint_positions_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning(f"Controller {idx} does not have attribute 'joints_positions'. Cannot plot joint positions.")

        # Plot center of mass trajectory if Idiff is within tolerance of key values
        if Idiff in [0, 1, 2, 3, 4]:
            plt.figure('Center of Mass trajectory')
            plot_center_of_mass_trajectory_with_circle(controller, label='Center of Mass', color='blue')
            plt.savefig(f'{log_path}/CenterofMass_Idiff_{Idiff}_I_{I}_w_stretch_{w_stretch}.png')
            plt.close()

    # Plotting heatmaps for the metrics
    for idiff_idx, Idiff in enumerate(Idiff_values):
        plt.figure(f'Frequency vs I and gss for Idiff={Idiff}')
        sns.heatmap(frequency_values[idiff_idx, :, :], xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis', fmt=".2f")
        plt.xlabel('gss')
        plt.ylabel('I')
        plt.title(f'Frequency vs I and gss for Idiff={Idiff}')
        plt.savefig(f'{log_path}/heatmap_frequency_vs_I_w_stretch_Idiff_{Idiff}.png')
        plt.close()

        plt.figure(f'Wave Frequency vs I and gss for Idiff={Idiff}')
        sns.heatmap(wavefrequency_values[idiff_idx, :, :], xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis', fmt=".2f")
        plt.xlabel('gss')
        plt.ylabel('I')
        plt.title(f'Wave Frequency vs I and gss for Idiff={Idiff}')
        plt.savefig(f'{log_path}/heatmap_wavefrequency_vs_I_w_stretch_Idiff_{Idiff}.png')
        plt.close()

        plt.figure(f'Forward Speed vs I and gss for Idiff={Idiff}')
        sns.heatmap(forward_speed_values[idiff_idx, :, :], xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis', fmt=".2f")
        plt.xlabel('gss')
        plt.ylabel('I')
        plt.title(f'Forward Speed vs I and gss for Idiff={Idiff}')
        plt.savefig(f'{log_path}/heatmap_forward_speed_vs_I_w_stretch_Idiff_{Idiff}.png')
        plt.close()

        plt.figure(f'Curvature vs I and gss for Idiff={Idiff}')
        sns.heatmap(curvature_values[idiff_idx, :, :], xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis', fmt=".2f")
        plt.xlabel('gss')
        plt.ylabel('I')
        plt.title(f'Curvature vs I and gss for Idiff={Idiff}')
        plt.savefig(f'{log_path}/heatmap_curvature_vs_I_w_stretch_Idiff_{Idiff}.png')
        plt.close()

        plt.figure(f'Lateral Speed vs I and gss for Idiff={Idiff}')
        sns.heatmap(lateral_speed_values[idiff_idx, :, :], xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis', fmt=".2f")
        plt.xlabel('gss')
        plt.ylabel('I')
        plt.title(f'Lateral Speed vs I and gss for Idiff={Idiff}')
        plt.savefig(f'{log_path}/heatmap_lateral_speed_vs_I_w_stretch_Idiff_{Idiff}.png')
        plt.close()

        plt.figure(f'Turning Radius vs I and gss for Idiff={Idiff}')
        sns.heatmap(radii[idiff_idx, :, :], xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis', fmt=".2f")
        plt.xlabel('gss')
        plt.ylabel('I')
        plt.title(f'Turning Radius vs I and gss for Idiff={Idiff}')
        plt.savefig(f'{log_path}/heatmap_radius_vs_I_w_stretch_Idiff_{Idiff}.png')
        plt.close()

        plt.figure(f'Total Distance Swum vs I and gss for Idiff={Idiff}')
        sns.heatmap(total_distance_values[idiff_idx, :, :], xticklabels=w_stretch_values, yticklabels=I_values, annot=True, cmap='viridis', fmt=".2f")
        plt.xlabel('gss')
        plt.ylabel('I')
        plt.title(f'Total Distance Swum vs I and gss for Idiff={Idiff}')
        plt.savefig(f'{log_path}/heatmap_total_distance_vs_I_w_stretch_Idiff_{Idiff}.png')
        plt.close()

    pylog.info("Plots saved successfully")

if __name__ == '__main__':
    exercise7()