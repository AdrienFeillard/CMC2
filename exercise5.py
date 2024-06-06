from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories

def exercise5():

    pylog.info("Ex 5")
    pylog.info("Implement exercise 5")
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    Idiff_values = np.linspace(0, 4, num=100)

    # List of SimulationParameters with varying Idiff
    params_list = [
        SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            Idiff=Idiff,  # Varying Idiff
            video_record=False,  # Disable video recording
            video_name=f"exercise5_simulation_{i}",  # Name of the video file
            video_fps=30  # Frames per second
        ) for i, Idiff in enumerate(Idiff_values)
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list, num_process=10)

    pylog.info("Simulations finished")

    # Initialize lists to store metrics
    curvature_values = []
    lateral_speed_values = []
    radii = []

    for i, controller in enumerate(controllers):
        Idiff = params_list[i].Idiff  # Get the Idiff value for the current simulation

        # Retrieve metrics
        metrics = controller.metrics
        curvature_values.append(metrics['curvature'])
        lateral_speed_values.append(metrics['lspeed_PCA'])

        # Calculate turning radius
        radius = 1 / metrics['curvature'] if metrics['curvature'] != 0 else np.inf
        radii.append(radius)

        # Plot muscle activities
        plt.figure(f'muscle_activities_{Idiff}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.muscle_l,
            controller.muscle_r,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/muscle_activities_Idiff_{Idiff}.png')  # Save the figure
        plt.close()

        # Plot trajectory
        if hasattr(controller, 'links_positions'):
            plt.figure(f"trajectory_{Idiff}")
            plot_trajectory(controller)
            plt.savefig(f'{log_path}/trajectory_Idiff_{Idiff}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning(f"Controller {i} does not have attribute 'links_positions'. Cannot plot trajectory.")

        # Plot joint positions
        if hasattr(controller, 'joints_positions'):
            plt.figure(f"joint_positions_{Idiff}")
            plot_time_histories(
                controller.times,
                controller.joints_positions,
                offset=-0.4,
                colors="green",
                ylabel="joint positions",
                lw=1
            )
            plt.savefig(f'{log_path}/joint_positions_Idiff_{Idiff}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning(f"Controller {i} does not have attribute 'joints_positions'. Cannot plot joint positions.")

    # Plotting curvature as a function of Idiff
    plt.figure('curvature_vs_Idiff')
    plt.plot(Idiff_values, curvature_values, marker='o')
    plt.xlabel('Idiff')
    plt.ylabel('Curvature')
    plt.title('Curvature vs Idiff')
    plt.grid(True)
    plt.savefig(f'{log_path}/curvature_vs_Idiff.png')
    plt.close()

    # Plotting lateral speed as a function of Idiff
    plt.figure('lateral_speed_vs_Idiff')
    plt.plot(Idiff_values, lateral_speed_values, marker='o')
    plt.xlabel('Idiff')
    plt.ylabel('Lateral Speed')
    plt.title('Lateral Speed vs Idiff')
    plt.grid(True)
    plt.savefig(f'{log_path}/lateral_speed_vs_Idiff.png')
    plt.close()

    # Plotting turning radius as a function of Idiff
    plt.figure('radius_vs_Idiff')
    plt.plot(Idiff_values, radii, marker='o')
    plt.xlabel('Idiff')
    plt.ylabel('Turning Radius')
    plt.title('Turning Radius vs Idiff')
    plt.grid(True)
    plt.savefig(f'{log_path}/radius_vs_Idiff.png')
    plt.close()

    pylog.info("Plots saved successfully")

if __name__ == '__main__':
    exercise5()