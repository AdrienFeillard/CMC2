from util.run_closed_loop import run_single  # Note the change from run_open_loop to run_closed_loop
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories
import numpy as np

def exercise3():

    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")
    log_path = './logs/exercise3/'
    os.makedirs(log_path, exist_ok=True)

    params = SimulationParameters(
        controller="firing_rate",  # Ensure we're using the firing rate controller
        n_iterations=5001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        I=10,  # default parameter, adjust as needed
        w_stretch=0,
        video_record=False,  # Disable video recording
        video_name="exercise3_simulation",  # Name of the video file
        video_fps=30  # Frames per second
    )

    pylog.info("Running the simulation")
    controller = run_single(params)

    pylog.info("Simulation finished")

    pylog.info("Plotting the results")

    # Plotting muscle activities
    plt.figure('muscle_activities')
    plot_left_right(
        controller.times,
        controller.state,
        controller.muscle_l,
        controller.muscle_r,
        cm="green",
        offset=0.1
    )
    plt.savefig(f'{log_path}/muscle_activities_3.png')
    plt.close()

    # Plotting CPG activities
    plt.figure('CPG_activities')
    plot_left_right(
        controller.times,
        controller.state,
        controller.left_v,
        controller.right_v,
        cm="green",
        offset=0.1
    )
    plt.savefig(f'{log_path}/CPG_activities_3.png')
    plt.close()

    # Plotting Muscle Cell (MC) activities
    plt.figure('MC_activities')
    plot_left_right(
        controller.times,
        controller.state,
        controller.left_m,
        controller.right_m,
        cm="green",
        offset=0.1
    )
    plt.savefig(f'{log_path}/MC_activities_3.png')
    plt.close()

    # Plotting trajectory
    if hasattr(controller, 'links_positions'):
        plt.figure("trajectory")
        plot_trajectory(controller)
        plt.savefig(f'{log_path}/trajectory_3.png')
        plt.close()
    else:
        pylog.warning("Controller does not have attribute 'links_positions'. Cannot plot trajectory.")

    # Plotting joint positions
    if hasattr(controller, 'joints_positions'):
        plt.figure("joint_positions")
        plot_time_histories(
            controller.times,
            controller.joints_positions,
            offset=-0.4,
            colors="green",
            ylabel="joint positions",
            lw=1
        )
        plt.savefig(f'{log_path}/joint_positions_3.png')
        plt.close()
    else:
        pylog.warning("Controller does not have attribute 'joints_positions'. Cannot plot joint positions.")

    pylog.info("Plots saved successfully")

if __name__ == '__main__':
    exercise3()