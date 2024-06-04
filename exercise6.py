from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_single
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories

def exercise6():

    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'
    os.makedirs(log_path, exist_ok=True)

    params = SimulationParameters(
        controller="firing_rate",  # Ensure we're using the firing rate controller
        n_iterations=5001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        gss=5,  # Example feedback weight
        video_record=True,  # Enable video recording
        video_name="exercise6_simulation",  # Name of the video file
        video_fps=30  # Frames per second
    )

    pylog.info("Running the simulation")
    controller = run_single(params)

    pylog.info("Simulation finished")

    pylog.info("Plotting the results")

    plt.figure('muscle_activities')
    plot_left_right(
        controller.times,
        controller.state,
        controller.muscle_l,
        controller.muscle_r,
        cm="green",
        offset=0.1
    )

    if hasattr(controller, 'links_positions'):
        plt.figure("trajectory")
        plot_trajectory(controller)
    else:
        pylog.warning("Controller does not have attribute 'links_positions'. Cannot plot trajectory.")

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
    else:
        pylog.warning("Controller does not have attribute 'joints_positions'. Cannot plot joint positions.")

    plt.show()

if __name__ == '__main__':
    exercise6()