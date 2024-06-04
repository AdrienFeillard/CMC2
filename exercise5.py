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

    params_list = [
        SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            Idiff=Idiff,  # Varying Idiff
            video_record=True,  # Enable video recording
            video_name=f"exercise5_simulation_{i}",  # Name of the video file
            video_fps=30  # Frames per second
        ) for i, Idiff in enumerate(np.linspace(0, 4, num=10))
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list)

    pylog.info("Simulations finished")

    pylog.info("Plotting the results")

    for i, controller in enumerate(controllers):
        plt.figure(f'muscle_activities_{i}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.muscle_l,
            controller.muscle_r,
            cm="green",
            offset=0.1
        )

        if hasattr(controller, 'links_positions'):
            plt.figure(f"trajectory_{i}")
            plot_trajectory(controller)
        else:
            pylog.warning(f"Controller {i} does not have attribute 'links_positions'. Cannot plot trajectory.")

        if hasattr(controller, 'joints_positions'):
            plt.figure(f"joint_positions_{i}")
            plot_time_histories(
                controller.times,
                controller.joints_positions,
                offset=-0.4,
                colors="green",
                ylabel="joint positions",
                lw=1
            )
        else:
            pylog.warning(f"Controller {i} does not have attribute 'joints_positions'. Cannot plot joint positions.")

    plt.show()

if __name__ == '__main__':
    exercise5()