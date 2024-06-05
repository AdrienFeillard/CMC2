from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories

def exercise7():

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    log_path = './logs/exercise7/'
    os.makedirs(log_path, exist_ok=True)

    params_list = [
        SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            I=I,  # Varying I
            gss=gss,  # Varying feedback weight
            video_record=False,  # Enable video recording
            video_name=f"exercise7_simulation_I_{I}_gss_{gss}",  # Name of the video file
            video_fps=30  # Frames per second
        ) for I in np.linspace(0, 30, num=5) for gss in np.linspace(0, 15, num=5)
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list, num_process=1)

    pylog.info("Simulations finished")

    pylog.info("Plotting the results")

    for i, controller in enumerate(controllers):
        w_stretch = params_list[i].w_stretch  # Get the w_stretch value for the current simulation

        plt.figure(f'muscle_activities_w_stretch_{w_stretch:.2f}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.muscle_l,
            controller.muscle_r,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/muscle_activities_w_stretch_{w_stretch:.2f}.png')  # Save the figure
        plt.close()

        if hasattr(controller, 'links_positions'):
            plt.figure(f"trajectory_w_stretch_{w_stretch:.2f}")
            plot_trajectory(controller)
            plt.savefig(f'{log_path}/trajectory_w_stretch_{w_stretch:.2f}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning("Controller does not have attribute 'links_positions'. Cannot plot trajectory.")

        if hasattr(controller, 'joints_positions'):
            plt.figure(f"joint_positions_w_stretch_{w_stretch:.2f}")
            plot_time_histories(
                controller.times,
                controller.joints_positions,
                offset=-0.4,
                colors="green",
                ylabel="joint positions",
                lw=1
            )
            plt.savefig(f'{log_path}/joint_positions_w_stretch_{w_stretch:.2f}.png')  # Save the figure
            plt.close()
        else:
            pylog.warning("Controller does not have attribute 'joints_positions'. Cannot plot joint positions.")

    pylog.info("Plots saved successfully")

if __name__ == '__main__':
    exercise7()