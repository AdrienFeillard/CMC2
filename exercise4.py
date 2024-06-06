from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories

def exercise4():

    pylog.info("Ex 4")
    pylog.info("Implement exercise 4")
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    I_values = np.linspace(0, 30, num=100)

    params_list = [
        SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            I=I,  # Varying I
            video_record=False,  # Disable video recording
            video_name=f"exercise4_simulation_{i}",  # Name of the video file
            video_fps=30  # Frames per second
        ) for i, I in enumerate(I_values)
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list, num_process=5)

    pylog.info("Simulations finished")

    # Initialize lists to store metrics
    ptcc_values = []
    frequency_values = []
    wavefrequency_values = []

    for i, controller in enumerate(controllers):
        metrics = controller.metrics
        ptcc_values.append(metrics['ptcc'])
        frequency_values.append(metrics['frequency'])
        wavefrequency_values.append(metrics['wavefrequency'])

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

    # Plotting ptcc, frequency, and wavefrequency as a function of I
    plt.figure('ptcc_vs_I')
    plt.plot(I_values, ptcc_values, marker='o')
    plt.xlabel('I')
    plt.ylabel('ptcc')
    plt.title('PTCC vs I')
    plt.grid(True)
    plt.savefig(f'{log_path}/ptcc_vs_I.png')
    plt.close()

    plt.figure('frequency_vs_I')
    plt.plot(I_values, frequency_values, marker='o')
    plt.xlabel('I')
    plt.ylabel('Frequency')
    plt.title('Frequency vs I')
    plt.grid(True)
    plt.savefig(f'{log_path}/frequency_vs_I.png')
    plt.close()

    plt.figure('wavefrequency_vs_I')
    plt.plot(I_values, wavefrequency_values, marker='o')
    plt.xlabel('I')
    plt.ylabel('Wave Frequency')
    plt.title('Wave Frequency vs I')
    plt.grid(True)
    plt.savefig(f'{log_path}/wavefrequency_vs_I.png')
    plt.close()

    pylog.info("Plots saved successfully")

if __name__ == '__main__':
    exercise4()