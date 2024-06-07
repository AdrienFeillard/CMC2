from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple , run_single
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories , plot_time_histories_multiple_windows

def exercise6():

    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'
    os.makedirs(log_path, exist_ok=True)

    w_stretch_values = np.linspace(0, 15, num=100)

    params_list = [
        SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            w_stretch=w_stretch,  # Varying w_stretch
            video_record=False,  # Disable video recording
            video_name=f"exercise6_simulation_{i}",  # Name of the video file
            video_fps=30  # Frames per second
        ) for i, w_stretch in enumerate(w_stretch_values)
    ]

    pylog.info("Running multiple simulations")
    controllers = run_multiple(params_list, num_process=10)

    pylog.info("Simulations finished")

    # Initialize lists to store metrics
    frequency_values = []
    wavefrequency_values = []
    forward_speed_values = []

    pylog.info("Plotting the results")

    for i, controller in enumerate(controllers):
        w_stretch = params_list[i].w_stretch  # Get the w_stretch value for the current simulation

        # Retrieve metrics
        metrics = controller.metrics
        frequency_values.append(metrics['frequency'])
        wavefrequency_values.append(metrics['wavefrequency'])
        forward_speed_values.append(metrics['fspeed_PCA'])

        # Plot CPG activities
        plt.figure(f'CPG_activities_w_stretch_{w_stretch:.2f}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.left_v,
            controller.right_v,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/CPG_activities_w_stretch_{w_stretch:.2f}.png')
        plt.close()

        # Plot muscle neuron activities
        plt.figure(f'muscle_activities_w_stretch_{w_stretch:.2f}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.muscle_l,
            controller.muscle_r,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/muscle_activities_w_stretch_{w_stretch:.2f}.png')
        plt.close()

        # Plot muscle neuron activities
        plt.figure(f'muscle_activities_w_stretch_2_{w_stretch:.2f}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.muscle_l,
            controller.muscle_r,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/muscle_activities_w_stretch_2_{w_stretch:.2f}.png')
        plt.close()

        # Plot sensory feedback activities
        plt.figure(f'sensory_feedback_activities_w_stretch_{w_stretch:.2f}')
        plot_left_right(
            controller.times,
            controller.state,
            controller.left_s,
            controller.right_s,
            cm="green",
            offset=0.1
        )
        plt.savefig(f'{log_path}/sensory_feedback_activities_w_stretch_{w_stretch:.2f}.png')
        plt.close()

        # Plot trajectory
        if hasattr(controller, 'links_positions'):
            plt.figure(f"trajectory_w_stretch_{w_stretch:.2f}")
            plot_trajectory(controller)
            plt.savefig(f'{log_path}/trajectory_w_stretch_{w_stretch:.2f}.png')
            plt.close()
        else:
            pylog.warning(f"Controller {i} does not have attribute 'links_positions'. Cannot plot trajectory.")

        # Plot joint positions
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
            plt.savefig(f'{log_path}/joint_positions_w_stretch_{w_stretch:.2f}.png')
            plt.close()
        else:
            pylog.warning(f"Controller {i} does not have attribute 'joints_positions'. Cannot plot joint positions.")

    # Plotting frequency as a function of w_stretch
    plt.figure('frequency_vs_w_stretch')
    plt.plot(w_stretch_values, frequency_values, marker='o')
    plt.xlabel('w_stretch')
    plt.ylabel('Frequency')
    plt.title('Frequency vs w_stretch')
    plt.grid(True)
    plt.savefig(f'{log_path}/frequency_vs_w_stretch.png')
    plt.close()

    # Plotting wavefrequency as a function of w_stretch
    plt.figure('wavefrequency_vs_w_stretch')
    plt.plot(w_stretch_values, wavefrequency_values, marker='o')
    plt.xlabel('w_stretch')
    plt.ylabel('Wave Frequency')
    plt.title('Wave Frequency vs w_stretch')
    plt.grid(True)
    plt.savefig(f'{log_path}/wavefrequency_vs_w_stretch.png')
    plt.close()

    # Plotting forward speed as a function of w_stretch
    plt.figure('forward_speed_vs_w_stretch')
    plt.plot(w_stretch_values, forward_speed_values, marker='o')
    plt.xlabel('w_stretch')
    plt.ylabel('Forward Speed')
    plt.title('Forward Speed vs w_stretch')
    plt.grid(True)
    plt.savefig(f'{log_path}/forward_speed_vs_w_stretch.png')
    plt.close()

    pylog.info("Plots saved successfully")

def exercise6b(**kwargs): 
    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    g_ss = 5

    # List of SimulationParameters with varying Idiff
    all_pars = SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=4001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            w_stretch =g_ss,  # Varying Idiff
            video_record=False,  # Disable video recording
            video_name=f"exercise5_simulation",  # Name of the video file
            video_fps=30 , # Frames per second
            **kwargs
        ) 
    

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )

    pylog.info("Plotting the result")

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
    plt.savefig(f'{log_path}/muscle_activities_6.png')
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
    plt.savefig(f'{log_path}/CPG_activities_6.png')
    plt.close()

    plt.figure('Sensory_activities')
    plot_left_right(
        controller.times,
        controller.state,
        controller.left_s,
        controller.right_s,
        cm="blue",  
        offset=0.1
    )
    plt.savefig(f'{log_path}/Sensory_activities_6.png')
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
    plt.savefig(f'{log_path}/MC_activities_6.png')
    plt.close()

    # example plot using plot_time_histories_multiple_windows
    plt.figure("joint positions_single")
    plot_time_histories_multiple_windows(
        controller.times,
        controller.joints_positions,
        offset=-0.4,
        colors="green",
        ylabel="joint positions",
        lw=1
    )
    plt.savefig(f'{log_path}/Joint_positions_6.png')
    plt.close()

    

if __name__ == '__main__':
    exercise6b(headless = True)