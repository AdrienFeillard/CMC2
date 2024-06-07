from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
from util.run_closed_loop import run_single
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_time_histories ,plot_center_of_mass_trajectory



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

def exercise5b(**kwargs): 
    pylog.info("Ex 5")
    pylog.info("Implement exercise 5")
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    Idif = 2

    # List of SimulationParameters with varying Idiff
    all_pars = SimulationParameters(
            controller="firing_rate",  # Ensure we're using the firing rate controller
            n_iterations=4001,
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            Idiff=Idif,  # Varying Idiff
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
    plt.savefig(f'{log_path}/CPG_activities_5.png')
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
    plt.savefig(f'{log_path}/MC_activities_5.png')
    plt.close()

    metrics = controller.metrics
    radius = 1 / metrics['curvature'] if metrics['curvature'] != 0 else np.inf
    # Print comparison to ensure it matches the expected turning radius
    print(f"Computed radius: {radius}")

    # Plot center of mass trajectory
    plt.figure('Center of Mass trajectory')
    plot_center_of_mass_trajectory(controller, label='Center of Mass', color='blue')
    plt.show()
    plt.savefig(f'{log_path}/CenterofMAss.png')
    plt.close()
    


    



if __name__ == '__main__':
    exercise5()
    exercise5b()