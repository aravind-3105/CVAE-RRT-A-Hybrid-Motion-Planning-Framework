import sys
import os
import random
import pickle
from multiprocessing import Pool, cpu_count
from datetime import datetime
from configparser import ConfigParser
import ast

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_2d import Env2D
from rrt2d import RRT
from rrtstar2d import RRTstar
import argparse

def collect_data_for_env(parameters, env_name):
    """Data collection for a single environment."""
    print(f"Starting data collection for environment: {env_name}")
    start = parameters['start']
    goal = parameters['goal']
    x_lims = parameters['x_lims']
    y_lims = parameters['y_lims']
    num_each_map = parameters['num_each_map']
    time_horizon = parameters['time_horizon']
    data_collection_folder = parameters['data_collection_folder']
    data_collection_pickle_folder = parameters['data_collection_pickle_folder']
    extend_len = parameters['extend_len']

    # Data collection subfolder for the current environment
    data_collection_subfolder = f"{env_name}__{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    map_folder = os.path.join(parameters['map_folder'], env_name)
    map_subfolder = "train"
    maps = os.listdir(os.path.join(map_folder, map_subfolder))
    maps = [map for map in maps if map.endswith(".png")]
    num_maps = len(maps)

    # Create necessary directories
    os.makedirs(os.path.join(data_collection_folder, data_collection_subfolder), exist_ok=True)
    os.makedirs(os.path.join(data_collection_pickle_folder, data_collection_subfolder), exist_ok=True)

    data_buffer = []
    data_count = 0

    for num in range(num_each_map):
        # print every 50 iterations to keep track of progress
        print(f"Map {num + 1} of {num_each_map} for environment {env_name}")
        for i in range(num_maps):
            map = maps[i]
            map_path = os.path.join(map_folder, map_subfolder, map)

            env = Env2D()
            env.initialize(map_path, {'x_lims': x_lims, 'y_lims': y_lims})

            planner = RRTstar(start, env, extend_len=extend_len)

            if not env.collision_free(start) or not env.collision_free(goal):
                print(f"Skipping map {map} due to collision at start or goal")
                continue

            attempt = 0
            while attempt < time_horizon:
                attempt += 1
                if random.random() > 0.2:
                    sample_r = (
                        random.uniform(x_lims[0], x_lims[1] - 1),
                        random.uniform(y_lims[0], y_lims[1] - 1)
                    )
                    if planner.is_collision(sample_r) or planner.is_contain(sample_r):
                        continue
                else:
                    sample_r = goal

                new_node = planner.extend(sample_r)
                if not new_node:
                    continue

                planner.rewire(new_node)
                if planner.is_goal_reached(new_node, goal, goal_region_radius=3):
                    planner._q_goal_set.append(new_node)

            if planner._q_goal_set == []:
                continue

            data_count += 1
            planner.update_best(goal)
            solution = planner.get_solution_node(planner._q_best)

            for sample in solution:
                # Normalize the state to within the environment limits
                # If [0,100] are the limits, then (sample[0]/100, sample[1]/100)
                sample = ((sample[0]-x_lims[0])/(x_lims[1]-x_lims[0]),
                          (sample[1]-y_lims[0])/(y_lims[1]-y_lims[0]))
                # data_buffer.append((env.get_env_image(), state_normalize(sample)))
                data_buffer.append((env.get_env_image(), sample))

            if data_count % 10 == 0:
                env.initialize_plot(start, goal, plot_grid=False)
                env.plot_path(planner.reconstruct_path(planner._q_best), 'solid', 'red', 3)
                # env.plot_states(solution, 'green', alpha=0.8, msize=9)
                env.plot_states(solution, 'green')
                env.plot_save(f"{data_collection_folder}/{data_collection_subfolder}/{data_count}")

            if data_count % 50 == 0:
                file_name = os.path.join(
                    data_collection_pickle_folder,
                    data_collection_subfolder,
                    f"{env_name}__data_collection_partial.pickle"
                )
                with open(file_name, "wb") as file:
                    pickle.dump(data_buffer, file)
                    
            # Print every 50 iterations
            if data_count % 50 == 0:
                print(f"Data collection: {data_count} samples collected of map {map}")

    file_name = os.path.join(
        data_collection_pickle_folder,
        data_collection_subfolder,
        f"{env_name}_data_collection.pickle"
    )
    with open(file_name, "wb") as file:
        pickle.dump(data_buffer, file)
    
    # When the final data collection is complete, delete the partial file
    partial_file_name = os.path.join(
        data_collection_pickle_folder,
        data_collection_subfolder,
        f"{env_name}__data_collection_partial.pickle"
    )
    if os.path.exists(partial_file_name):
        os.remove(partial_file_name)

    print(f"Data collection complete for environment: {env_name}")
    return data_buffer


def data_collection_multiprocessing(parameters, env_names, num_processes=None):
    """Data collection with multiprocessing."""
    if num_processes is None:
        num_processes = cpu_count()

    # Print all the parameters
    print("Parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")
    print(f"Number of processes: {num_processes}")
    print(f"Environments to process: {env_names}")
    print(f"Starting multiprocessing with {num_processes} processes")
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(collect_data_for_env, [(parameters, env_name) for env_name in env_names])
    print("All environments processed")
    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Data collection for Project')
    parser.add_argument('--env', type=str, default=["forest"], nargs='+', 
                        help='List of environment names (default: ["bugtrap_forest"])", options are ["shifting_gaps","forest", "bugtrap_forest","mazes","alternating_gaps","gaps_and_forest","multiple_bugtraps","single_bugtrap"]')
    parser.add_argument('map_folder', type=str, default='../../motion_planning_datasets/', help='Map folder')
    parser.add_argument('data_collection_folder', type=str, default='data_collection', help='Data collection folder')
    parser.add_argument('data_collection_pickle_folder', type=str, default='data_collection_pickle', help='Data collection pickle folder')
    parser.add_argument('extend_len', type=float, default=5, help='Extend length for RRT')
    parser.add_argument('time_horizon', type=int, default=1000, help='Time horizon for RRT')
    parser.add_argument('num_each_map', type=int, default=5, help='Number of samples to collect for each map')
    parser.add_argument('x_lims', type=ast.literal_eval, default=(0, 100), help='X limits for the environment')
    parser.add_argument('y_lims', type=ast.literal_eval, default=(0, 100), help='Y limits for the environment')
    parser.add_argument('start', type=ast.literal_eval, default=(1, 1), help='Start position for the robot')
    parser.add_argument('goal', type=ast.literal_eval, default=(99, 99), help='Goal position for the robot')
    
    args = parser.parse_args()
    
    # To run:
        # Sample: python data_collection.py ../../motion_planning_datasets/ data_collection data_collection_pickle 5 1000 10 "(0,100)" "(0,100)" "(1,1)" "(99,99)" --env shifting_gaps forest
        # General formar: python data_collection.py map_folder data_collection_folder data_collection_pickle_folder extend_len time_horizon num_each_map x_lims y_lims start goal --env env1 env2 env3 ...
    
    parameters = {
        'start': args.start,
        'goal': args.goal,
        'x_lims': args.x_lims,
        'y_lims': args.y_lims,
        'num_each_map': args.num_each_map,
        'time_horizon': args.time_horizon,
        'data_collection_folder': args.data_collection_folder,
        'map_folder': args.map_folder,
        'extend_len': args.extend_len,
        'data_collection_pickle_folder': args.data_collection_pickle_folder
    }
    
    
    
    
    # parameters = {
    #     'start': (1, 1),
    #     'goal': (99, 99),
    #     'x_lims': (0, 100),
    #     'y_lims': (0, 100),
    #     'num_each_map': 5,
    #     'time_horizon': 1000,
    #     'data_collection_folder': "data_collection",
    #     # 'map_folder': config.get('paths', 'env_directory'),
    #     'map_folder': "./../../motion_planning_datasets/",
    #     'extend_len': 5,
    #     'data_collection_pickle_folder': "data_collection_pickle"
    # }
    
    # Print the parameters
    print("Parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")

    # List of environment names to process
    # env_names = ["shifting_gaps","forest", "bugtrap_forest"]
    env_names = ["bugtrap_forest"]
    data_collection_multiprocessing(parameters, env_names)
    print("Data collection for all environments completed")
