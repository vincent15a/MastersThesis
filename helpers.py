# Contains all functions used by all simulations
import networkx as nx
import random
from scipy.stats import binom
import subprocess
import itertools

HOTSPOT_PATH = "/home/vincent/MasterThesis/CustomHotSpot/hotspot"

class Processor:
    """
    Represents a 3D multi-processor with configurable amount of layters, PEs and ambient temperature

    ...

    Attributes
    ----------
    proc_per_layer : int
        the number of PEs on each layer of the processor
    nr_layers : int
        the number of layers the processor consists of
    config_directory : str
        path to directory that the Hotspot config files of this processor are located at
    T_amb : float
        the ambient temperature of the processors environment in Kelvin
    path_to_config : str
        complete path to the configuration files of the processor
    coefficients : list[list[int]]
        thermal coefficient matrix of the Processor (as defined by the Matrix Model in Chapter 2 of Thesis)

    Methods
    -------
    temperature(core_nr : int, power_trace : list[int]) : int
        returns the temperature of a core under a certain power trace
    tpp_to_power_trace(tpp: list[list[tuple[int, int, int, int]]], makespan : int, output_file : str, dt : float)
        saves the power trace of a tpp (tasks per PE) data structure
    get_stats(transientFile : str, return_top_1_percent : bool = False, avg : bool = False)
        returns the stats of a transient file produced by HotSpot
    compute_max_ptrace_temp(ptrace : str, return_top_1_percent : bool = False, avg_temp : bool = False)
        returns the maximum temperature of a power trace
    """
    def __init__(self, proc_per_layer: int, nr_layers: int, config_directory: str, T_amb: float):
        
        self.proc_per_layer = proc_per_layer # PEs on each layer of the processor
        self.nr_layers = nr_layers           # Number of layers
        self.nr_pes = self.proc_per_layer * self.nr_layers # Number of PEs in total
        self.path_to_config = self.__initialize_path_to_config(config_directory) # Path to Hotspot config files
        self.coefficients = self.__get_thermalcoefficients() # Thermal coefficients of processor
        self.T_amb = T_amb # Ambient temperature of processor

    def __initialize_path_to_config(self, config_directory):
        directory = config_directory
        directory += f"/{self.nr_layers}-layer-chip"

        if self.proc_per_layer == 9:
            directory += "-3x3"

        return directory

    def __get_thermalcoefficients(self):
        coefficients = []
        f = open(self.path_to_config + "/thermal_coefficients.txt", "r")
        for line in f.readlines():
            coefficients.append(list(map(float, line.strip().split())))
        return coefficients
    
    def temperature(self, core_nr, power_trace):
        """ Returns the steady state temperature of a core under a certain power trace.
        
        Parameters
        ----------
        core_nr : int
            The core number of the core whose temperature shall be computed
        power_trace : list[float]
            The power trace that is running, where the i'th index is the power consumption of core i
        
        """
        # Temperature upperbound, based on steady state temperature.
        return sum(self.coefficients[core_nr][j] * power_trace[j] for j in range(self.nr_pes)) + self.T_amb
    
    def write_power_trace(self, power_trace_lines, output):
        """
        Write power trace data to a specified output file.

        Parameters:
        -----------
        power_trace_lines : list[list[float]]
            A list where each sublist contains power consumption values for all PEs at a given timestep.
            Each sublist must have a length equal to `self.nr_pes`.
        output : str
            The file name to write the power trace data to. The file will be created in the directory
            specified by `self.path_to_config`.
        """
        f = open(f'{self.path_to_config}/{output}', 'w')
        for i in range(1, self.nr_layers + 1):
            for j in range(1, self.proc_per_layer + 1):
                f.write(f"PE_L{i}_{j} ")
        f.write("\n")
        
        for line in power_trace_lines:
            power_trace_line = []
            if len(line) != self.nr_pes:
                raise Exception("Wrong power trace file!")
            for nr in line:
                f.write(f"{nr} ")
                power_trace_line.append(nr)
            f.write("\n")

        f.close()
    
    def tpp_to_power_trace(self, tpp, makespan, output_file, dt):
        """
        Generate a power trace file from task per PE (tpp) data structure.

        Parameters:
        -----------
        tpp : list[list[tuple[int,float, float, float]]]
            A list where each element corresponds to a PE. Each element is a list of 
            tuples representing the tasks, where each tuple contains:
            - v (int): Task number
            - st (float): The start time of the task.
            - ft (float): The finish time of the task.
            - pow (float): The power consumption of the task during its execution.
        makespan : float
            The makespan of the power trace.
        output_file : str
            The file path where the generated power trace will saved to.
        dt : float
            The time interval between each timestep in the power trace.
        """
        for t in tpp:
            t.sort(key = lambda x: x[1])
        timesteps = int(makespan / dt) + 1
        power_trace = []
        for timestep in range(timesteps):
            t = timestep * dt
            power_line = []
            for p in range(self.nr_pes):
                for v, st, ft, pow in tpp[p]:
                    if round(st) <= t < round(ft):
                        power_line.append(pow)
                        break
                else:
                    power_line.append(0)
            power_trace.append(power_line)
        
        self.write_power_trace(power_trace, output_file)

    def get_stats(self, transientFile, return_top_1_percent = False, avg = False):
        """
        Compute and return temperature statistics from a transient file.

        Parameters:
        -----------
        transientFile : str
            The path to the file containing the transient temperatures.
        return_top_1_percent : bool, optional
            If True, return the 
            average of the top 1% of the highest temperatures measured. Defaults to False.
        avg : bool, optional
            If True, return the average temperature. Defaults to False.

        
        Returns:
        --------
        tuple: 
            - max_overall (float): The maximum temperature recorded across all cores.
            - If `return_top_1_percent` is True, also returns `max_top_1_percent` (float), 
            which is the average of the top 1% of the highest temperatures.
            - If `avg` is True, also returns `avg_overall` (float), which is the average temperature 
            across all cores.
        """
        f = open(f'{self.path_to_config}/{transientFile}', 'r')
        lines = f.readlines()
        f.close()
        names = lines[0].split()
        temp_lines = [list(map(float, line.split())) for line in lines[1:]]

        temp_per_core = [[t[i] for t in temp_lines] for i in range(len(names))]
        max_temp_core = [max(temp_per_core[i]) for i in range(len(names))]
        max_overall = max(max_temp_core)
        all_temps = list(itertools.chain.from_iterable(temps for temps in temp_lines))
        all_temps.sort(reverse = True)
        nr_temps = len(all_temps)
        
        if nr_temps // 100 > 0:
            top_1_percent = all_temps[:nr_temps // 100]
            max_top_1_percent = sum(top_1_percent) / len(top_1_percent)
        else:
            max_top_1_percent = max_overall

        avg_temp_core = [sum(temp_per_core[i]) / len(temp_per_core[i]) for i in range(len(names))]
        avg_overall = sum(avg_temp_core) / len(avg_temp_core)

        print(avg_overall)

        if return_top_1_percent:
            return max_overall, max_top_1_percent
        elif avg: 
            return max_overall, avg_overall
        else:
            return max_overall
    
    def compute_max_ptrace_temp(self, ptrace, return_top_1_percent = False, avg_temp = False):
        """
        Compute and return temperature statistics from a power trace.

        Parameters:
        -----------
        transientFile : str
            The path to the file containing the power trace.
        return_top_1_percent : bool, optional
            If True, return the 
            average of the top 1% of the highest temperatures measured. Defaults to False.
        avg : bool, optional
            If True, return the average temperature. Defaults to False.

        
        Returns:
        --------
        tuple: 
            - max_overall (float): The maximum temperature recorded across all cores.
            - If `return_top_1_percent` is True, also returns `max_top_1_percent` (float), 
            which is the average of the top 1% of the highest temperatures.
            - If `avg` is True, also returns `avg_overall` (float), which is the average temperature 
            across all cores.
        """
        subprocess.run(construct_hotspot_command(ptrace, steady_file=f'{ptrace}.steady', grid_steady_file=f'{ptrace}.grid.steady'), cwd=self.path_to_config)
        subprocess.run(construct_hotspot_command(ptrace, init_file=f'{ptrace}.steady', output=f'{ptrace}.transient'), cwd=self.path_to_config)
        return self.get_stats(f'{ptrace}.transient', return_top_1_percent, avg=avg_temp)
    
    
    

def create_random_graph(number_tasks, skewed = False):
    """
    Generate a task graph with specified characteristics.
    All task graphs will only consist out of nodes, no edges

    Parameters:
    -----------
    number_tasks : int
        The number of tasks (nodes) to include in the task graph.
    skewed : bool, optional
        If True, the power consumption values will be skewed according to a binomial distribution.
        If False, the power consumption values will be uniformly distributed. Default is False.

    Returns:
    --------
    nx.DiGraph
        A graph where each node represents a task with the following attributes:
        - 'weight': An integer representing the task's weight, randomly assigned between 1 and 20.
        - 'power_cons': An integer representing the task's power consumption.
            - If `skewed` is False, it is uniformly assigned between 20 and 95.
            - If `skewed` is True, it is assigned based on a binomial distribution.
    """
    task_graph = nx.DiGraph()
    random.seed(1234)

    if not skewed:
        for task in range(number_tasks):
            task_graph.add_node(task, weight=random.randint(1, 20), power_cons = random.randint(20, 95))

    else:
        range_of_powers = range(20, 95)
        probability = [binom.pmf(i, 75, 0.2) for i in range(75)]
        for task in range(number_tasks):
            task_graph.add_node(task, weight=random.randint(1, 20), power_cons = random.choices(range_of_powers, probability)[0])


    return task_graph





def construct_hotspot_command(power_trace, init_file=None, config=f'example.config', grid_layer_file=f"layers.lcf", steady_file=None, grid_steady_file=None, output=None, grid_output=None, grid = True, output_config=None):
    """
    Construct a command for running the HotSpot thermal simulation tool.

    Parameters:
    -----------
    power_trace : str
        The file path to the power trace input file.
    init_file : str, optional
        The file path to the initial temperature file, if provided.
    config : str, optional
        The file path to the HotSpot configuration file. Default is 'example.config'.
    grid_layer_file : str, optional
        The file path to the grid layer configuration file. Default is "layers.lcf".
    steady_file : str, optional
        The file path to save the steady-state temperature output file, if provided.
    grid_steady_file : str, optional
        The file path to save the grid steady-state temperature output file, if provided.
    output : str, optional
        The file path to save the transient temperature output file, if provided.
    grid_output : str, optional
        The file path to save the grid transient temperature output file, if provided.
    grid : bool, optional
        If True, the simulation will use the grid model. If False, it will use the block model. Default is True.
    output_config : str, optional
        The file path to the output configuration file, if provided.

    Returns:
    --------
    list[str]
        A list of command-line arguments to run the HotSpot tool with the specified parameters.

    """
    command = [HOTSPOT_PATH, "-c", config, "-p", power_trace, "-grid_layer_file", grid_layer_file, "-model_type", "grid" if grid else "block", "-detailed_3D", "on"]
    if init_file != None:
        command = command + ["-init_file", init_file]
    if grid_steady_file != None:
        command = command + ["-grid_steady_file", grid_steady_file]
    if steady_file != None:
        command = command + ["-steady_file", steady_file]
    if output != None:
        command = command + ["-o", output]
    if grid_output != None:
        command = command + ["-grid_transient_file", grid_output]
    if output_config != None:
        command = command + ["-d", output_config]
    return command







