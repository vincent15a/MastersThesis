#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import networkx as nx
import random
import subprocess
import typing
from abc import ABC, ABCMeta
from scipy.optimize import curve_fit
from scipy.stats import binom
import numpy as np
import copy
import gurobipy as gp
from gurobipy import GRB

# Processor Specs
# Speed in GHz
proc_speed = 1.25

# proc_cores is the number of processors per layer.
proc_layers = 2
proc_cores = 4

proc_total = proc_layers * proc_cores

# Path to chip configuration
proc_config_path = "../../HotspotConfiguration/2-layer-chip"

# Location of the hotspot executable
hotspot = "/home/vincent/MasterThesis/CustomHotSpot/hotspot"

nr_tasks = None

timestep = 1e-1

T_amb = 318.15

# Power Parameter of Processor (power_usage = power_parameter * (speed in GHz)^3  )
power_parameter = 14.81481

# In[2]:


def powerUsage(speed):
    return power_parameter * (speed)**3

def speedAtPower(power):
    return (power / power_parameter)**(1/3)


# In[ ]:





# In[3]:


# PowerConsumptionLimit
coefficients = []
f = open(proc_config_path + "/thermal_coefficients.txt", "r")
for line in f.readlines():
    coefficients.append(list(map(float, line.strip().split())))

coefficients


# In[4]:


def read_data(data_path, limit = None):

    task_graph = nx.DiGraph()

    f = open(data_path, 'r')

    random.seed(1234)
    max_task_length = 0
    lines = f.readlines()
    if limit != None:
        lines = lines[:limit + 1]

    for line in lines:
        line = line.strip().split()
        if line[0] == "v":
            task_graph.add_node(int(line[1]), weight=int(line[2]), power_cons = random.randint(20, 60))
            max_task_length = max(max_task_length, int(line[2]))
        if line[0] == "e":
            task_graph.add_edge(int(line[2]), int(line[1]))

    pos = nx.spring_layout(task_graph, scale=15, weight='t', seed=20)

    random.seed(1234)
    nr_tasks = len(task_graph.nodes)
    nx.set_node_attributes(task_graph, dict(enumerate([random.randint(0, (nr_tasks * max_task_length) // 2) for _ in task_graph.nodes])), "arrival_time")
    

    # nx.draw(task_graph, pos=pos, with_labels=True)
    plt.show()
    return task_graph

def create_random_graph(number_tasks):
    task_graph = nx.DiGraph()
    random.seed(1234)

    for task in range(number_tasks):
        task_graph.add_node(task, weight=random.randint(1, 20), power_cons = random.randint(20, 95))

    
    # range_of_powers = range(20, 95)
    # probability = [binom.pmf(i, 75, 0.2) for i in range(75)]
    # print(probability)

    # for task in range(number_tasks):
    #     task_graph.add_node(task, weight=random.randint(1, 20), power_cons = random.choices(range_of_powers, probability)[0])


    # nx.draw(task_graph, pos=pos, with_labels=True)
    # plt.show()
    return task_graph


# In[5]:


create_random_graph(10)


# In[6]:


# task_graph = read_data("testdata.txt", 10)


# In[7]:


def create_model(graph : nx.DiGraph, T_crit : float, MIPGap = False):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j
    sig = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"sig[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # sig[i,j] == 1 iff task i overlaps with task j and st[i] <= st[j]
    sigp = [[[m.addVar(vtype=GRB.BINARY, name=f"sigp[{i},{j},{p}]") for p in range(nr_processors)] for j in range(nr_tasks)] for i in range(nr_tasks)] # sigp[i,j] == 1 iff sig[i,j] == 1 and x[j,p2] == 1 and x[i,p1] == 1
    d = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"d[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # d[i,j] == 1 iff task i and task j start at same time
    diff = [[m.addVar(lb=-float("inf"), name=f"diff[{i},{j}]") for j in range(i, nr_tasks)] for i in range(nr_tasks)] # diff[i,j] == start_time[i] - start_time[j]
    absdiff = [[m.addVar(name=f"absdiff[{i},{j}]") for j in range(i, nr_tasks)] for i in range(nr_tasks)] # absdiff[i,j] == |start_time[i] - start_time[j]|

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # Define sig[i][j]
    m.addConstrs((M * (1 - o[i][j]) + M * (1 - sig[i][j]) + M * d[i][j] + start_time[i] >= start_time[j] + 0.5) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs(((1 - o[i][j]) + sig[i][j] + sig[j][i] >= 1) for i in range(nr_tasks) for j in range(nr_tasks))
    # add constr if st[i] == st[j] => sig[i][j] == sig[j][i] = 1 wrs

    # Define sigp[i][j][p]
    m.addConstrs(((sig[i][j] + x[j][p]) + 3 * (1 - sigp[i][j][p]) <= 4) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) )

    # Define delta[u][v]
    m.addConstrs((diff[i][j] == start_time[i] - start_time[i + j]) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] == gp.abs_(diff[i][j])) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] <= 1/2 +  M * (1 - d[i][i + j])) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] >= 1 -  M * d[i][i + j]) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs(d[i][j] == d[j][i] for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((2 * (1 - d[i][j]) + sig[i][j] + sig[j][i] >= 2) for i in range(nr_tasks) for j in range(nr_tasks))


    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # Do not overheat
    m.addConstrs((sum(sum(coefficients[p1][p2] * sigp[u][v][p2] * task_power[v] for v in range(nr_tasks)) for p2 in range(nr_processors)) <= T_crit - T_amb) for u in range(nr_tasks) for p1 in range(nr_processors) )

    m.setParam('TimeLimit', 1200)
    # Optimize model
    m.optimize()


    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    for u in range(nr_tasks):
        for p1 in range(nr_processors):
            print(f"T[{u},{p1}] = ", T_amb + sum(sum(coefficients[p1][p2] * sigp[u][v][p2].X * task_power[v] for v in range(nr_tasks)) for p2 in range(nr_processors)))
        
    ms = round(makespan.X)
    gap = float(m.MIPGap)

    m.dispose()
    
    if MIPGap:
        return ms, tasks_per_proc, gap
    return ms, tasks_per_proc



# In[8]:


def create_model_relaxed(graph : nx.DiGraph, T_crit : float):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j
    sig = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"sig[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # sig[i,j] == 1 iff task i overlaps with task j and st[i] <= st[j]
    sigp = [[[m.addVar(lb=0, ub = 1/3, name=f"sigp[{i},{j},{p}]") for p in range(nr_processors)] for j in range(nr_tasks)] for i in range(nr_tasks)] # sigp[i,j] == 1 iff sig[i,j] == 1 and x[j,p2] == 1 and x[i,p1] == 1
    d = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"d[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # d[i,j] == 1 iff task i and task j start at same time
    diff = [[m.addVar(lb = -float("inf"), name=f"diff[{i},{j}]") for j in range(i, nr_tasks)] for i in range(nr_tasks)] # diff[i,j] == start_time[i] - start_time[j]
    absdiff = [[m.addVar(name=f"absdiff[{i},{j}]") for j in range(i, nr_tasks)] for i in range(nr_tasks)] # absdiff[i,j] == |start_time[i] - start_time[j]|

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # Define sig[i][j]
    m.addConstrs((M * (1 - o[i][j]) + M * (1 - sig[i][j]) + M * d[i][j] + start_time[i] >= start_time[j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs(((1 - o[i][j]) + sig[i][j] + sig[j][i] >= 1) for i in range(nr_tasks) for j in range(nr_tasks))
    # add constr if st[i] == st[j] => sig[i][j] == sig[j][i] = 1 wrs

    # Define sigp[i][j][p]
    m.addConstrs(((sig[i][j] + x[j][p]) + 3 * (1 - sigp[i][j][p]) <= 4) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) )

    # Define delta[u][v]
    m.addConstrs((diff[i][j] == start_time[i] - start_time[i + j]) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] == gp.abs_(diff[i][j])) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] <= 1/2 +  M * (1 - d[i][i + j])) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] >= 1 -  M * d[i][i + j]) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs(d[i][j] == d[j][i] for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((2 * (1 - d[i][j]) + sig[i][j] + sig[j][i] >= 2) for i in range(nr_tasks) for j in range(nr_tasks))


    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # Do not overheat
    m.addConstrs((sum(sum(coefficients[p1][p2] * 3 * sigp[u][v][p2] * task_power[v] for v in range(nr_tasks)) for p2 in range(nr_processors)) <= T_crit - T_amb) for u in range(nr_tasks) for p1 in range(nr_processors) )

    # Optimize model
    m.optimize()

    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    for u in range(nr_tasks):
        for p1 in range(nr_processors):
            print(f"T[{u},{p1}] = ", T_amb + sum(sum(coefficients[p1][p2] * 3 * sigp[u][v][p2].X * task_power[v] for v in range(nr_tasks)) for p2 in range(nr_processors)))
        
    ms = round(makespan.X)
    m.dispose()
    return ms, tasks_per_proc

    


# In[9]:


def create_model_arrivaltimes(graph : nx.DiGraph, T_crit : float):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    random.seed(1234)
    max_task_length = max(task_lengths)
    task_arrival_time = [random.randint(0, (nr_tasks * max_task_length) // 2) for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j
    sig = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"sig[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # sig[i,j] == 1 iff task i overlaps with task j and st[i] <= st[j]
    sigp = [[[m.addVar(lb=0, ub = 1/3, name=f"sigp[{i},{j},{p}]") for p in range(nr_processors)] for j in range(nr_tasks)] for i in range(nr_tasks)] # sigp[i,j] == 1 iff sig[i,j] == 1 and x[j,p2] == 1 and x[i,p1] == 1
    d = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"d[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # d[i,j] == 1 iff task i and task j start at same time
    diff = [[m.addVar(name=f"diff[{i},{j}]") for j in range(i, nr_tasks)] for i in range(nr_tasks)] # diff[i,j] == start_time[i] - start_time[j]
    absdiff = [[m.addVar(name=f"absdiff[{i},{j}]") for j in range(i, nr_tasks)] for i in range(nr_tasks)] # absdiff[i,j] == |start_time[i] - start_time[j]|

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")


    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # Define sig[i][j]
    m.addConstrs((M * (1 - o[i][j]) + M * (1 - sig[i][j]) + M * d[i][j] + start_time[i] >= start_time[j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs(((1 - o[i][j]) + sig[i][j] + sig[j][i] >= 1) for i in range(nr_tasks) for j in range(nr_tasks))
    # add constr if st[i] == st[j] => sig[i][j] == sig[j][i] = 1 wrs

    # Define sigp[i][j][p]
    m.addConstrs(((sig[i][j] + x[j][p]) + 3 * (1 - sigp[i][j][p]) <= 4) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) )

    # Define delta[u][v]
    m.addConstrs((diff[i][j] == start_time[i] - start_time[i + j]) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] == gp.abs_(diff[i][j])) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] <= 1/2 +  M * (1 - d[i][i + j])) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((absdiff[i][j] >= 1 -  M * d[i][i + j]) for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs(d[i][j] == d[j][i] for i in range(nr_tasks) for j in range(nr_tasks - i))
    m.addConstrs((2 * (1 - d[i][j]) + sig[i][j] + sig[j][i] >= 2) for i in range(nr_tasks) for j in range(nr_tasks))


    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # Do not overheat
    m.addConstrs((sum(sum(coefficients[p1][p2] * 3 * sigp[u][v][p2] * task_power[v] for v in range(nr_tasks)) for p2 in range(nr_processors)) <= T_crit - T_amb) for u in range(nr_tasks) for p1 in range(nr_processors) )

    # Consider arrival times
    m.addConstrs(start_time[i] >= task_arrival_time[i] for i in range(nr_tasks))

    # Optimize model
    m.optimize()


    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    for u in range(nr_tasks):
        for p1 in range(nr_processors):
            print(f"T[{u},{p1}] = ", T_amb + sum(sum(coefficients[p1][p2] * 3 * sigp[u][v][p2].X * task_power[v] for v in range(nr_tasks)) for p2 in range(nr_processors)))
    
        
    
    print("Makespan", makespan.X)
    return makespan.X, tasks_per_proc

    


# In[10]:


def create_model_discrete(graph : nx.DiGraph, T_crit : float):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    timesteps = list(range(sum(task_lengths) + 1))
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(vtype=GRB.INTEGER, name="makespan")
    start_time = [m.addVar(vtype=GRB.INTEGER, name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j
    ab = [[m.addVar(vtype=GRB.BINARY, name=f"ab[{i},{t}]") for t in timesteps] for i in range(nr_tasks)] # ab[i,t] == 1 <= task i has been active before time t
    aa = [[m.addVar(vtype=GRB.BINARY, name=f"aa[{i},{t}]") for t in timesteps] for i in range(nr_tasks)] # aa[i,t] == 1 <= task i has been active after time t
    ca = [[m.addVar(vtype=GRB.BINARY, name=f"ca[{i},{t}]") for t in timesteps] for i in range(nr_tasks)] # ca[i,t] == 1 <= task i is currently active
    Ppow = [[m.addVar(vtype=GRB.INTEGER, name=f"p[{p},{t}]") for t in timesteps] for p in range(nr_processors)] # Ppow[p,t] == power consumption on PE p at time t

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((gp.quicksum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # Define ab, aa, ca
    m.addConstrs((start_time[i] + 1 <= t + M * (1 - ab[i][t])) for i in range(nr_tasks) for t in timesteps)
    m.addConstrs((M * ab[i][t] + start_time[i] >= t) for i in range(nr_tasks) for t in timesteps)
    m.addConstrs((M * (1 - aa[i][t]) + start_time[i] + task_lengths[i] >= t + 1) for i in range(nr_tasks) for t in timesteps)
    m.addConstrs((start_time[i] + task_lengths[i] <= t + M * (aa[i][t])) for i in range(nr_tasks) for t in timesteps)
    m.addConstrs((ca[i][t] + (1 - ab[i][t]) + (1 - aa[i][t]) >= 1) for i in range(nr_tasks) for t in timesteps)

    # Define Ppow
    M2 = max(task_power)
    m.addConstrs((M2 * (1 - x[i][p]) + M2 * (1 - ca[i][t]) + Ppow[p][t] >= task_power[i]) for i in range(nr_tasks) for p in range(nr_processors) for t in timesteps)

    # Do not overheat
    m.addConstrs((gp.quicksum(coefficients[p1][p2] * Ppow[p2][t] for p2 in range(nr_processors)) <= T_crit - T_amb) for p1 in range(nr_processors) for t in timesteps)

    # Optimize model
    m.optimize()

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])

    
    print("Max temp : ", max([T_amb + sum(coefficients[p1][p2] * Ppow[p2][t].X for p2 in range(nr_processors)) for p1 in range(nr_processors) for t in timesteps]))

    print("Makespan", makespan.X)

    


# In[ ]:





# In[11]:


def splitting_heuristic(graph : nx.DiGraph, T_crit : float, group_size : int):
    class Graph:
        def __init__(self) -> None:
            self.nodes = dict()
    nr_tasks = len(graph.nodes)
    nr_groups = nr_tasks // group_size
    if nr_tasks % group_size != 0:
        nr_groups += 1
    
    complete_tpp = [[] for _ in range(proc_total)]
    complete_ms = 0
    for i in range(nr_groups):
        modified_graph = Graph()
        index = 0
        for r in range(group_size * i, min(group_size * (i + 1), nr_tasks)):
            modified_graph.nodes[index] = graph.nodes[r]
            index += 1
        print(modified_graph.nodes)
        ms, tpp = create_model(modified_graph, T_crit)
        for p in range(proc_total):
            for t, st, ft, pow in tpp[p]:
                complete_tpp[p].append((t + i * group_size, st + complete_ms, ft + complete_ms, pow))
        complete_ms += ms
    return complete_ms , complete_tpp


def combine_tpp(tpp1, tpp2):
    ms1 = 0
    for ptpp in tpp1:
        for i, st, ft, pow in ptpp:
            ms1 = max(ms1, ft)
    ms2 = 0
    complete_tpp = copy.deepcopy(tpp1)
    for p in range(len(tpp2)):
        for i, st, ft, pow in tpp2[p]:
            ms2 = max(ms2, ft)
            complete_tpp[p].append((i, st + ms1, ft + ms1, pow))
 
    return ms1 + ms2, complete_tpp
    
    



# In[12]:


def temperature(core_nr, power_trace):
    # Temperature upperbound, based on steady state temperature.
    return sum(coefficients[core_nr][j] * power_trace[j] for j in range(proc_total)) + T_amb
    
def greedy_scheduling(graph : nx.DiGraph, T_crit: float):
    tpp = [[] for _ in range(proc_total)]
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    
    makespan = 0

    def power_consumption_of_core(time, core):
        for t, st, ft, pow in tpp[core]:
            if round(st) <= time < round(ft):
                return pow
        return 0


    for i in range(nr_tasks):
        opt_t = None
        opt_p = None
        for p in range(nr_processors):
            possible_time_stamps = []
            for t in range(makespan):
                if power_consumption_of_core(t, p) == 0:
                    power_trace = [power_consumption_of_core(t, x) for x in range(nr_processors)]
                    power_trace[p] = task_power[i]
                    for p2 in range(nr_processors):
                        if temperature(p2, power_trace) > T_crit:
                            break
                    else:
                        possible_time_stamps.append(t)
            for t in possible_time_stamps:
                for t_run in range(t, t + task_lengths[i]):
                    if t_run not in possible_time_stamps and t_run <= makespan:
                        break
                else:
                    if opt_t == None or t < opt_t:
                        opt_t = t
                        opt_p = p
            else:
                if opt_t == None:
                    opt_t = makespan
                    opt_p = p
        
        tpp[opt_p].append((i, opt_t, opt_t + task_lengths[i], task_power[i]))
        makespan = max(makespan, opt_t + task_lengths[i])
    
    return makespan, tpp
    


                
    
        


# In[13]:


# temperature(1, [24, 84, 31, 0, 0, 0, 0, 0 ])


# In[14]:


# task_graph = read_data("testdata.txt")
# ms, tpp = splitting_heuristic(task_graph, 450, 10)


# In[15]:


# ms2, tpp2 = greedy_scheduling(task_graph, 399)


# In[16]:


# tpp2


# In[17]:


# tpp


# In[18]:


def write_power_trace(power_trace_lines, output):
    f = open(f'{proc_config_path}/{output}', 'w')
    for i in range(1, proc_layers + 1):
        for j in range(1, proc_cores + 1):
            f.write(f"PE_L{i}_{j} ")
    f.write("\n")
    
    for line in power_trace_lines:
        power_trace_line = []
        if len(line) != proc_layers * proc_cores:
            raise Exception("Wrong power trace file!")
        for nr in line:
            f.write(f"{nr} ")
            power_trace_line.append(nr)
        f.write("\n")

    f.close()

def tpp_to_power_trace(tpp, makespan, output_file, dt):
    # dt = timestep
    for t in tpp:
        t.sort(key = lambda x: x[1])
    timesteps = int(makespan / dt) + 1
    power_trace = []
    for timestep in range(timesteps):
        t = timestep * dt
        power_line = []
        for p in range(proc_total):
            for v, st, ft, pow in tpp[p]:
                if round(st) <= t < round(ft):
                    power_line.append(pow)
                    break
            else:
                power_line.append(0)
        power_trace.append(power_line)
    
    write_power_trace(power_trace, output_file)
    
    
def construct_hotspot_command(power_trace, init_file=None, config=f'example.config', grid_layer_file=f"layers.lcf", steady_file=None, grid_steady_file=None, output=None, grid_output=None, grid = True, output_config=None):
    command = [hotspot, "-c", config, "-p", power_trace, "-grid_layer_file", grid_layer_file, "-model_type", "grid" if grid else "block", "-detailed_3D", "on"]
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
import itertools
def get_stats(transientFile, return_top_1_percent = False, avg = False):
    f = open(f'{proc_config_path}/{transientFile}', 'r')
    lines = f.readlines()
    f.close()
    names = lines[0].split()
    temp_lines = [list(map(float, line.split())) for line in lines[1:]]

    temp_per_core = [[t[i] for t in temp_lines] for i in range(len(names))]
    max_temp_core = [max(temp_per_core[i]) for i in range(len(names))]
    min_temp_core = [min(temp_per_core[i]) for i in range(len(names))]
    max_overall = max(max_temp_core)
    all_temps = list(itertools.chain.from_iterable(temps for temps in temp_lines))
    all_temps.sort(reverse = True)
    nr_temps = len(all_temps)
    
    if nr_temps // 100 > 0:
        top_1_percent = all_temps[:nr_temps // 100]
        max_top_1_percent = sum(top_1_percent) / len(top_1_percent)
    else:
        max_top_1_percent = max_overall
    min_overall = min(min_temp_core)

    avg_temp_core = [sum(temp_per_core[i]) / len(temp_per_core[i]) for i in range(len(names))]
    avg_overall = sum(avg_temp_core) / len(avg_temp_core)

    print(avg_overall)

    if return_top_1_percent:
        return max_overall, max_top_1_percent
    elif avg: 
        return max_overall, avg_overall
    else:
        return max_overall
    


def compute_max_ptrace_temp(ptrace, return_top_1_percent = False, avg_temp = False):
    subprocess.run(construct_hotspot_command(ptrace, steady_file=f'{ptrace}.steady', grid_steady_file=f'{ptrace}.grid.steady'), cwd=proc_config_path)
    subprocess.run(construct_hotspot_command(ptrace, init_file=f'{ptrace}.steady', output=f'{ptrace}.transient'), cwd=proc_config_path)
    return get_stats(f'{ptrace}.transient', return_top_1_percent, avg=avg_temp)


# In[ ]:





# In[19]:


import time
import sys
# output_file_normal = "output_normal_model_approx.txt"
# output_file_relaxed = "output_relaxed_model.txt"
# output_file_splitting = "output_splitting_model.txt"
output_file_greedy = "output_greedy_model.txt"

# # f_normal = open(output_file_normal, 'w')
# # f_relaxed = open(output_file_relaxed, 'w')
# # f_splitting = open(output_file_splitting, 'w')
# # f_greedy = open(output_file_greedy, 'w')
# # f_normal.close()
# # f_relaxed.close()
# # f_splitting.close()
# # f_greedy.close()

prev_time = 0
prev_makespan = 0
prev_tpp = [[] for _ in range(proc_total)]
# for i in [15]:
# for i in range(100, 501, 10):
# # for i in [400]:
#     task_graph = create_random_graph(i)
#     # start = time.time()
#     # ms, tpp, gap = create_model(task_graph, 400, True)
#     # tpp_to_power_trace(tpp, ms, output_file_normal, 1e-1)
#     # t_max = compute_max_ptrace_temp(output_file_normal)
#     # f = open(output_file_normal, 'a')
#     # f.write(f"Nr tasks {i}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max} , GAP {gap * 100}%\n")
#     # f.close()
#     # if gap * 100 > 25:
#     #     break

#     # start = time.time()
#     # ms, tpp = create_model_relaxed(task_graph, 400)
#     # tpp_to_power_trace(tpp, ms, output_file_relaxed, 1e-1)
#     # t_max = compute_max_ptrace_temp(output_file_relaxed)
#     # f = open(output_file_relaxed, 'a')
#     # f.write(f"Nr tasks {i}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max} \n")
#     # f.close()

#     start = time.time()
#     ms, tpp = greedy_scheduling(task_graph, 400)
#     tpp_to_power_trace(tpp, ms, output_file_greedy, 1e-1)
#     end = time.time()
#     t_max, top1p = compute_max_ptrace_temp(output_file_greedy, True)
#     f = open(output_file_greedy, 'a')
#     f.write(f"Nr tasks {i}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Top 1 percent highs {top1p} \n")
#     f.close()
#     # size = 10
    
#     # start = time.time()
#     # reduced_task_graph = task_graph.copy()
#     # for j in range(max(0, i - (i % size))):
#     #     reduced_task_graph.remove_node(j)


#     # ms, tpp = create_model(reduced_task_graph, 400)

#     # print(tpp)
#     # ms, tpp = combine_tpp(prev_tpp, tpp)
#     # print(tpp)
#     # end = time.time() - start
#     # end = end + prev_time

#     # if (i + 1) % size == 0:
#     #     prev_time = end
#     #     prev_tpp = tpp


#     # tpp_to_power_trace(tpp, ms, output_file_splitting, 1e-1)

#     # t_max = compute_max_ptrace_temp(output_file_splitting)
#     # f = open(output_file_splitting, 'a')
#     # f.write(f"Nr tasks {i + 1}, Time {end}, Makespan {round(ms)}, Max Temp {t_max} \n")
#     # f.close()

    

# In[ ]:


# import time
# import sys
# f = open("out_different_times.txt", 'w')
# f.close()
# for i in range(1, 52):
#     start = time.time()
#     task_graph = read_data("testdata.txt", i)
#     create_model_arrivaltimes(task_graph, 400)
#     f = open("out_different_times.txt", 'a')
#     f.write(f"Nr tasks {i} Time {time.time() - start} \n")
#     f.close()


# In[ ]:


# task_graph = read_data("testdata.txt")


# In[ ]:


# for i in range(len(task_graph.nodes)):
#     print(i, task_graph.nodes[i])


# In[ ]:


def create_lb_model(graph : nx.DiGraph):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    
    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs(((sum(task_lengths[i] * x[i][p] for i in range(nr_tasks)) <= makespan) for p in range(nr_processors)), name="ms_def")
    
    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Optimize model
    m.optimize()


    tasks_per_proc = [[] for _ in range(nr_processors)]

    load_per_p = [0 for p in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", load_per_p[p], ", finish time:", load_per_p[p] + task_lengths[i])
                tasks_per_proc[p].append((i, load_per_p[p], load_per_p[p] + task_lengths[i], task_power[i]))
                load_per_p[p] += task_lengths[i]
    print(load_per_p)
    ms = round(makespan.X)
    m.dispose()
    return ms, tasks_per_proc



# In[ ]:


def create_TATSND2_model(graph : nx.DiGraph):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    task_power_level = [graph.nodes[i]['power_level'] for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # No two tasks of power level 2 may overlap
    m.addConstrs((o[i][j] <= 0) for i in range(nr_tasks) for j in range(nr_tasks) if (i != j and task_power_level[i] == 1 and task_power_level[j] == 1))

    # Optimize model
    m.optimize()


    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    ms = round(makespan.X)
    m.dispose()
    return ms, tasks_per_proc



# In[ ]:


def create_TATSND3_model(graph : nx.DiGraph):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    task_power_level = [graph.nodes[i]['power_level'] for i in graph.nodes]
    # task_power_level = [2 for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # No two tasks of power level 1 may overlap
    m.addConstrs((o[i][j] <= 0) for i in range(nr_tasks) for j in range(nr_tasks) if (i != j and task_power_level[i] == 1 and task_power_level[j] == 1))

    # No 3 tasks of power level 2 may overlap
    m.addConstrs((o[i][j] + o[j][k] + o[i][k] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for k in range(nr_tasks) if (i != j and j != k and i != k and task_power_level[i] == 2 and task_power_level[j] == 2 and task_power_level[k] == 2))

    # Optimize model
    m.optimize()


    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    ms = round(makespan.X)
    m.dispose()
    return ms, tasks_per_proc



# In[ ]:


def create_TATSND4_model(graph : nx.DiGraph):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    task_power_level = [graph.nodes[i]['power_level'] for i in graph.nodes]
    # task_power_level = [2 for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # No two tasks of power level 1 may overlap
    m.addConstrs((o[i][j] <= 0) for i in range(nr_tasks) for j in range(nr_tasks) if (i != j and task_power_level[i] == 1 and task_power_level[j] == 1))

    # No 3 tasks of power level 2 may overlap
    m.addConstrs((o[i][j] + o[j][k] + o[i][k] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for k in range(nr_tasks) if (i != j and j != k and i != k and task_power_level[i] == 2 and task_power_level[j] == 2 and task_power_level[k] == 2))

    # No 4 tasks of power level 3 may overlap
    m.addConstrs((o[i][j] + o[i][k] + o[i][l] + o[j][k] + o[j][l] + o[k][l]  <= 5) for i in range(nr_tasks) for j in range(nr_tasks) for k in range(nr_tasks) for l in range(nr_tasks) if (len(set([i,j,k,l])) == 4 and task_power_level[i] == 3 and task_power_level[j] == 3 and task_power_level[k] == 3 and task_power_level[l] == 3))


    # Optimize model
    m.optimize()


    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    ms = round(makespan.X)
    m.dispose()
    return ms, tasks_per_proc



# In[ ]:





# In[ ]:


# import time
# import sys
# output_file_lb = "output_load_balancing.txt"
# output_file_TATSND2 = "output_TATSND2.txt"
# output_file_TATSND3 = "output_TATSND3.txt"
# output_file_TATSND4 = "output_TATSND4.txt"

# # # f_lb = open(output_file_lb, 'w')
# # # f_TATSND2 = open(output_file_TATSND2, 'w')
# # # f_TATSND3 = open(output_file_TATSND3, 'w')
# # # f_TATSND4 = open(output_file_TATSND4, 'w')
# # # f_lb.close()
# # # f_TATSND3.close()
# # # f_TATSND2.close()
# # # f_TATSND4.close()

# tpp1 = None
# tpp2 = None
# tpp3 = None
# tpp4 = None

# for j in range(31, 41, 1):
# # for i in [25]:
#     task_graph = create_random_graph(j)
#     start = time.time()
#     ms, tpp1 = create_lb_model(task_graph)
#     tpp_to_power_trace(tpp1, ms, output_file_lb, 1e-1)
#     t_max, t_avg = compute_max_ptrace_temp(output_file_lb, avg_temp=True)
#     f = open(output_file_lb, 'a')
#     f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     f.close()


#     for i in range(len(task_graph.nodes)):
#         task_graph.nodes[i]['power_level'] = 1 if task_graph.nodes[i]['power_cons'] >= 80 else 2
#     start = time.time()
#     ms, tpp2 = create_TATSND2_model(task_graph)
#     tpp_to_power_trace(tpp2, ms, output_file_TATSND2, 1e-1)
#     t_max, t_avg = compute_max_ptrace_temp(output_file_TATSND2, avg_temp=True)
#     f = open(output_file_TATSND2, 'a')
#     f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     f.close()

#     for i in range(len(task_graph.nodes)):
#         if task_graph.nodes[i]['power_cons'] >= 80:
#             task_graph.nodes[i]['power_level'] = 1
#         elif task_graph.nodes[i]['power_cons'] <= 45:
#             task_graph.nodes[i]['power_level'] = 3
#         else:
#             task_graph.nodes[i]['power_level'] = 2

#     start = time.time()
#     ms, tpp3 = create_TATSND3_model(task_graph)
#     tpp_to_power_trace(tpp3, ms, output_file_TATSND3, 1e-1)
#     end = time.time()
#     t_max, t_avg = compute_max_ptrace_temp(output_file_TATSND3, avg_temp=True)
#     f = open(output_file_TATSND3, 'a')
#     f.write(f"Nr tasks {j}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     f.close()

#     for i in range(len(task_graph.nodes)):
#         if task_graph.nodes[i]['power_cons'] >= 80:
#             task_graph.nodes[i]['power_level'] = 1
#         elif task_graph.nodes[i]['power_cons'] <= 30:
#             task_graph.nodes[i]['power_level'] = 4
#         elif task_graph.nodes[i]['power_cons'] <= 45:
#             task_graph.nodes[i]['power_level'] = 3
#         else:
#             task_graph.nodes[i]['power_level'] = 2

#     start = time.time()
#     ms, tpp4 = create_TATSND4_model(task_graph)
#     tpp_to_power_trace(tpp4, ms, output_file_TATSND4, 1e-1)
#     end = time.time()
#     t_max, t_avg = compute_max_ptrace_temp(output_file_TATSND4, avg_temp=True)
#     f = open(output_file_TATSND4, 'a')
#     f.write(f"Nr tasks {j}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     f.close()


# In[ ]:


def create_RestrictedAssignment(graph : nx.DiGraph, restriction: list[list[int]]):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    # task_power_level = [graph.nodes[i]['power_level'] for i in graph.nodes]
    # task_power_level = [2 for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # Restrictions must be adhered to:
    m.addConstrs((sum(x[i][p] for p in restriction[i]) == 1 ) for i in range(nr_tasks))

    # Optimize model
    m.optimize()


    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    ms = round(makespan.X)
    m.dispose()
    return ms, tasks_per_proc



# In[ ]:


# import time
# import sys
# output_file_lb = "output_load_balancing.txt"
# output_file_RA = "output_RA.txt"
# output_file_RA_2 = "output_RA_2.txt"


# # f_lb = open(output_file_lb, 'w')
# # f_TATSND2 = open(output_file_TATSND2, 'w')
# # f_TATSND3 = open(output_file_TATSND3, 'w')
# # f_TATSND4 = open(output_file_TATSND4, 'w')
# # f_lb.close()
# # f_TATSND3.close()
# # f_TATSND2.close()
# # f_TATSND4.close()

# tpp1 = None
# tpp2 = None
# tpp3 = None
# tpp4 = None

# for j in range(1, 26, 1):
# # for i in [25]:
#     task_graph = create_random_graph(j)
#     # start = time.time()
#     # ms, tpp1 = create_lb_model(task_graph)
#     # tpp_to_power_trace(tpp1, ms, output_file_lb, 1e-1)
#     # t_max, t_avg = compute_max_ptrace_temp(output_file_lb, avg_temp=True)
#     # f = open(output_file_lb, 'a')
#     # f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     # f.close()

#     restriction = []
#     for i in range(len(task_graph.nodes)):
#         if task_graph.nodes[i]['power_cons'] >= 70:
#             restriction.append([4,5,6,7])
#         else:
#             restriction.append([0,1,2,3,4,5,6,7])
#     start = time.time()
#     ms, tpp2 = create_RestrictedAssignment(task_graph, restriction)
#     tpp_to_power_trace(tpp2, ms, output_file_RA, 1e-1)
#     t_max, t_avg = compute_max_ptrace_temp(output_file_RA, avg_temp=True)
#     f = open(output_file_RA, 'a')
#     f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     f.close()

#     if j < 19:
#         restriction = []
#         for i in range(len(task_graph.nodes)):
#             if task_graph.nodes[i]['power_cons'] >= 80:
#                 restriction.append([4,6])
#             elif task_graph.nodes[i]['power_cons'] >= 70:
#                 restriction.append([5,7])
#             elif task_graph.nodes[i]['power_cons'] >= 35:
#                 restriction.append([0,1])
#             else:
#                 restriction.append([0,1,2,3,4,5,6,7])
#         start = time.time()
#         ms, tpp2 = create_RestrictedAssignment(task_graph, restriction)
#         tpp_to_power_trace(tpp2, ms, output_file_RA_2, 1e-1)
#         t_max, t_avg = compute_max_ptrace_temp(output_file_RA_2, avg_temp=True)
#         f = open(output_file_RA_2, 'a')
#         f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#         f.close()

    


# In[ ]:


def greedy_lb(graph : nx.DiGraph):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    sorted_tasks = sorted([i for i in range(nr_tasks)], key=lambda x: task_lengths[x], reverse=True)

    load = [0] * nr_processors
    tasks_per_proc = [[] for _ in range(nr_processors)]

    for i in sorted_tasks:
        p = min(range(nr_processors), key = lambda x: load[x])
        tasks_per_proc[p].append((i, load[p], load[p] + task_lengths[i], task_power[i]))
        load[p] += task_lengths[i]
    
    return max(load), tasks_per_proc

def greedy_TATSND2(graph : nx.DiGraph):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    task_power_level = [graph.nodes[i]['power_level'] for i in graph.nodes]

    load = [0] * nr_processors
    tasks_per_proc = [[] for _ in range(nr_processors)]
    sorted_tasks = sorted([i for i in range(nr_tasks)], key=lambda x: task_lengths[x], reverse=True)

    for i in sorted_tasks:
        p = 0
        if task_power_level[i] == 1:
            p = 7
        else:
            p = min(range(nr_processors - 1), key = lambda x: load[x])
        tasks_per_proc[p].append((i, load[p], load[p] + task_lengths[i], task_power[i]))
        load[p] += task_lengths[i]
    
    return max(load), tasks_per_proc

def greedy_TATSND3(graph : nx.DiGraph):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    task_power_level = [graph.nodes[i]['power_level'] for i in graph.nodes]

    load = [0] * nr_processors
    tasks_per_proc = [[] for _ in range(nr_processors)]

    # PE 0 is for power level 1
    # PE 1 is for power level 2
    # PE 3 is for power level 2

    sorted_tasks = sorted([i for i in range(nr_tasks)], key=lambda x: task_lengths[x], reverse=True)
    for i in sorted_tasks:
        p = 0
        if task_power_level[i] == 1:
            p = 7
        elif task_power_level[i] == 2:
            p = min([4,6], key = lambda x: load[x])
        else:
            p = min([0,1,2,3,5], key = lambda x: load[x])
        tasks_per_proc[p].append((i, load[p], load[p] + task_lengths[i], task_power[i]))
        load[p] += task_lengths[i]
    
    return max(load), tasks_per_proc


# In[ ]:


# import time
# import sys
# output_file_lb = "output_load_balancing_approx_lowpower.txt"
# output_file_TATSND2 = "output_TATSND2_approx_lowpower.txt"
# output_file_TATSND3 = "output_TATSND3_approx_lowpower.txt"

# # f_lb = open(output_file_lb, 'w')
# # f_TATSND2 = open(output_file_TATSND2, 'w')
# # f_TATSND3 = open(output_file_TATSND3, 'w')
# # f_lb.write("nr; time; makespan; maxtemp; avgtemp \n")
# # f_TATSND2.write("nr; time; makespan; maxtemp; avgtemp \n")
# # f_TATSND3.write("nr; time; makespan; maxtemp; avgtemp \n")
# # f_lb.close()
# # f_TATSND3.close()
# # f_TATSND2.close()

# tpp1 = None
# tpp2 = None
# tpp3 = None

# for j in range(0, 1001, 10):
#     task_graph = create_random_graph(j)
#     start = time.time()
#     ms, tpp1 = greedy_lb(task_graph)
#     tpp_to_power_trace(tpp1, ms, output_file_lb, 1e-1)
#     t_max, t_avg = compute_max_ptrace_temp(output_file_lb, avg_temp=True)
#     f = open(output_file_lb, 'a')
#     f.write(f"{j}; {time.time() - start}; {ms}; {t_max}; {t_avg} \n")
#     f.close()


#     for i in range(len(task_graph.nodes)):
#         task_graph.nodes[i]['power_level'] = 1 if task_graph.nodes[i]['power_cons'] >= 80 else 2
#     start = time.time()
#     ms, tpp2 = greedy_TATSND2(task_graph)
#     tpp_to_power_trace(tpp2, ms, output_file_TATSND2, 1e-1)
#     t_max, t_avg = compute_max_ptrace_temp(output_file_TATSND2, avg_temp=True)
#     f = open(output_file_TATSND2, 'a')
#     f.write(f"{j}; {time.time() - start}; {ms}; {t_max}; {t_avg} \n")
#     f.close()

#     for i in range(len(task_graph.nodes)):
#         if task_graph.nodes[i]['power_cons'] >= 80:
#             task_graph.nodes[i]['power_level'] = 1
#         elif task_graph.nodes[i]['power_cons'] <= 45:
#             task_graph.nodes[i]['power_level'] = 3
#         else:
#             task_graph.nodes[i]['power_level'] = 2

#     start = time.time()
#     ms, tpp3 = greedy_TATSND3(task_graph)
#     tpp_to_power_trace(tpp3, ms, output_file_TATSND3, 1e-1)
#     end = time.time()
#     t_max, t_avg = compute_max_ptrace_temp(output_file_TATSND3, avg_temp=True)
#     f = open(output_file_TATSND3, 'a')
#     f.write(f"{j}; {time.time() - start}; {ms}; {t_max}; {t_avg} \n")
#     f.close()



# In[ ]:


def twoApproxRestrictedAssignment(graph : nx.DiGraph, restriction: list[list[int]]):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    def LP(lam):
        # Create a new model
        m = gp.Model("LP")

        # Create variables
        x = [[m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)]
        
        for i in range(nr_tasks):
            m.addConstr(sum(x[i][j] for j in restriction[i]) == 1)
        
        for j in range(nr_processors):
            m.addConstr(sum(task_lengths[i] * x[i][j] for i in range(nr_tasks)) <= lam)
        
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            return False

        print([[x[i][j].X for j in range(nr_processors)] for i in range(nr_tasks)])
        return [[x[i][j].X for j in range(nr_processors)] for i in range(nr_tasks)]

    lb = max(task_lengths)
    ub = sum(task_lengths)
    last_feasible = LP(ub)
    while ub - lb > 1:
        mid = (lb + ub) // 2
        lp = LP(mid)
        if lp == False:
            lb = mid
        else:
            ub = mid
            last_feasible = lp
    lamb = ub
    load = [0] * nr_processors
    tasks_per_proc = [[] for _ in range(nr_processors)]
    x = last_feasible

    H = nx.Graph()

    fractional_tasks = set()


    for i in range(nr_tasks):
        for j in range(nr_processors):
            if x[i][j] == 1:
                tasks_per_proc[j].append((i, load[j], load[j] + task_lengths[i], task_power[i]))
                load[j] += task_lengths[i]
            elif x[i][j] > 0:
                fractional_tasks.add(i)
                H.add_edge(i, str(j))

    print(H.edges)
    matching = nx.bipartite.maximum_matching(H, fractional_tasks)

    for i in fractional_tasks:
        j = int(matching[i])
        tasks_per_proc[j].append((i, load[j], load[j] + task_lengths[i], task_power[i]))
        load[j] += task_lengths[i]
    
    return max(load), tasks_per_proc
    

            




    




# In[ ]:


# import time
# import sys
# output_file_RA = "output_RA_approx.txt"
# output_file_RA_2 = "output_RA_2_approx.txt"



# f_1 = open(output_file_RA, 'w')
# f_2 = open(output_file_RA_2, 'w')
# f_1.close()
# f_2.close()


# tpp1 = None
# tpp2 = None
# tpp3 = None
# tpp4 = None

# for j in range(20, 1001, 20):
# # for j in [1]:
#     task_graph = create_random_graph(j)
#     # start = time.time()
#     # ms, tpp1 = create_lb_model(task_graph)
#     # tpp_to_power_trace(tpp1, ms, output_file_lb, 1e-1)
#     # t_max, t_avg = compute_max_ptrace_temp(output_file_lb, avg_temp=True)
#     # f = open(output_file_lb, 'a')
#     # f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     # f.close()89

#     restriction = []
#     for i in range(len(task_graph.nodes)):
#         if task_graph.nodes[i]['power_cons'] >= 70:
#             restriction.append([4,5,6,7])
#         else:
#             restriction.append([0,1,2,3,4,5,6,7])
#     start = time.time()
#     ms, tpp1 = twoApproxRestrictedAssignment(task_graph, restriction)
#     tpp_to_power_trace(tpp1, ms, output_file_RA, 1e-1)
#     t_max, t_avg = compute_max_ptrace_temp(output_file_RA, avg_temp=True)
#     f = open(output_file_RA, 'a')
#     f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     f.close()

#     restriction = []
#     for i in range(len(task_graph.nodes)):
#         if task_graph.nodes[i]['power_cons'] >= 80:
#             restriction.append([4,6])
#         elif task_graph.nodes[i]['power_cons'] >= 70:
#             restriction.append([5,7])
#         elif task_graph.nodes[i]['power_cons'] >= 35:
#             restriction.append([0,1])
#         else:
#             restriction.append([0,1,2,3,4,5,6,7])
#     start = time.time()
#     ms, tpp2 = twoApproxRestrictedAssignment(task_graph, restriction)
#     tpp_to_power_trace(tpp2, ms, output_file_RA_2, 1e-1)
#     t_max, t_avg = compute_max_ptrace_temp(output_file_RA_2, avg_temp=True)
#     f = open(output_file_RA_2, 'a')
#     f.write(f"Nr tasks {j}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max}, Avg Temp {t_avg} \n")
#     f.close()

    


# In[ ]:


def create_RA_BC(graph : nx.DiGraph, restriction: list[list[int]], block: list[list[int]], MIPGap = False):
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    # task_lengths = [1 for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    # task_power_level = [graph.nodes[i]['power_level'] for i in graph.nodes]
    # task_power_level = [2 for i in graph.nodes]
    
    # Create a new model
    m = gp.Model("Matching")

    # Create variables
    makespan = m.addVar(name="makespan")
    start_time = [m.addVar(name=f"s({i})") for i in range(nr_tasks)]
    x = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = f"x[{i},{p}]") for p in range(nr_processors)] for i in range(nr_tasks)] # x[i,p] == 1 iff task i scheduled on PE p
    b = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # b[i,j] == 1 iff task i finishes before task j starts
    o = [[m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"o[{i},{j}]") for j in range(nr_tasks)] for i in range(nr_tasks)] # o[i,j] == 1 iff task i overlaps with task j

    # Set objective: minimize makespan
    m.setObjective(1.0 * makespan, GRB.MINIMIZE)

    # Define makespan
    m.addConstrs((start_time[i] + task_lengths[i] <= makespan for i in range(nr_tasks)), name="ms_def")

    # Task is planned on exactly one processor:
    m.addConstrs((sum(x[i][p] for p in range(nr_processors)) == 1 ) for i in range(nr_tasks))

    # Define b[i,j] and o[i,j]
    M = sum(task_lengths)  # Large constant: Worst case execution time, i.e. executing all tasks on the slowest core.
    m.addConstrs((start_time[i] + task_lengths[i] <= start_time[j] + M * (1 - b[i][j]) + M * o[i][j]) for i in range(nr_tasks) for j in range(nr_tasks))
    m.addConstrs((start_time[i] + M * (b[i][j] + o[i][j]) >= start_time[j] + task_lengths[j]) for i in range(nr_tasks) for j in range(nr_tasks))

    # If  two task on same processor, then one of them comes before the other
    m.addConstrs((x[i][p] + x[j][p] + o[i][j] <= 2) for i in range(nr_tasks) for j in range(nr_tasks) for p in range(nr_processors) if i != j)

    # Restrictions must be adhered to:
    m.addConstrs((sum(x[i][p] for p in restriction[i]) == 1 ) for i in range(nr_tasks))

    # Blocking tasks do not overlap
    m.addConstrs((x[u][p] + o[v][u] <= 1) for u in range(nr_tasks) for v in range(nr_tasks) for p in block[v])
    m.setParam('MIPGap', 0.25)
    m.setParam('TimeLimit', 1200)
    # Optimize model
    m.optimize()



    tasks_per_proc = [[] for _ in range(nr_processors)]

    # Print the values of all variables
    for i in range(nr_tasks):
        for p in range(nr_processors):
            if x[i][p].X == 1.0:
                print("Task", i, "is mapped on PE", p, ". start time:", start_time[i].X, ", finish time:", start_time[i].X + task_lengths[i])
                tasks_per_proc[p].append((i, start_time[i].X, start_time[i].X + task_lengths[i], task_power[i]))
    
    ms = round(makespan.X)
    gap = float(m.MIPGap)
    m.dispose()
    
    
    if MIPGap:
        return ms, tasks_per_proc, gap
    return ms, tasks_per_proc



# In[ ]:


def greedy_RA_BC(graph : nx.DiGraph, restriction: list[list[int]], block: list[list[int]]):
    tpp = [[] for _ in range(proc_total)]
    nr_tasks = len(graph.nodes)
    nr_processors = proc_total
    task_lengths = [graph.nodes[i]['weight'] for i in graph.nodes]
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    
    makespan = 0

    def power_consumption_of_core(time, core):
        for t, st, ft, pow in tpp[core]:
            if round(st) <= time < round(ft):
                return pow
        return 0
    
    def is_blocked(time, core):
        for t, st, ft, pow in tpp[core]:
            if round(st) <= time < round(ft) and pow == 0:
                return True
        return False
    
    def is_idle(time, core):
        return power_consumption_of_core(time, core) == 0 and not(is_blocked(time, core))


    for i in range(nr_tasks):
        opt_t = None
        opt_p = None
        for p in restriction[i]:

            possible_time_stamps = []
            for t in range(makespan):
                if all([is_idle(time, p) for time in range(t, t + task_lengths[i])]) and all([power_consumption_of_core(time, proc) == 0 for time in range(t, t + task_lengths[i]) for proc in block[i]]):
                    possible_time_stamps.append(t)
            for t in possible_time_stamps:
                for t_run in range(t, t + task_lengths[i]):
                    if t_run not in possible_time_stamps and t_run <= makespan:
                        break
                else:
                    if opt_t == None or t < opt_t:
                        opt_t = t
                        opt_p = p
            else:
                if opt_t == None:
                    opt_t = makespan
                    opt_p = p
        
        tpp[opt_p].append((i, opt_t, opt_t + task_lengths[i], task_power[i]))
        for p in block[i]:
            tpp[p].append((i, opt_t, opt_t + task_lengths[i], 0))
        makespan = max(makespan, opt_t + task_lengths[i])
    
    return makespan, tpp
    


# In[ ]:


def restriction_blocking_set(stack: list[int], T_crit: float, T_amb: float):
    S_org = [s for s in stack]
    PowerLimits = []
    while len(stack) != 0:
        m = gp.Model("LP")
        x = m.addVar(lb=0, name="MaxPower")
        m.setObjective(x, GRB.MAXIMIZE)
        m.addConstrs((sum(coefficients[i][j] * x for j in stack) <= (T_crit - T_amb)) for i in stack)
        m.optimize()

        max_power = x.X
        PowerLimits.append((max_power, stack.copy(), [p for p in S_org if p not in stack]))
        stack.remove(min(stack))
    return PowerLimits


# In[ ]:


def MatrixModel_to_RA_BC_no_hor(graph: nx.DiGraph, T_crit: float, T_amb: float):
    nr_tasks = len(graph.nodes)
    S = [[] for _ in range(proc_cores)]
    for i in range(proc_cores):
        S[i] = [i + proc_cores * x for x in range(proc_layers)]
    res = []
    block = []
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]

    PowerLimits = [restriction_blocking_set(s, T_crit, T_amb) for s in S]
    
    for task in range(nr_tasks):
        restriction = set()
        blocking = set()
        found = False
        for PL in PowerLimits:
            for (x, R, B) in PL:
                if task_power[task] <= x:
                    restriction = restriction.union(R)
                    blocking = blocking.union(B)
                    found = True
                    break
        if found == False:
            raise Exception("Unsolvable instance!")
        res.append(list(restriction))
        block.append(list(blocking))
    
    return res, block


# In[ ]:


def MatrixModel_to_RA_BC(graph : nx.DiGraph, T_crit: float, T_amb: float):
    P = list(range(proc_total))
    P_org = list(range(proc_total))
    PowerLimits = []
    nr_tasks = len(graph.nodes)
    res = []
    block = []
    task_power = [graph.nodes[i]['power_cons'] for i in graph.nodes]
    while len(P) != 0:
        m = gp.Model("LP")
        x = m.addVar(lb=0, name="MaxPower")
        m.setObjective(x, GRB.MAXIMIZE)
        m.addConstrs((sum(coefficients[i][j] * x for j in P) <= (T_crit - T_amb)) for i in P)
        m.optimize()
        max_power = x.X
        PowerLimits.append((max_power, P.copy(), [p for p in P_org if p not in P]))
        P.remove(max(P, key=lambda x: coefficients[x][x]))
    
    for task in range(nr_tasks):
        for (x, R, B) in PowerLimits:
            if task_power[task] <= x:
                res.append(R)
                block.append(B)
                break
        else:
            raise Exception("Instance unsolvable!")

    return res, block


# In[ ]:


task_graph = create_random_graph(10)
r, b = MatrixModel_to_RA_BC(task_graph, 400, 318.15)

ms, tpp = greedy_RA_BC(task_graph, r, b)


# In[ ]:


# tpp_to_power_trace(tpp, ms,  "bruh.txt", 1e-1)


# In[ ]:


import time
import sys
output_file_mm_normal = "output_matrixmodel_model.txt"
output_file_mm_greedy = "output_greedymatrixmodel_model.txt"

output_file_rabc_nohor_normal = "output_rabc_nohor_model_approx.txt"
output_file_rabc_nohor_greedy = "output_greedyrabc_nohor_model.txt"

output_file_rabc_normal = "output_file_rabc_normal_approx.txt"
output_file_rabc_greedy = "output_file_rabc_greedy.txt"

# files = [output_file_mm_normal, output_file_mm_greedy, output_file_rabc_nohor_normal, output_file_rabc_nohor_greedy, output_file_rabc_normal, output_file_rabc_greedy]
# # Create and empty files
# for file in files:
#     f = open(file, "w")
#     f.close()

ms1, tpp1, ms2, tpp2 = None,None,None,None

found_no_hor = False
found = False

for i in range(340, 501, 20):
# for i in [13]:
    task_graph = create_random_graph(i)
    # start = time.time()
    # ms, tpp = create_model(task_graph, 400)
    # tpp_to_power_trace(tpp, ms, output_file_mm_normal, 1e-1)
    # t_max = compute_max_ptrace_temp(output_file_mm_normal)
    # f = open(output_file_mm_normal, 'a')
    # f.write(f"Nr tasks {i}, Time {time.time() - start}, Makespan {ms}, Max Temp {t_max} \n")
    # f.close()

    # start = time.time()
    # ms, tpp = greedy_scheduling(task_graph, 400)
    # tpp_to_power_trace(tpp, ms, output_file_mm_greedy, 1e-1)
    # end = time.time()
    # t_max, top1p = compute_max_ptrace_temp(output_file_mm_greedy, True)
    # f = open(output_file_mm_greedy, 'a')
    # f.write(f"Nr tasks {i}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Top 1 percent highs {top1p} \n")
    # f.close()

    
    r, b = MatrixModel_to_RA_BC_no_hor(task_graph, 400, 318.15)

    # start = time.time()
    # ms, tpp, gap = create_RA_BC(task_graph, r, b, True)
    # tpp_to_power_trace(tpp, ms, output_file_rabc_nohor_normal, 1e-1)
    # end = time.time()
    # t_max, top1p = compute_max_ptrace_temp(output_file_rabc_nohor_normal, True)
    # f = open(output_file_rabc_nohor_normal, 'a')
    # f.write(f"Nr tasks {i}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Top 1 percent highs {top1p}, GAP {gap * 100}% \n")
    # f.close()


    start = time.time()
    ms, tpp = greedy_RA_BC(task_graph, r, b)
    tpp_to_power_trace(tpp, ms, output_file_rabc_nohor_greedy, 1e-1)
    end = time.time()
    t_max, top1p = compute_max_ptrace_temp(output_file_rabc_nohor_greedy, True)
    f = open(output_file_rabc_nohor_greedy, 'a')
    f.write(f"Nr tasks {i}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Top 1 percent highs {top1p} \n")
    f.close()

    r, b = MatrixModel_to_RA_BC(task_graph, 400, 318.15)
        
    #     start = time.time()
    #     ms, tpp, gap = create_RA_BC(task_graph, r, b, True)
    #     tpp_to_power_trace(tpp, ms, output_file_rabc_normal, 1e-1)
    #     end = time.time()
    #     t_max, top1p = compute_max_ptrace_temp(output_file_rabc_normal, True)
    #     f = open(output_file_rabc_normal, 'a')
    #     f.write(f"Nr tasks {i}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Top 1 percent highs {top1p}, GAP {gap * 100}% \n")
    #     f.close()


    start = time.time()
    ms, tpp = greedy_RA_BC(task_graph, r, b)
    tpp_to_power_trace(tpp, ms, output_file_rabc_greedy, 1e-1)
    end = time.time()
    t_max, top1p = compute_max_ptrace_temp(output_file_rabc_greedy, True)
    f = open(output_file_rabc_greedy, 'a')
    f.write(f"Nr tasks {i}, Time {end - start}, Makespan {ms}, Max Temp {t_max}, Top 1 percent highs {top1p} \n")
    f.close()


# In[ ]:





# In[ ]:





# In[ ]:


# file = open("../HotspotConfiguration/2-layer-chip/output_file_rabc_normal.txt", "r")
# maxtemp = 0
# for line in file.readlines()[1:]:
#     for p in range(proc_total):
#         maxtemp = max(maxtemp, temperature(p, list(map(float, line.strip().split()))))

# maxtemp

