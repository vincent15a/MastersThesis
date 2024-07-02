import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('output_file_rabc_greedy.csv')
dfMILP = pd.read_csv('output_normal_model.csv')
dfMILPRelaxed = pd.read_csv('output_relaxed_model.csv')
dfGreedy = pd.read_csv('output_greedy_model.csv')
dfSplitting = pd.read_csv('output_splitting_model.csv')
dfMILPApprox = pd.read_csv('output_normal_model_approx.csv')

# Define a function to create individual plots
def create_plot(pairs, xlabel, ylabel, title: str):
    size = 50
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ["MILP", "MILP Relaxed", "Greedy", "Splitting", "MILP Approximation"]
    c_index = 0
    for x, y in pairs:
        plt.plot(x.head(size), y.head(size), '--', color=colors[c_index], label = labels[c_index])
        c_index += 1
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}TATS-ND-FP.pdf")

# Create individual plots for each column

create_plot([(dfMILP['Nrtasks'], dfMILP['Time']), (dfMILPRelaxed['Nrtasks'], dfMILPRelaxed['Time']), (dfGreedy['Nrtasks'], dfGreedy['Time']), (dfSplitting['Nrtasks'], dfSplitting['Time']), (dfMILPApprox['Nrtasks'], dfMILPApprox['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
create_plot([(dfMILP['Nrtasks'], dfMILP['Makespan']), (dfMILPRelaxed['Nrtasks'], dfMILPRelaxed['Makespan']), (dfGreedy['Nrtasks'], dfGreedy['Makespan']), (dfSplitting['Nrtasks'], dfSplitting['Makespan']), (dfMILPApprox['Nrtasks'], dfMILPApprox['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
create_plot([(dfMILP['Nrtasks'], dfMILP['MaxTemp']), (dfMILPRelaxed['Nrtasks'], dfMILPRelaxed['MaxTemp']), (dfGreedy['Nrtasks'], dfGreedy['MaxTemp']), (dfSplitting['Nrtasks'], dfSplitting['MaxTemp']), (dfMILPApprox['Nrtasks'], dfMILPApprox['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')



# Read the CSV file
dfLB = pd.read_csv('output_load_balancing.csv')
dfTATSND2 = pd.read_csv('output_TATSND2.csv')
dfTATSND3 = pd.read_csv('output_TATSND3.csv')
dfTATSND4 = pd.read_csv('output_TATSND4.csv')


# Define a function to create individual plots
def create_plot2(pairs, xlabel, ylabel, title: str):
    size = 34
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ["MM", "TATS-ND-2", "TATS-ND-3", "TATS-ND-4"]
    c_index = 0
    for x, y in pairs:
        plt.plot(x.head(size), y.head(size), '--', color=colors[c_index], label = labels[c_index])
        c_index += 1
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}TATSNDk.pdf")

# Create individual plots for each column

create_plot2([(dfLB['Nrtasks'], dfLB['Time']), (dfTATSND2['Nrtasks'], dfTATSND2['Time']), (dfTATSND3['Nrtasks'], dfTATSND3['Time']), (dfTATSND4['Nrtasks'], dfTATSND4['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
create_plot2([(dfLB['Nrtasks'], dfLB['Makespan']), (dfTATSND2['Nrtasks'], dfTATSND2['Makespan']), (dfTATSND3['Nrtasks'], dfTATSND3['Makespan']), (dfTATSND4['Nrtasks'], dfTATSND4['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
create_plot2([(dfLB['Nrtasks'], dfLB['MaxTemp']), (dfTATSND2['Nrtasks'], dfTATSND2['MaxTemp']), (dfTATSND3['Nrtasks'], dfTATSND3['MaxTemp']), (dfTATSND4['Nrtasks'], dfTATSND4['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')
create_plot2([(dfLB['Nrtasks'], dfLB['AvgTemp']), (dfTATSND2['Nrtasks'], dfTATSND2['AvgTemp']), (dfTATSND3['Nrtasks'], dfTATSND3['AvgTemp']), (dfTATSND4['Nrtasks'], dfTATSND4['AvgTemp'])], 'Number of tasks', 'Average Temperature (K)', 'Average Temperature')



# Read the CSV file
dfLB = pd.read_csv('output_load_balancing_highpower.csv')
dfTATSND2 = pd.read_csv('output_TATSND2_highpower.csv')
dfTATSND3 = pd.read_csv('output_TATSND3_highpower.csv')


# Define a function to create individual plots
def create_plot3(pairs, xlabel, ylabel, title: str):
    size = 30
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ["MM", "TATS-ND-2", "TATS-ND-3", "TATS-ND-4"]
    c_index = 0
    for x, y in pairs:
        plt.plot(x.head(size), y.head(size), '--', color=colors[c_index], label = labels[c_index])
        c_index += 1
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}TATSNDk_highpower.pdf")

# Create individual plots for each column

create_plot3([(dfLB['Nrtasks'], dfLB['Time']), (dfTATSND2['Nrtasks'], dfTATSND2['Time']), (dfTATSND3['Nrtasks'], dfTATSND3['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
create_plot3([(dfLB['Nrtasks'], dfLB['Makespan']), (dfTATSND2['Nrtasks'], dfTATSND2['Makespan']), (dfTATSND3['Nrtasks'], dfTATSND3['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
create_plot3([(dfLB['Nrtasks'], dfLB['MaxTemp']), (dfTATSND2['Nrtasks'], dfTATSND2['MaxTemp']), (dfTATSND3['Nrtasks'], dfTATSND3['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')
create_plot3([(dfLB['Nrtasks'], dfLB['AvgTemp']), (dfTATSND2['Nrtasks'], dfTATSND2['AvgTemp']), (dfTATSND3['Nrtasks'], dfTATSND3['AvgTemp'])], 'Number of tasks', 'Average Temperature (K)', 'Average Temperature')



# Read the CSV file
dfLB = pd.read_csv('output_load_balancing_approx.txt')
dfTATSND2 = pd.read_csv('output_TATSND2_approx.txt')
dfTATSND3 = pd.read_csv('output_TATSND3_approx.txt')


# Define a function to create individual plots
def create_plot4(pairs, xlabel, ylabel, title: str):
    size = 30
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ["MM", "TATS-ND-2", "TATS-ND-3", "TATS-ND-4"]
    c_index = 0
    for x, y in pairs:
        plt.plot(x, y, '--', color=colors[c_index], label = labels[c_index])
        c_index += 1
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}TATSNDk_approx.pdf")

# Create individual plots for each column

create_plot4([(dfLB['Nrtasks'], dfLB['Time']), (dfTATSND2['Nrtasks'], dfTATSND2['Time']), (dfTATSND3['Nrtasks'], dfTATSND3['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
create_plot4([(dfLB['Nrtasks'], dfLB['Makespan']), (dfTATSND2['Nrtasks'], dfTATSND2['Makespan']), (dfTATSND3['Nrtasks'], dfTATSND3['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
create_plot4([(dfLB['Nrtasks'], dfLB['MaxTemp']), (dfTATSND2['Nrtasks'], dfTATSND2['MaxTemp']), (dfTATSND3['Nrtasks'], dfTATSND3['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')
create_plot4([(dfLB['Nrtasks'], dfLB['Avgtemp']), (dfTATSND2['Nrtasks'], dfTATSND2['Avgtemp']), (dfTATSND3['Nrtasks'], dfTATSND3['Avgtemp'])], 'Number of tasks', 'Average Temperature (K)', 'Average Temperature')


# Read the CSV file
dfLB = pd.read_csv('output_load_balancing_approx_lowpower.txt')
dfTATSND2 = pd.read_csv('output_TATSND2_approx_lowpower.txt')
dfTATSND3 = pd.read_csv('output_TATSND3_approx_lowpower.txt')


# Define a function to create individual plots
def create_plot5(pairs, xlabel, ylabel, title: str):
    size = 30
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ["MM", "TATS-ND-2", "TATS-ND-3", "TATS-ND-4"]
    c_index = 0
    for x, y in pairs:
        plt.plot(x, y, '--', color=colors[c_index], label = labels[c_index])
        c_index += 1
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}TATSNDk_approx_lowpower.pdf")

# Create individual plots for each column

create_plot5([(dfLB['Nrtasks'], dfLB['Time']), (dfTATSND2['Nrtasks'], dfTATSND2['Time']), (dfTATSND3['Nrtasks'], dfTATSND3['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
create_plot5([(dfLB['Nrtasks'], dfLB['Makespan']), (dfTATSND2['Nrtasks'], dfTATSND2['Makespan']), (dfTATSND3['Nrtasks'], dfTATSND3['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
create_plot5([(dfLB['Nrtasks'], dfLB['MaxTemp']), (dfTATSND2['Nrtasks'], dfTATSND2['MaxTemp']), (dfTATSND3['Nrtasks'], dfTATSND3['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')
create_plot5([(dfLB['Nrtasks'], dfLB['Avgtemp']), (dfTATSND2['Nrtasks'], dfTATSND2['Avgtemp']), (dfTATSND3['Nrtasks'], dfTATSND3['Avgtemp'])], 'Number of tasks', 'Average Temperature (K)', 'Average Temperature')

dfMILP = pd.read_csv('output_normal_model.csv')
dfGreedy = pd.read_csv('output_greedy_model.csv')
dfMILPApprox = pd.read_csv('output_normal_model_approx.csv')

dfMILP_RABC = pd.read_csv('output_file_rabc_normal.csv')
dfGreedy_RABC = pd.read_csv('output_file_rabc_greedy.csv')
dfMILPApprox_RABC = pd.read_csv('output_file_rabc_normal_approx.csv')

dfMILP_RABC_nohor = pd.read_csv('output_rabc_nohor_model.csv')
dfGreedy_RABC_nohor = pd.read_csv('output_greedyrabc_nohor_model.csv')
dfMILPApprox_RABC_nohor = pd.read_csv('output_rabc_nohor_model_approx.csv')


# Define a function to create individual plots
def create_plot6(pairs, xlabel, ylabel, title: str):
    size = 50
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm', 'y', 'c']
    labels = ["Matrix Model", "RA-BC", "RA-BC No Horizontal Dissapation", "Matrix Model Approximation", "RA-BC Approximation", "RA-BC No Horizontal Dissapation Approximation"]
    c_index = 0
    for x, y in pairs:
        plt.plot(x, y, '--', color=colors[c_index], label = labels[c_index])
        c_index += 1
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}RABC.pdf")

# Create individual plots for each column

create_plot6([(dfMILP['Nrtasks'], dfMILP['Time']), (dfMILP_RABC['Nrtasks'], dfMILP_RABC['Time']), (dfMILP_RABC_nohor['Nrtasks'], dfMILP_RABC_nohor['Time']), (dfMILPApprox['Nrtasks'], dfMILPApprox['Time']), (dfMILPApprox_RABC['Nrtasks'], dfMILPApprox_RABC['Time']), (dfMILPApprox_RABC_nohor['Nrtasks'], dfMILPApprox_RABC_nohor['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
create_plot6([(dfMILP['Nrtasks'], dfMILP['Makespan']), (dfMILP_RABC['Nrtasks'], dfMILP_RABC['Makespan']), (dfMILP_RABC_nohor['Nrtasks'], dfMILP_RABC_nohor['Makespan']), (dfMILPApprox['Nrtasks'], dfMILPApprox['Makespan']), (dfMILPApprox_RABC['Nrtasks'], dfMILPApprox_RABC['Makespan']), (dfMILPApprox_RABC_nohor['Nrtasks'], dfMILPApprox_RABC_nohor['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
create_plot6([(dfMILP['Nrtasks'], dfMILP['MaxTemp']), (dfMILP_RABC['Nrtasks'], dfMILP_RABC['MaxTemp']), (dfMILP_RABC_nohor['Nrtasks'], dfMILP_RABC_nohor['MaxTemp']), (dfMILPApprox['Nrtasks'], dfMILPApprox['MaxTemp']), (dfMILPApprox_RABC['Nrtasks'], dfMILPApprox_RABC['MaxTemp']), (dfMILPApprox_RABC_nohor['Nrtasks'], dfMILPApprox_RABC_nohor['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')


# # Define a function to create individual plots
# def create_plot7(pairs, xlabel, ylabel, title: str):
#     size = 50
#     plt.figure(figsize=(10, 6))
#     colors = ['b', 'g', 'r', 'm', 'y']
#     labels = ["Matrix Model", "RA-BC", "RA-BC No Horizontal Dissapation"]
#     c_index = 0
#     for x, y in pairs:
#         plt.plot(x, y, '--', color=colors[c_index], label = labels[c_index])
#         c_index += 1
#     plt.title(title, fontsize=20)
#     plt.xlabel(xlabel, fontsize=15)
#     plt.ylabel(ylabel, fontsize=15)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     plt.legend()
#     plt.savefig(f"{title.replace(' ', '')}RABC_Approx.pdf")

# # Create individual plots for each column

# create_plot7([(dfMILPApprox['Nrtasks'], dfMILPApprox['Time']), (dfMILPApprox_RABC['Nrtasks'], dfMILPApprox_RABC['Time']), (dfMILPApprox_RABC_nohor['Nrtasks'], dfMILPApprox_RABC_nohor['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
# create_plot7([(dfMILPApprox['Nrtasks'], dfMILPApprox['Makespan']), (dfMILPApprox_RABC['Nrtasks'], dfMILPApprox_RABC['Makespan']), (dfMILPApprox_RABC_nohor['Nrtasks'], dfMILPApprox_RABC_nohor['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
# create_plot7([(dfMILPApprox['Nrtasks'], dfMILPApprox['MaxTemp']), (dfMILPApprox_RABC['Nrtasks'], dfMILPApprox_RABC['MaxTemp']), (dfMILPApprox_RABC_nohor['Nrtasks'], dfMILPApprox_RABC_nohor['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')



# Define a function to create individual plots
def create_plot8(pairs, xlabel, ylabel, title: str):
    size = 50
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ["Matrix Model", "RA-BC", "RA-BC No Horizontal Dissapation"]
    c_index = 0
    for x, y in pairs:
        plt.plot(x, y, '--', color=colors[c_index], label = labels[c_index])
        c_index += 1
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}RABC_Greedy.pdf")

# Create individual plots for each column

create_plot8([(dfGreedy['Nrtasks'], dfGreedy['Time']), (dfGreedy_RABC['Nrtasks'], dfGreedy_RABC['Time']), (dfGreedy_RABC_nohor['Nrtasks'], dfGreedy_RABC_nohor['Time'])], 'Number of tasks', 'Time (s)', 'Running Time')
create_plot8([(dfGreedy['Nrtasks'], dfGreedy['Makespan']), (dfGreedy_RABC['Nrtasks'], dfGreedy_RABC['Makespan']), (dfGreedy_RABC_nohor['Nrtasks'], dfGreedy_RABC_nohor['Makespan'])], 'Number of tasks', 'Makespan', 'Makespan')
create_plot8([(dfGreedy['Nrtasks'], dfGreedy['MaxTemp']), (dfGreedy_RABC['Nrtasks'], dfGreedy_RABC['MaxTemp']), (dfGreedy_RABC_nohor['Nrtasks'], dfGreedy_RABC_nohor['MaxTemp'])], 'Number of tasks', 'Maximum Temperature (K)', 'Maximum Temperature')
