import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('output_file_rabc_greedy.csv')

# Plot the data
plt.figure(figsize=(10, 6))

# Plot each column
plt.plot(df['Nrtasks'], df['Time'], label='Time', marker='o')
plt.plot(df['Nrtasks'], df['Makespan'], label='Makespan', marker='o')
plt.plot(df['Nrtasks'], df['MaxTemp'], label='Max Temp', marker='o')
# plt.plot(df['Nrtasks'], df['Top1percenthighs'], label='Top 1 percent highs', marker='o')

# Add titles and labels with larger font size
plt.title('Task Data Over Time', fontsize=20)
plt.xlabel('Nr tasks', fontsize=15)
plt.ylabel('Values', fontsize=15)
plt.legend(fontsize=12)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("bruh.pdf")