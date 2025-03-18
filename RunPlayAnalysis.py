# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:04:33 2024

@author: NAWri
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib as mpl

seasons = [2021,2022,2023]
pbpdata = nfl.import_pbp_data(seasons)
runplays = pbpdata[pbpdata['play_type']=='run']
run_filtered = runplays[(runplays['score_differential'].abs() <= 16)]
columns = ['run_location','run_gap','yardline_100','game_seconds_remaining',
           'quarter_seconds_remaining','half_seconds_remaining','down',
           'goal_to_go','ydstogo','yards_gained','play_type','shotgun',
           'no_huddle','no_score_prob','opp_fg_prob','opp_td_prob','fg_prob',
           'td_prob','wp','def_wp','penalty','drive_play_count','drive_time_of_possession',
           'div_game','roof','surface','temp','defenders_in_box','offense_personnel',
           'offense_formation','epa']
runplaysfiltered = run_filtered[columns]

# change grass to be binary grass or turf
runplaysfiltered = runplaysfiltered.rename(columns ={
    'surface': 'grass'
})

runplaysfiltered['grass'] = runplaysfiltered['grass'].apply(lambda x: 1 if x in ['grass'] else 0)

# Modify the 'roof' variable to be binary with outdoors and open being 1, else being 0
runplaysfiltered['roof'] = runplaysfiltered['roof'].apply(lambda x: 1 if x in ['outdoors', 'open'] else 0)
#-----------------------------------------------------------------------------------------
# Generating Personnel Groupings
# Function to extract the number of RBs, TEs, and WRs from the 'offense_personnel' string
import re

def extract_personnel_counts(personnel):
    positions = {'RB': 0, 'TE': 0, 'WR': 0}
    if pd.isna(personnel):
        return pd.Series(positions)

    # Use regex to find all occurrences of "number + position"
    matches = re.findall(r'(\d+)\s*(RB|TE|WR)', personnel)
    
    for count, position in matches:
        positions[position] = int(count)

    return pd.Series(positions)

# Apply function to create new columns
runplaysfiltered[['RB_count', 'TE_count', 'WR_count']] = runplaysfiltered['offense_personnel'].apply(extract_personnel_counts)
#-----------------------------------------------------------------------------------------
shotgunruns = runplaysfiltered[runplaysfiltered['shotgun']==1.0]
undercenter = runplaysfiltered[runplaysfiltered['shotgun']==0.0]
# changing invalid empty sets to shotgun
# Create a new DataFrame to store the updated rows
updatedshotgun = shotgunruns.copy()

# Iterate through the rows where the formation is 'EMPTY'
for idx, row in updatedshotgun[updatedshotgun['offense_formation'] == 'EMPTY'].iterrows():
    # Check if there is more than 1 running back or tight end
    if row['RB_count'] > 1 or row['TE_count'] > 1:
        # Change formation to 'SHOTGUN'
        updatedshotgun.at[idx, 'offense_formation'] = 'SHOTGUN'

import matplotlib.pyplot as plt
import seaborn as sns

# Filter down to necessary columns for grouping (assuming you've already done this with updatedshotgun, empty, and pistol)
# Group by offense formation, RB_count, TE_count to calculate YPC and EPA
personnel_stats = updatedshotgun.groupby(['offense_formation', 'RB_count', 'TE_count']).agg(
    YPC=('yards_gained', 'mean'),
    EPA=('epa', 'mean')
).reset_index()

# Get unique formations
formations = personnel_stats['offense_formation'].unique()

# Create a clustered bar graph for each formation
for formation in formations:
    data = personnel_stats[personnel_stats['offense_formation'] == formation]
    
    # Sort by TE count first, then RB count for better visualization
    data = data.sort_values(by=['TE_count', 'RB_count'])
    
    # Create a new 'personnel_combo' column for labeling purposes
    data['personnel_combo'] = data.apply(lambda row: f"{row['TE_count']} TE, {row['RB_count']} RB", axis=1)
    
    # Create the figure and axis for plotting
    plt.figure(figsize=(10, 6))
    
    # Plot YPC (blue) and EPA (orange)
    sns.barplot(data=data, x='personnel_combo', y='YPC', color='blue', label='YPC')
    sns.barplot(data=data, x='personnel_combo', y='EPA', color='orange', label='EPA', alpha=0.6)

    # Set labels and title
    plt.xlabel('Personnel Packages (TE count → RB count)')
    plt.ylabel('Metric Value')
    plt.title(f'Run Performance in {formation} Formation')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Show legend
    plt.legend(title="Metrics")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

"""
# Filter for 11 personnel (1 RB, 1 TE)
run_11p = runplaysfiltered[(runplaysfiltered['RB_count'] == 1) & (runplaysfiltered['TE_count'] == 1)]

# Group by formation and calculate YPC and EPA
formation_stats = run_11p.groupby('offense_formation').agg(
    YPC=('yards_gained', 'mean'),
    EPA=('epa', 'mean')  # Change 'wp' to 'epa' if available in your dataset
).reset_index()

print(formation_stats.head())  # Check the output
"""
#----------------------------------------------------------------------------------------
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Filter down to necessary columns for grouping
personnel_stats = runplaysfiltered.groupby(['offense_formation', 'RB_count', 'TE_count']).agg(
    YPC=('yards_gained', 'mean'),
    EPA=('epa', 'mean')
).reset_index()

# Get unique formations
valid_personnel = {
    'Jumbo': [(1, 4), (2, 3), (3, 2)],  # Heavy TE/RB sets
    'Singleback': [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],  # Typically 1-2 RBs, no FB
    'I_Form': [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)],  # Common RB/TE distributions
    'Shotgun': [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)],  # 1-2 RBs allowed
    'Pistol': [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)],  # Similar to Shotgun
    'Goal Line': [(1, 3), (1, 4), (2, 3)],  # Heavy personnel only
}
filtered_personnel_stats = personnel_stats[
    personnel_stats.apply(lambda row: (row['RB_count'], row['TE_count']) in valid_personnel.get(row['offense_formation'], []), axis=1)
]
# Create a clustered bar graph for each formation
for formation in valid_personnel.keys():
    data = filtered_personnel_stats[filtered_personnel_stats['offense_formation'] == formation]
    
    # Sort by TE count first, then RB count for better visualization
    data = data.sort_values(by=['TE_count', 'RB_count'])
    
    # Combine TE count and RB count as labels for the x-axis
    data['personnel_combo'] = [f"{te} TE, {rb} RB" for te, rb in zip(data['TE_count'], data['RB_count'])]
    
    plt.figure(figsize=(10, 6))
    
    # Plot YPC (blue) and EPA (orange)
    sns.barplot(data=data, x='personnel_combo', y='YPC', color='blue', label='YPC')
    sns.barplot(data=data, x='personnel_combo', y='EPA', color='orange', label='EPA')
    
    # Set labels and title
    plt.xlabel('Personnel Packages (TE count → RB count)')
    plt.ylabel('Metric Value')
    plt.title(f'Run Performance in {formation} Formation')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
"""
#-----------------------------------------------------------------------------------------
#------------------------------- Exploratory Graphics ------------------------------------
#-----------------------------------------------------------------------------------------
# YPC by Temperature
import numpy as np
import matplotlib.pyplot as plt

# Group by temperature and calculate the mean YPC
yards_gained_temp = runplaysfiltered.groupby('temp')['yards_gained'].mean().reset_index()

# Extract x and y values
x = yards_gained_temp['temp']
y = yards_gained_temp['yards_gained']

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the YPC by temperature
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='Yards Gained')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Change in Yards per Carry with Temperature')
plt.xlabel('Temperature (F)')
plt.ylabel('Yards Gained')
plt.legend()
plt.grid(True)
plt.show()
print(f"Slope of the trend line: {z[0]}")

# YPC by Roof
# Calculate the success rates for games with and without a roof
import matplotlib.pyplot as plt

YPC_roof = runplaysfiltered.groupby('roof')['yards_gained'].mean()

# Plot the bar chart
ax = YPC_roof.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'])
plt.title('YPC: Roof (Open/Outdoors) vs. No Roof (Closed/Indoors)')
plt.xlabel('Roof (1) vs. No Roof (0)')
plt.ylabel('Yards Gained')
plt.xticks(ticks=[0, 1], labels=['No Roof', 'Roof'], rotation=0)
plt.grid(axis='y')

# Add numerical values on top of the bars
for i, v in enumerate(YPC_roof):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

plt.show()

# YPC by Surface
import matplotlib.pyplot as plt

# Grouping by 'grass' and calculating the mean yards gained
YPC_grass = runplaysfiltered.groupby('grass')['yards_gained'].mean()

# Plot the bar chart
ax = YPC_grass.plot(kind='bar', figsize=(8, 5), color=['black', 'green'])
plt.title('YPC: Grass Surface vs. Non-Grass Surface')
plt.xlabel('Surface Type (0 = Non-Grass, 1 = Grass)')
plt.ylabel('Yards Gained')
plt.xticks(ticks=[0, 1], labels=['Non-Grass', 'Grass'], rotation=0)
plt.grid(axis='y')

# Add numerical values on top of the bars
for i, v in enumerate(YPC_grass):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

plt.show()

# YPC by Shotgun
import matplotlib.pyplot as plt

# Grouping by 'Shotgun' and calculating the mean yards gained
YPC_shotgun = runplaysfiltered.groupby('shotgun')['yards_gained'].mean()

# Plot the bar chart
ax = YPC_shotgun.plot(kind='bar', figsize=(8, 5), color=['black', 'red'])
plt.title('YPC: Shotgun vs Under Center')
plt.xlabel('Formation Type (0 = Under Center, 1 = Shotgun)')
plt.ylabel('Yards Gained')
plt.xticks(ticks=[0, 1], labels=['Under Center', 'Shotgun'], rotation=0)
plt.grid(axis='y')

# Add numerical values on top of the bars
for i, v in enumerate(YPC_shotgun):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

plt.show()
#---------------------------------------------------------------------------------------
# Group data into bins for game_seconds_remaining and reverse the order
runplaysfiltered['seconds_bin'] = pd.cut(
    runplaysfiltered['game_seconds_remaining'], 
    bins=[0, 900, 1800, 2700, 3600], 
    labels=["Q4", "Q3", "Q2", "Q1"]
)

# Reverse the index order for plotting
YPC_seconds = runplaysfiltered.groupby('seconds_bin')['yards_gained'].mean()
YPC_seconds = YPC_seconds[::-1]  # Reverse the order

# Fit a linear trend line
x_numeric = range(len(YPC_seconds))  # Convert categories to numerical indices
coeffs = np.polyfit(x_numeric, YPC_seconds.values, 1)  # Linear regression (slope, intercept)
trend_line = np.polyval(coeffs, x_numeric)  # Compute trend line values

# Plot the data points and trend line
plt.figure(figsize=(10, 6))
plt.plot(YPC_seconds.index, YPC_seconds.values, marker='o', color='purple', linestyle='-', label='YPC')
plt.plot(YPC_seconds.index, trend_line, linestyle='--', color='red', label='Trend Line')
plt.title('YPC by Game Seconds Remaining (Quarterly Bins)')
plt.xlabel('Quarter')
plt.ylabel('Yards Gained')
plt.grid(axis='y')
plt.legend()

# Display slope on the graph
slope = coeffs[0]
plt.text(2.5, YPC_seconds.min() - 0.3, f"Slope: {slope:.2f}", ha='center', va='top', fontsize=10)

plt.tight_layout()
plt.show()

# YPC by Time of Possession
# Convert time of possession from "MM:SS" to seconds
runplaysfiltered['drive_time_of_possession_sec'] = runplaysfiltered['drive_time_of_possession'].str.split(':').apply(
    lambda x: int(x[0]) * 60 + int(x[1])
)

# Group data into bins for drive_time_of_possession in seconds
runplaysfiltered['time_possession_bin'] = pd.cut(
    runplaysfiltered['drive_time_of_possession_sec'], 
    bins=[0, 60, 120, 180, 240, 300, 900], 
    labels=["0-1 min", "1-2 min", "2-3 min", "3-4 min", "4-5 min", "5+ min"]
)

# Calculate mean YPC for each bin
YPC_time_possession = runplaysfiltered.groupby('time_possession_bin')['yards_gained'].mean()

# Fit a linear trend line
x_numeric = range(len(YPC_time_possession))
coeffs = np.polyfit(x_numeric, YPC_time_possession.values, 1)
trend_line = np.polyval(coeffs, x_numeric)

# Plot the data points and trend line
plt.figure(figsize=(10, 6))
plt.plot(YPC_time_possession.index, YPC_time_possession.values, marker='o', color='orange', linestyle='-', label='YPC')
plt.plot(YPC_time_possession.index, trend_line, linestyle='--', color='red', label='Trend Line')
plt.title('YPC by Drive Time of Possession')
plt.xlabel('Drive Time of Possession')
plt.ylabel('Yards Gained')
plt.grid(axis='y')
plt.legend()

# Display slope on the graph
slope = coeffs[0]
plt.text(2.5, YPC_time_possession.min() - 0.3, f"Slope: {slope:.2f}", ha='center', va='top', fontsize=10)

plt.tight_layout()
plt.show()


# YPC by Drive Play Count
# Group data into bins for drive_play_count
runplaysfiltered['play_count_bin'] = pd.cut(
    runplaysfiltered['drive_play_count'], 
    bins=[0, 5, 10, 15, 20, 30], 
    labels=["1-5", "6-10", "11-15", "16-20", "21+"]
)

# Calculate mean YPC for each bin
YPC_play_count = runplaysfiltered.groupby('play_count_bin')['yards_gained'].mean()

# Fit a linear trend line
x_numeric = range(len(YPC_play_count))
coeffs = np.polyfit(x_numeric, YPC_play_count.values, 1)
trend_line = np.polyval(coeffs, x_numeric)

# Plot the data points and trend line
plt.figure(figsize=(10, 6))
plt.plot(YPC_play_count.index, YPC_play_count.values, marker='o', color='blue', linestyle='-', label='YPC')
plt.plot(YPC_play_count.index, trend_line, linestyle='--', color='red', label='Trend Line')
plt.title('YPC by Drive Play Count')
plt.xlabel('Drive Play Count Range')
plt.ylabel('Yards Gained')
plt.grid(axis='y')
plt.legend()

# Display slope on the graph
slope = coeffs[0]
plt.text(2.5, YPC_play_count.min() - 0.3, f"Slope: {slope:.2f}", ha='center', va='top', fontsize=10)

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------
#-------------------------------------- Defining Run Gaps ---------------------------------
#------------------------------------------------------------------------------------------
def assign_run_gap_number(run_location, run_gap):
    if run_location == "left":
        return {"guard": 1, "tackle": 3, "end": 5}.get(run_gap, None)
    elif run_location == "middle":
        return 0
    elif run_location == "right":
        return {"guard": 2, "tackle": 4, "end": 6}.get(run_gap, None)
    return None

# Apply function to dataset
runplaysfiltered['run_gap_number'] = runplaysfiltered.apply(lambda row: assign_run_gap_number(row['run_location'], row['run_gap']), axis=1)

# Define the desired order of gaps
gap_order = [5, 3, 1, 0, 2, 4, 6]

# Convert 'numerical_run_gap' to a categorical variable with a specified order
runplaysfiltered['numerical_run_gap'] = pd.Categorical(runplaysfiltered['run_gap_number'], categories=gap_order, ordered=True)

# Group data by numerical run gap and calculate YPC
YPC_run_gap = runplaysfiltered.groupby('numerical_run_gap')['yards_gained'].mean()

# Plot YPC by run gap (left-to-right)
plt.figure(figsize=(10, 6))
YPC_run_gap.plot(kind='bar', color='red', edgecolor='black')

# Title and labels
plt.title('YPC by Numerical Run Gap (Ordered Left-to-Right on Offensive Line)')
plt.xlabel('Numerical Run Gap')
plt.ylabel('Yards Per Carry')
plt.xticks(ticks=range(len(gap_order)), labels=gap_order, rotation=0)  # Ensure correct tick labels
plt.grid(axis='y')

# Add numerical values on top of bars
for i, val in enumerate(YPC_run_gap.values):
    plt.text(i, val + 0.1, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------
#-------------------------- Segmenting the Field into 20 yard chunks ------------------------------------
#--------------------------------------------------------------------------------------------------------

# Segment the field into 20-yard chunks
insideown20 = runplaysfiltered[runplaysfiltered['yardline_100'] > 80]
own20to40 = runplaysfiltered[(runplaysfiltered['yardline_100'] <= 80) & (runplaysfiltered['yardline_100'] > 60)]
midfield = runplaysfiltered[(runplaysfiltered['yardline_100'] <= 60) & (runplaysfiltered['yardline_100'] > 40)]
opp40to20 = runplaysfiltered[(runplaysfiltered['yardline_100'] <= 40) & (runplaysfiltered['yardline_100'] > 20)]
oppredzone = runplaysfiltered[runplaysfiltered['yardline_100'] <= 20]

import pandas as pd
import numpy as np
def plot_ypc_by_run_gap(segment_df, segment_name):
    # Define the desired order of gaps
    gap_order = [5, 3, 1, 0, 2, 4, 6]
    
    # Group by numerical run gap and calculate YPC
    YPC_run_gap = segment_df.groupby('numerical_run_gap')['yards_gained'].mean()
    
    # Reindex gaps to ensure proper order
    YPC_run_gap = YPC_run_gap.reindex(gap_order).sort_index(key=lambda x: [gap_order.index(i) for i in x])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(gap_order)), YPC_run_gap.values, color='skyblue', edgecolor='black')
    plt.title(f'YPC by Numerical Run Gap ({segment_name})')
    plt.xlabel('Numerical Run Gap (Left to Right on O-Line)')
    plt.ylabel('Yards Per Carry')
    plt.xticks(ticks=range(len(gap_order)), labels=gap_order, rotation=0)  # Ensure correct tick labels
    plt.grid(axis='y')
    
    # Add numerical values on top of bars
    for i, val in enumerate(YPC_run_gap.values):
        if not pd.isna(val):  # Avoid NaN values
            plt.text(i, val + 0.1, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# Plot YPC by run gap for each field segment
plot_ypc_by_run_gap(insideown20, 'Inside Own 20')
plot_ypc_by_run_gap(own20to40, 'Own 20 to 40')
plot_ypc_by_run_gap(midfield, 'Midfield')
plot_ypc_by_run_gap(opp40to20, 'Opponent 40 to 20')
plot_ypc_by_run_gap(oppredzone, 'Opponent Red Zone')
