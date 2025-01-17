# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:04:33 2024

@author: NAWri
"""
pip install nfl_data_py
pip install matplotlib

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
           'div_game','roof','surface','temp','defenders_in_box','offense_personnel',]
runplaysfiltered = run_filtered[columns]

# change grass to be binary grass or turf
runplaysfiltered = runplaysfiltered.rename(columns ={
    'surface': 'grass'
})

runplaysfiltered['grass'] = runplaysfiltered['grass'].apply(lambda x: 1 if x in ['grass'] else 0)

# Modify the 'roof' variable to be binary with outdoors and open being 1, else being 0
runplaysfiltered['roof'] = runplaysfiltered['roof'].apply(lambda x: 1 if x in ['outdoors', 'open'] else 0)

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

# Define order of gaps
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

#------------------------------------------------------------------------------------------------------------
#------------------------------------------- YPC by OL Grade ------------------------------------------------
#------------------------------------------------------------------------------------------------------------
import pandas as pd
stats21 = pd.read_csv("C:/Users/natha/Documents/BGA/Summer2024/4th_Down_Prob/2021Stats.csv")
stats22 = pd.read_csv("C:/Users/natha/Documents/BGA/Summer2024/4th_Down_Prob/2022Stats.csv")
stats23 = pd.read_csv("C:/Users/natha/Documents/BGA/Summer2024/4th_Down_Prob/2023Stats.csv")
OLColumns = ['Year','Team','OLRank','PFFOL']
combined_stats = pd.concat([stats21, stats22, stats23], ignore_index=True)
OLStats = combined_stats[OLColumns]
OLStats = OLStats.rename(columns={
    'Team': 'posteam',
    'Year':'season'
})
run_filtered = pd.merge(
    run_filtered,
    OLStats,
    left_on=['posteam', 'season'],
    right_on=['posteam', 'season'],
    how='left'
)
columns = ['run_location','run_gap','yardline_100','game_seconds_remaining',
           'quarter_seconds_remaining','half_seconds_remaining','down',
           'goal_to_go','ydstogo','yards_gained','play_type','shotgun',
           'no_huddle','no_score_prob','opp_fg_prob','opp_td_prob','fg_prob',
           'td_prob','wp','def_wp','penalty','drive_play_count','drive_time_of_possession',
           'div_game','roof','surface','temp','defenders_in_box','offense_personnel','OLRank','PFFOL']
runplaysfiltered = run_filtered[columns]

yards_gained_OL = runplaysfiltered.groupby('OLRank')['yards_gained'].mean().reset_index()
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# Extract x and y values
x = yards_gained_OL['OLRank']
y = yards_gained_OL['yards_gained']

# Fit a linear trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Plot the YPC by temperature
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='Yards Gained')
plt.plot(x, p(x), "r--", label='Trend Line')  # Trend line

plt.title('Change in Yards per Carry with O Line Rank')
plt.xlabel('OLRank')
plt.ylabel('Yards Gained')
plt.legend()
plt.grid(True)
plt.show()
print(f"Slope of the trend line: {z[0]}")

#--------------------------------------------------------------------------------------------------------------
#-------------------------------------------- YPC feature testing ---------------------------------------------
#--------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Define predictors and target variable
features = [
    'numerical_run_gap',  # Run gap
    'yardline_100',       # Field position
    'roof',               # Roof type (binary or categorical)
    'surface',            # Surface type
    'field_segment'       # Segmented field position
]
target = 'yards_gained'

# Create field_segment feature if not already created
if 'field_segment' not in runplaysfiltered.columns:
    runplaysfiltered['field_segment'] = pd.cut(
        runplaysfiltered['yardline_100'], 
        bins=[0, 20, 40, 60, 80, 100], 
        labels=['oppredzone', 'opp40to20', 'midfield', 'own20to40', 'insideown20']
    )

# Extract relevant data and drop rows with missing target
data = runplaysfiltered[features + [target]].dropna(subset=[target])

# Separate predictors (X) and target (y)
X = data[features]
y = data[target]

# Identify categorical and numerical features
categorical_features = ['roof', 'surface', 'field_segment']
numerical_features = ['numerical_run_gap', 'yardline_100']

# Define the preprocessing steps with imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numerical values with mean
            ('scaler', StandardScaler())                 # Scale numerical features
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values with mode
            ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical features
        ]), categorical_features)
    ]
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model pipeline with imputation
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Define cross-validation strategy
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation with error handling
try:
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error'
    )
    
    # Calculate RMSE from cross-validation scores
    cv_rmse = np.sqrt(-cv_scores)
    print(f"Mean RMSE from cross-validation: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")
except ValueError as e:
    print("Error during cross-validation:", e)

# Fit the model on the training set
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE on the test set
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.2f}")

# Extract feature names after preprocessing
# Get numerical feature names
num_features = numerical_features

# Get categorical feature names from one-hot encoding
cat_features = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)

# Combine all feature names
all_features = list(num_features) + list(cat_features)

# Extract feature importances from the Random Forest model
importances = model.named_steps['regressor'].feature_importances_

# Create a DataFrame for visualization
feature_importances_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top 10 feature importances
print("\nTop 10 Feature Importances:")
print(feature_importances_df.head(10))

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Predicting Yards Per Carry (YPC)')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()
