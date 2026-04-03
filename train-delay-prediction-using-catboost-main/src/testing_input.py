import pandas as pd

sample = pd.DataFrame({
    'Distance Between Stations (km)': [100],
    'Weather Conditions': ['Clear'],
    'Day of the Week': ['Monday'],
    'Time of Day': ['Morning'],
    'Train Type': ['Express'],
    'Route Congestion': ['Low']
})

sample.to_csv(r'D:\trainDelayPrediction\data\new_schedule.csv', index=False)
