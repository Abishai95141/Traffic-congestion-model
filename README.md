# Traffic-congestion-model

## Program

```
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import time

# Step 1: Load and preprocess the dataset
# Load the new dataset with geographic and road width features
df = pd.read_csv('/content/TRAFFIC_DATA.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
df.set_index('timestamp', inplace=True)
df.info()
df.head()

# Step 2: Handle missing data
# Check for missing values and fill them with forward-fill method
df.fillna(method='ffill', inplace=True)

# Step 3: Define feature sets
# Define the features including new geographic and road width features
features = ['vehicle_count', 'avg_speed', 'traffic_density', 'event', 'crowd_size',
            'weather_condition', 'temperature', 'road_condition', 'zone',
            'latitude', 'longitude', 'altitude', 'road_width']

# Separate categorical and numerical features, including geographic and road width
categorical_features = ['event', 'weather_condition', 'road_condition']
numerical_features = ['vehicle_count', 'avg_speed', 'traffic_density', 'crowd_size',
                      'temperature', 'latitude', 'longitude', 'altitude', 'road_width']

# Step 4: Encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = encoder.fit_transform(df[categorical_features])

# Step 5: Scale numerical features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_numerical = scaler.fit_transform(df[numerical_features])

# Step 6: Combine scaled numerical and encoded categorical features
data = np.concatenate([scaled_numerical, encoded_cats], axis=1)

# Step 7: Define function to calculate maximum vehicle capacity based on road width
def get_max_vehicle_capacity(road_width):
    # Assume each meter of road width can accommodate up to 2 vehicles
    return road_width * 2

# Step 8: Create sequences of data including road width for calculating capacity
def create_sequences_with_capacity(data, seq_length, road_widths):
    sequences = []
    labels = []
    capacities = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        labels.append(data[i][0])  # vehicle count as the label
        capacities.append(get_max_vehicle_capacity(road_widths[i]))  # Calculate capacity
    return np.array(sequences), np.array(labels), np.array(capacities)

# Step 9: Extract road widths and create input sequences
road_widths = df['road_width'].values
SEQ_LENGTH = 60
X, y, max_capacities = create_sequences_with_capacity(data, SEQ_LENGTH, road_widths)

# Step 10: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Define LSTM model for traffic prediction
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 12: Train the LSTM model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 13: Make predictions using the trained model
predictions = model.predict(X_test)

# Step 14: Rescale predictions to original scale
predicted_congestion_rescaled = scaler.inverse_transform(
    np.concatenate((predictions, X_test[:, -1, 1:len(numerical_features)]), axis=1)
)[:, 0]

# Step 15: Adjust vehicle counts based on road capacity
predicted_congestion_adjusted = np.minimum(predicted_congestion_rescaled, max_capacities[:len(predicted_congestion_rescaled)])

# Step 16: Define zones from the original dataset
zones = df['zone'].unique()

# Step 17: Perform rerouting if predicted congestion exceeds capacity
rerouted_zones = [zones[(int(cong) + 1) % len(zones)] if cong > max_capacities[i]
                  else zones[int(cong) % len(zones)]
                  for i, cong in enumerate(predicted_congestion_adjusted)]

# Step 18: Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 19: Output predicted congestion and rerouting suggestions
# Convert index (timestamp) to list to access it easily
timestamps = df.index.tolist()
current_timestamp = None
# Displays Congestion & Rerouting details for the first 24 hours
for i in range(24*6):
    if timestamps[i] != current_timestamp:
        current_timestamp = timestamps[i]
        print(f"\nTimestamp: {current_timestamp.strftime('%Y/%m/%d %H:%M')}")

    print(f"Predicted Congestion: {predicted_congestion_adjusted[i]:<15.2f}, "
          f"Current Zone: {df['zone'][i]:<10}, "
          f"Suggested Rerouting Zone: {rerouted_zones[i]:<15}")

# Step 20: Create a DataFrame with necessary features for the prediction report
predictions_df = pd.DataFrame({
    'timestamp': df.index[:len(predicted_congestion_adjusted)],
    'congestion_level': predicted_congestion_adjusted / max_capacities[:len(predicted_congestion_adjusted)],  # Normalized congestion level
    'zone': df['zone'][:len(predicted_congestion_adjusted)],
    'rerouted_zone': rerouted_zones[:len(predicted_congestion_adjusted)],
    'max_capacity': max_capacities[:len(predicted_congestion_adjusted)],
    'vehicle_count_before': df['vehicle_count'][:len(predicted_congestion_adjusted)],
    'vehicle_count_after': np.minimum(predicted_congestion_adjusted, max_capacities[:len(predicted_congestion_adjusted)]),
})

# Step 21: Save to a CSV file
predictions_csv_path = '/content/prediction.csv'  # Specify your desired path
predictions_df.to_csv(predictions_csv_path, index=False)

print(f"Predictions saved to {predictions_csv_path}")

# Step 22: Add rerouted zones to the DataFrame
df['rerouted_zones'] = np.nan  # Initialize with NaN
df.iloc[-len(rerouted_zones):, df.columns.get_loc('rerouted_zones')] = rerouted_zones

# Step 23: Group by zones and calculate average vehicle count and other features before and after rerouting
before_rerouting_data = df.groupby('zone')[['vehicle_count', 'latitude', 'longitude', 'altitude']].mean().reset_index()
after_rerouting_data = df.groupby('rerouted_zones')[['vehicle_count', 'latitude', 'longitude', 'altitude']].mean().reset_index()

# Step 24: Debugging - print grouped data
print(before_rerouting_data.head())
print(after_rerouting_data.head())

# Step 25: Perform K-Means clustering before rerouting
kmeans_before = KMeans(n_clusters=3, random_state=42)
before_rerouting_data['cluster'] = kmeans_before.fit_predict(before_rerouting_data[['vehicle_count', 'latitude', 'longitude', 'altitude']])

# Step 26: Plot clustering results before rerouting
plt.figure(figsize=(10, 6))
sns.scatterplot(x=before_rerouting_data['zone'],
                y=before_rerouting_data['vehicle_count'],
                hue=before_rerouting_data['cluster'],
                palette='Set1', s=100)
plt.xlabel("Zones")
plt.ylabel("Vehicle Count")
plt.title("Clustering Before Rerouting")
plt.xticks(rotation=90)
plt.show()

# Step 27: Handle mismatched length of rerouted zones and original DataFrame
if len(rerouted_zones) < len(df):
    rerouted_zones = np.resize(rerouted_zones, len(df))
elif len(rerouted_zones) > len(df):
    rerouted_zones = rerouted_zones[:len(df)]

# Assign rerouted zones back to DataFrame
df['rerouted_zone'] = rerouted_zones

# Step 28: Group data by rerouted zones and perform clustering
after_rerouting_data = df.groupby('rerouted_zone')['vehicle_count'].mean().reset_index()

# Ensure enough samples for clustering
if len(after_rerouting_data) >= 3:
    kmeans_after = KMeans(n_clusters=3, random_state=42)
    after_rerouting_data['cluster'] = kmeans_after.fit_predict(after_rerouting_data[['vehicle_count']])
else:
    print(f"Not enough samples for KMeans clustering. Available samples: {len(after_rerouting_data)}")
    after_rerouting_data['cluster'] = np.zeros(len(after_rerouting_data))  # Assign all to one cluster if not enough samples

# Step 29: Plot clustering results after rerouting
plt.figure(figsize=(10, 6))
sns.scatterplot(x=after_rerouting_data['rerouted_zone'],
                y=after_rerouting_data['vehicle_count'],
                hue=after_rerouting_data['cluster'],
                palette='Set2', s=100)
plt.xlabel("Zones")
plt.ylabel("Vehicle Count")
plt.title("Clustering After Rerouting")
plt.xticks(rotation=90)
plt.show()

# Step 30: Group data by zone before rerouting and rerouted zones after rerouting
before_rerouting_data = df.groupby('zone')['vehicle_count'].mean().reset_index()
after_rerouting_data = df.groupby('rerouted_zone')['vehicle_count'].mean().reset_index()

# Step 31: Merge before and after rerouting data for comparison
comparison_data = before_rerouting_data.merge(after_rerouting_data, left_on='zone', right_on='rerouted_zone', suffixes=('_before', '_after'))

# Step 32: Plot congestion diversion across zones before and after rerouting
plt.figure(figsize=(10, 6))
for zone in comparison_data['zone']:
    zone_data = comparison_data[comparison_data['zone'] == zone]
    plt.plot(['Before', 'After'],
             [zone_data['vehicle_count_before'].values[0], zone_data['vehicle_count_after'].values[0]],
             marker='o', label=f'Zone {zone}')
plt.title("Congestion Diversion Across Zones Before and After Rerouting")
plt.xlabel("Condition")
plt.ylabel("Average Vehicle Count")
plt.legend(title='Zone')

# Step 33: Visualize traffic intensity across different zones and times of day
plt.figure(figsize=(12, 8))
traffic_pivot = df.pivot_table(values='vehicle_count', index='zone', columns=df.index.hour, aggfunc='mean')
sns.heatmap(traffic_pivot, cmap='YlOrRd')
plt.xlabel('Hour of the Day')
plt.ylabel('Zone')
plt.title('Traffic Volume Heatmap by Zone and Time')
plt.show()

# Step 34: Plot a clustered bar chart of traffic volume before and after rerouting by zone
comparison_data.set_index('zone', inplace=True)
comparison_data[['vehicle_count_before', 'vehicle_count_after']].plot(kind='bar', figsize=(12, 6))
plt.xlabel('Zone')
plt.ylabel('Average Vehicle Count')
plt.title('Traffic Volume Before and After Rerouting by Zone')
plt.legend(['Before Rerouting', 'After Rerouting'])
plt.show()

# Function to generate a traffic report from data
def generate_traffic_report(timestamp, congestion_level, zone, rerouted_zone, max_capacity, vehicle_count_before, vehicle_count_after):
    report = f"""
    Traffic Report:
    At {timestamp}, the congestion level in {zone} is {congestion_level*100:.1f}%. 
    Rerouting is suggested to {rerouted_zone}. The maximum road capacity is {max_capacity} vehicles. 
    Before rerouting, there were {vehicle_count_before} vehicles, and after rerouting, there are {vehicle_count_after} vehicles.
    """
    return report

# Step 35: Load the CSV file containing predictions and data
# Loop through the data to generate reports
for index, row in predictions_df.head(10).iterrows(): 
    timestamp = row['timestamp']
    congestion_level = row['congestion_level']
    zone = row['zone']
    rerouted_zone = row['rerouted_zone']
    max_capacity = row['max_capacity']
    vehicle_count_before = row['vehicle_count_before']
    vehicle_count_after = row['vehicle_count_after']

    # Generate the traffic report using the data from the CSV
    traffic_report = generate_traffic_report(timestamp, congestion_level, zone, rerouted_zone, max_capacity, vehicle_count_before, vehicle_count_after)
    print(f"Traffic Report for {timestamp}:\n{traffic_report}\n")

    # Optional: Add a delay between reports to simulate processing time
    time.sleep(1)
```
