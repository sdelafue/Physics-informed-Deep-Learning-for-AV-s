import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load the JSON files
data_root = Path("datasets/v1.0-mini")

with open(data_root / "sample.json") as f:
    samples = json.load(f)
    
with open(data_root / "sample_data.json") as f:
    sample_data_list = json.load(f)

# Create lookup: sample_token -> sample_data entries
sample_to_data = {}
for sd in sample_data_list:
    sample_token = sd['sample_token']
    if sample_token not in sample_to_data:
        sample_to_data[sample_token] = []
    sample_to_data[sample_token].append(sd)

# Build sequential list of LiDAR scans
lidar_scans = []

for sample in samples:
    sample_token = sample['token']
    timestamp = sample['timestamp']
    scene_token = sample['scene_token']
    
    # Get sensor data for this sample
    sensor_data = sample_to_data.get(sample_token, [])
    
    # Find the LIDAR_TOP keyframe
    for sd in sensor_data:
        if sd['is_key_frame'] and 'LIDAR_TOP' in sd['filename']:
            lidar_scans.append({
                'sample_token': sample_token,
                'scene_token': scene_token,
                'timestamp': timestamp,
                'filename': sd['filename'],
                'ego_pose_token': sd['ego_pose_token'],
                'calibrated_sensor_token': sd['calibrated_sensor_token'],
                'prev_sample': sample['prev'],
                'next_sample': sample['next']
            })
            break

# Convert to DataFrame
df_lidar = pd.DataFrame(lidar_scans)

# Display the DataFrame
print(f"Total LiDAR scans: {len(df_lidar)}")
print("\nFirst 10 scans:")
print(df_lidar.head(10))

# Save to CSV for later use (optional)
df_lidar.to_csv('lidar_scans_sequential.csv', index=False)
print("\nDataFrame saved to 'lidar_scans_sequential.csv'")

# Example: Access specific scan
print("\n" + "="*60)
print("Example - Accessing scan #5:")
print("="*60)
scan_5 = df_lidar.iloc[4]  # 0-indexed, so scan #5 is index 4
print(f"Sample token: {scan_5['sample_token']}")
print(f"Timestamp: {scan_5['timestamp']}")
print(f"Filename: {scan_5['filename']}")
print(f"Ego pose token: {scan_5['ego_pose_token']}")