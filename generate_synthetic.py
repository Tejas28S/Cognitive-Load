import pandas as pd
import numpy as np
import os

# Load the upload template to get the correct column names
template_path = os.path.join("serving", "upload_template.csv")
if not os.path.exists(template_path):
    print(f"Error: {template_path} not found.")
    exit(1)

template_df = pd.read_csv(template_path)
columns = template_df.columns.tolist()

# Number of synthetic samples to generate
n_samples = 10

# Create an empty dataframe with the right columns
synthetic_df = pd.DataFrame(columns=columns)

# Fill metadata columns if they exist
metadata_cols = ['subject_id', 'trial_id', 'task_type', 'window_idx', 'chunk_id', 'label']
for col in metadata_cols:
    if col in columns:
        if col == 'subject_id':
            synthetic_df[col] = ['SYNTH_' + str(i) for i in range(n_samples)]
        elif col == 'label':
            synthetic_df[col] = np.random.choice([0, 1, 2], size=n_samples)
        else:
            synthetic_df[col] = 0

# Fill feature columns with random noise (e.g., normally distributed around 0 with std 1, or just uniform)
# Using standard normal distribution as a base for synthetic signals
feature_cols = [c for c in columns if c not in metadata_cols]
synthetic_df[feature_cols] = np.random.randn(n_samples, len(feature_cols)) * 2.5 + 0.5

# Let's add some artificial patterns just so it's not pure noise
# Make the first 5 samples "High" load like and the next 5 "Low" load like (if the model caught on to mean values)
synthetic_df.loc[0:4, feature_cols[:100]] += 5.0  # Add a strong signal to the first 100 EEG features for the first 5 samples
synthetic_df.loc[5:, feature_cols[:100]] -= 5.0   # Add a strong negative signal to the first 100 EEG features for the last 5 samples

# Set labels just in case the frontend/backend uses them for display
if 'label' in columns:
    synthetic_df.loc[0:4, 'label'] = 2 # High
    synthetic_df.loc[5:, 'label'] = 0  # Low

out_path = "synthetic_upload_test.csv"
synthetic_df.to_csv(out_path, index=False)
print(f"Generated synthetic data with {n_samples} rows at {out_path}")
