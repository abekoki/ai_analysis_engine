
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# X-axis column configuration
x_axis_column = 'frame_num'

# Load data
df = pd.read_csv(r'../development_datas/03_algorithm_output\v0.1.1+fa5172ba\20250922-133705\1.csv')

# Filter to target interval if available


# Define frame range variables for axis limits
start_frame_val = None
end_frame_val = None

# Thresholds configuration
thresholds = []

# Determine columns to plot
plot_columns = ['is_drowsy']

# Filter to columns that exist in the data
plot_cols = [col for col in plot_columns if col in df.columns]
if not plot_cols:
    plot_cols = df.columns[:min(5, len(df.columns))]

# Create figure with subplots
fig, axes = plt.subplots(len(plot_cols), 1, figsize=(12, 4*len(plot_cols)))
if len(plot_cols) == 1:
    axes = [axes]

    # Determine x-axis data using inferred column
    x_data = df.index
    x_label = 'Index'
    if x_axis_column and x_axis_column in df.columns:
        x_data = df[x_axis_column]
        x_label = x_axis_column.replace('_', ' ').title()
    else:
        # Fallback to common patterns
        for col in ['frame', 'frame_num', 'timestamp', 'time']:
            if col in df.columns:
                x_data = df[col]
                x_label = col.replace('_', ' ').title()
                break

# Plot each column with thresholds
for i, col in enumerate(plot_cols):
    axes[i].plot(x_data, df[col], label=col, linewidth=2)

    # Add threshold lines if available and column matches threshold
    for threshold_name, threshold_value in thresholds:
        # Apply threshold if column name matches threshold name (case-insensitive partial match)
        if threshold_name.lower().replace('_threshold', '').replace('_', '') in col.lower():
            label_text = threshold_name + ': ' + str(threshold_value)
            axes[i].axhline(y=threshold_value, color='red', linestyle='--', label=label_text)

    axes[i].set_title(col + ' - Algorithm Output')
    axes[i].set_xlabel(x_label)
    axes[i].set_ylabel('Value')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
