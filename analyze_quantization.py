
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Path to the CSV files
path = '/home/perelman/RoboticsDiffusionTransformer/pretrained_models/rdt/'
all_files = glob.glob(os.path.join(path, "report_w*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    # Extract bit number from filename
    bits = int(os.path.basename(filename).split('report_w')[1].split('_')[0])
    df['bits'] = bits
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

# Data cleaning and preparation
# The 'name' column has long names, let's simplify them for plotting
# For example, group by block number
frame['block'] = frame['name'].str.extract(r'model\.blocks\.(\d+)\.')
frame['layer_type'] = frame['name'].str.extract(r'model\.blocks\.\d+\.([a-z_]+)\.')

# Let's focus on the 'mse' and 'mae'
# We can plot the average mse/mae per bit width

avg_error = frame.groupby('bits')[['mse', 'mae']].mean().reset_index()

plt.figure(figsize=(12, 6))

# Plotting MSE
plt.subplot(1, 2, 1)
plt.plot(avg_error['bits'], avg_error['mse'], marker='o')
plt.xlabel('Number of Bits')
plt.ylabel('Average MSE')
plt.title('Average MSE vs. Number of Bits')
plt.grid(True)

# Plotting MAE
plt.subplot(1, 2, 2)
plt.plot(avg_error['bits'], avg_error['mae'], marker='o', color='r')
plt.xlabel('Number of Bits')
plt.ylabel('Average MAE')
plt.title('Average MAE vs. Number of Bits')
plt.grid(True)

plt.tight_layout()
plt.savefig('/home/perelman/RoboticsDiffusionTransformer/quantization_analysis.png')

print("Figure saved to /home/perelman/RoboticsDiffusionTransformer/quantization_analysis.png")

# Let's also look at the error distribution for a specific bit width, e.g., 4 bits
plt.figure(figsize=(12, 6))
frame_4bit = frame[frame['bits'] == 4]
plt.hist(frame_4bit['mse'], bins=50, alpha=0.7, label='MSE')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.title('Distribution of MSE for 4-bit quantization')
plt.legend()
plt.grid(True)
plt.savefig('/home/perelman/RoboticsDiffusionTransformer/mse_distribution_4bit.png')
print("Figure saved to /home/perelman/RoboticsDiffusionTransformer/mse_distribution_4bit.png")

# Another interesting plot could be to see which layers are most sensitive to quantization.
# Let's plot the top 10 layers with the highest MSE for 2-bit quantization.
frame_2bit = frame[frame['bits'] == 2].sort_values(by='mse', ascending=False).head(10)

plt.figure(figsize=(12, 8))
plt.barh(frame_2bit['name'], frame_2bit['mse'])
plt.xlabel('MSE')
plt.title('Top 10 Layers with Highest MSE (2-bit quantization)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/home/perelman/RoboticsDiffusionTransformer/top_mse_layers_2bit.png')
print("Figure saved to /home/perelman/RoboticsDiffusionTransformer/top_mse_layers_2bit.png")