import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import sys

# Check for command line arguments for file paths
if len(sys.argv) > 2:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
elif len(sys.argv) > 1:
    file1 = sys.argv[1]
    file2 = None
else:
    file1 = 'SUZ - DLS - BOT FP2R2.txt'
    file2 = 'SUZ - CAR - BOT FP2R2.txt'

# Read the data files
data1 = pd.read_csv(file1, sep='\t', skiprows=2, header=None)
data1 = data1.dropna()  # Remove rows with NaN values
if file2:
    data2 = pd.read_csv(file2, sep='\t', skiprows=2, header=None)
    data2 = data2.dropna()

# Columns: 0: time, 1: hRideF, 2: hRideR, 3: gVert, 4: gVertF, 5: gVertR, 6: undefined, 7: m
channels = ['hRideF', 'hRideR', 'gVert', 'gVertF', 'gVertR']
fs = 100  # Sampling frequency in Hz

# Create separate plots for each channel
for i, ch in enumerate(channels):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    signal1 = data1.iloc[:, i+1].values
    # Compute PSD using Welch's method
    f1, Pxx1 = welch(signal1, fs=fs, nperseg=256)
    # Plot
    ax.semilogy(f1, Pxx1, label=file1.split('/')[-1].split('\\')[-1])
    
    if file2:
        signal2 = data2.iloc[:, i+1].values
        f2, Pxx2 = welch(signal2, fs=fs, nperseg=256)
        ax.semilogy(f2, Pxx2, label=file2.split('/')[-1].split('\\')[-1])
    
    ax.set_title(f'Power Spectral Density of {ch}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_ylabel('PSD [V²/Hz]', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    if file2:
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'psd_{ch}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory