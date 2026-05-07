import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os


# Prompt user to select the Bonn dataset folder
print("Select the Bonn dataset folder to visualize:")
options = ['F', 'N', 'O', 'S', 'Z']
for i, opt in enumerate(options):
    print(f"{i+1}. {opt}")
choice = input(f"Enter number (1-{len(options)}), or folder name: ").strip()
if choice.isdigit() and 1 <= int(choice) <= len(options):
    folder = options[int(choice)-1]
else:
    folder = choice.upper()

BONN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'Bonn Univeristy Dataset', folder)

# List all .txt files in the directory
bonn_files = sorted([f for f in os.listdir(BONN_DIR) if f.endswith('.txt')])

# Directory to save plots
save_dir = pathlib.Path(__file__).parent / 'Bonn Sample'
save_dir.mkdir(exist_ok=True)

# Function to load a Bonn EEG .txt file (single channel)
def load_bonn_txt(filepath):
    return np.loadtxt(filepath)

# Plot and save all samples with a slider
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

sample_count = len(bonn_files)

fig, ax = plt.subplots(figsize=(12, 4))
plt.subplots_adjust(bottom=0.25)

# Load the first sample
signal = load_bonn_txt(os.path.join(BONN_DIR, bonn_files[0]))
[line] = ax.plot(signal, label=bonn_files[0])
title = ax.set_title(f"Bonn EEG Sample #0 | {bonn_files[0]}")
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.legend(loc='upper right', fontsize='small')

ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(ax_slider, 'Sample', 0, sample_count - 1, valinit=0, valstep=1)


# Save button callback
def save_current_plot(event):
    idx = int(slider.val)
    fname = save_dir / f'sample_{idx}_{bonn_files[idx].replace('.txt','')}.png'
    fig.savefig(fname)
    print(f"Saved plot: {fname}")

def update(val):
    idx = int(slider.val)
    signal = load_bonn_txt(os.path.join(BONN_DIR, bonn_files[idx]))
    line.set_ydata(signal)
    line.set_label(bonn_files[idx])
    ax.relim()
    ax.autoscale_view()
    title.set_text(f"Bonn EEG Sample #{idx} | {bonn_files[idx]}")
    ax.legend(loc='upper right', fontsize='small')
    fig.canvas.draw_idle()


# Add save button
ax_save = plt.axes([0.85, 0.025, 0.1, 0.04])
btn_save = Button(ax_save, 'Save Sample')
btn_save.on_clicked(save_current_plot)

slider.on_changed(update)
plt.show()
