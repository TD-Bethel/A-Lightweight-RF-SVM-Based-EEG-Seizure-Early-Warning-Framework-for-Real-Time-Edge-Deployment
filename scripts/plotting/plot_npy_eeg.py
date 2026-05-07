
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Slider, Button
import pathlib

# Path to the npy files
NPY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'Mendelay dataset', 'Npy_files')
X_TEST_PATH = os.path.join(NPY_DIR, 'x_test.npy')
Y_TEST_PATH = os.path.join(NPY_DIR, 'y_test.npy')

# Load the data
x_test = np.load(X_TEST_PATH)
try:
    y_test = np.load(Y_TEST_PATH)
except Exception:
    y_test = None

print(f"x_test shape: {x_test.shape}")
if y_test is not None:
    print(f"y_test shape: {y_test.shape}")


# Interactive plot with slider
sample_count = x_test.shape[0]

# Directory to save plots
save_dir = pathlib.Path(__file__).parent / 'Mendelay Sample'
save_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 4))
plt.subplots_adjust(bottom=0.25)

# Plot all channels for the first sample
lines = []
for ch in range(x_test.shape[1]):
    line, = ax.plot(x_test[0][ch], label=f'Ch {ch+1}')
    lines.append(line)

title = ax.set_title(f"EEG Signal Sample #0" + (f" | Label: {y_test[0]}" if y_test is not None else ""))
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.legend(loc='upper right', fontsize='small', ncol=2)

ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(ax_slider, 'Sample', 0, sample_count - 1, valinit=0, valstep=1)


# Save button callback
def save_current_plot(event):
    idx = int(slider.val)
    fname = save_dir / f'sample_{idx}.png'
    fig.savefig(fname)
    print(f"Saved plot: {fname}")

def update(val):
    idx = int(slider.val)
    for ch, line in enumerate(lines):
        line.set_ydata(x_test[idx][ch])
    title.set_text(f"EEG Signal Sample #{idx}" + (f" | Label: {y_test[idx]}" if y_test is not None else ""))
    fig.canvas.draw_idle()


# Add save button
ax_save = plt.axes([0.85, 0.025, 0.1, 0.04])
btn_save = Button(ax_save, 'Save Sample')
btn_save.on_clicked(save_current_plot)

slider.on_changed(update)
plt.show()
