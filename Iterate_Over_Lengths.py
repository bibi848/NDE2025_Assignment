#%%
# Import Functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import NDE_functions as nde
from scipy.fft import fft, ifft, fftfreq
from skimage import measure
import time

from my_functions import simScan
from my_functions import makeFilter
from my_functions import filterData
from my_functions import TFM_loop
from my_functions import image_crop
from my_functions import image_mask
from my_functions import contour_sizing
from my_functions import extract_diagonal

#%%
# Testing Algorithms
# Parameters
mode = 'zn22608'
centre_freq = 2.25e6  # Hz
velocity = 6020.0     # m/s
wavelength = velocity / centre_freq

element_pitch = wavelength * 0.5
element_width = element_pitch - 0.05e-3
no_elements = 64                 # number of elements
first_element_position = 1e-3    # m from weld edge

# Imaging region
width_m  = 80e-3 
depth_m  = 70e-3 
pixel    = 0.25e-3

# Crop Area
x_min = -32e-3
x_max = -1.5e-3
y_min = 27e-3
y_max = 57.7e-3

# Display Settings
db_dyn = 40.0 # dynamic range for image dB display
dt = 2e-8

H_smooth = makeFilter(2412, dt, centre_freq)
H_smooth = makeFilter(2414, dt, centre_freq)

scan_position = 0.13
B = 7

# Ray Tracing Sim
start = time.time()
fmc_data, time_vec, element_positions = simScan(mode, scan_position, no_elements, element_pitch, element_width, 
                                                first_element_position, centre_freq)
end = time.time()
process_time = end-start
Tx, Rx, Nt = fmc_data.shape

#%%
# Filtering Data
analytic = filterData(fmc_data, H_smooth, Tx, Rx)

# Full Image
x_vec = np.arange(-width_m/2, width_m/2 + pixel, pixel)
y_vec = np.arange(0.0, depth_m + pixel, pixel)
X, Y  = np.meshgrid(x_vec, y_vec)
image_tfm = TFM_loop(analytic, element_positions, X, Y, dt, no_elements, velocity, Rx, Nt)
image_amp_tfm = np.abs(image_tfm.astype(np.complex64))
image_db_tfm = 20.0 * np.log10(image_amp_tfm / (np.max(image_amp_tfm) + 1e-12))

# Cropped Image
x_vec_c, y_vec_c, image_db_c = image_crop(x_vec, y_vec, image_db_tfm, x_min, x_max, y_min, y_max)

# Masking Image
image_masked = image_mask(B, pixel, image_db_c, x_vec_c, y_vec_c)

# Defect Detection and Sizing
defect_length_mm, peak_db, level_db, xs_mm, ys_mm, peak_x_mm, peak_y_mm = contour_sizing(image_masked, x_vec_c, y_vec_c)

if defect_length_mm != 0 and peak_db > -34.5:
    defect_boolean = 'Defect Present'
else:
    defect_boolean = 'Defect Not Present'

#%%
# Plotting
fig, ax = plt.subplots(figsize=(6,5))
extent = [x_vec[0]*1e3, x_vec[-1]*1e3, y_vec[-1]*1e3, y_vec[0]*1e3]
im = ax.imshow(image_db_tfm, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
ax.set_xlabel('x [mm]')
ax.set_ylabel('Depth [mm]')
ax.set_title(f'Pos: {scan_position}, Time:{round(process_time, 4)}, {defect_boolean}')
plt.colorbar(im, ax=ax, label='Relative dB')
plt.tight_layout()
plt.show()

if defect_boolean == 'Defect Present':
    fig, ax = plt.subplots(figsize=(6,5))
    extent = [x_vec_c[0]*1e3, x_vec_c[-1]*1e3, y_vec_c[-1]*1e3, y_vec_c[0]*1e3]
    im = ax.imshow(image_db_c, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Depth [mm]')
    ax.set_title(f'Cropped, Pos: {scan_position}')
    plt.colorbar(im, ax=ax, label='Relative dB')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6,5))
    extent = [x_vec_c[0]*1e3, x_vec_c[-1]*1e3, y_vec_c[-1]*1e3, y_vec_c[0]*1e3]
    im = ax.imshow(image_masked, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Depth [mm]')
    ax.set_title(f'Diagonal Mask, Pos: {scan_position}')
    plt.colorbar(im, ax=ax, label='Relative dB')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6,5))
    extent = [x_vec_c[0]*1e3, x_vec_c[-1]*1e3, y_vec_c[-1]*1e3, y_vec_c[0]*1e3]
    ax.imshow(image_masked, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
    ax.plot(xs_mm, ys_mm, 'r-', linewidth=1.0)
    ax.plot(peak_x_mm, peak_y_mm, 'ro', markersize=1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Depth [mm]')
    ax.set_title(f'Sizing, Pos: {scan_position}, Defect Size: {defect_length_mm:.2f}, Peak: {peak_db:.1f}')
    plt.show()

#%%
# Iterating over Lengths

# Parameters
mode = 'zn22608'
centre_freq = 2.25e6  # Hz
velocity = 6020.0     # m/s
wavelength = velocity / centre_freq

element_pitch = wavelength * 0.5
element_width = element_pitch - 0.05e-3
no_elements = 64                 # number of elements
first_element_position = 1e-3    # m from weld edge

# Imaging region
width_m  = 80e-3 
depth_m  = 70e-3 
pixel    = 0.25e-3

# Crop Area
x_min = -32e-3
x_max = -1.5e-3
y_min = 27e-3
y_max = 57.7e-3

# Display Settings
db_dyn = 40.0 # dynamic range for image dB display
dt = 2e-8
B = 7

H_smooth = makeFilter(2414, dt, centre_freq)

scan_positions = np.arange(0, 303, 3)/1000
# scan_positions = np.arange(0, 310, 10)/1000

u = 0
for scan_position in scan_positions:
    # Ray tracing using transducer
    start = time.time()
    fmc_data, time_vec, element_positions = simScan(mode, scan_position, no_elements, element_pitch, element_width, 
                                                    first_element_position, centre_freq)
    end = time.time()
    process_time = end-start
    Tx, Rx, Nt = fmc_data.shape

    # Filtering Data
    analytic = filterData(fmc_data, H_smooth, Tx, Rx)

    # Full Image
    x_vec = np.arange(-width_m/2, width_m/2 + pixel, pixel)
    y_vec = np.arange(0.0, depth_m + pixel, pixel)
    X, Y  = np.meshgrid(x_vec, y_vec)

    image_tfm = TFM_loop(analytic, element_positions, X, Y, dt, no_elements, velocity, Rx, Nt)

    image_amp_tfm = np.abs(image_tfm.astype(np.complex64))
    image_db_tfm = 20.0 * np.log10(image_amp_tfm / (np.max(image_amp_tfm) + 1e-12))

    # Cropped Image
    x_vec_c, y_vec_c, image_db_c = image_crop(x_vec, y_vec, image_db_tfm, x_min, x_max, y_min, y_max)

    # Masking Image
    image_masked = image_mask(B, pixel, image_db_c, x_vec_c, y_vec_c)

    # Defect Detection and Sizing
    defect_length_mm, peak_db, level_db, xs_mm, ys_mm, peak_x_mm, peak_y_mm = contour_sizing(image_masked, x_vec_c, y_vec_c)

    if defect_length_mm != 0 and peak_db > -34.5:
        defect_boolean = 'Defect Present'
    else:
        defect_boolean = 'Defect Not Present'

    # Plotting and Saving Figures
    folder = 'Online Folder'

    fig, ax = plt.subplots(figsize=(6,5))
    extent = [x_vec[0]*1e3, x_vec[-1]*1e3, y_vec[-1]*1e3, y_vec[0]*1e3]
    im = ax.imshow(image_db_tfm, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Depth [mm]')
    # ax.set_title(f'Pos: {scan_position}, Time:{round(process_time, 4)}, {defect_boolean}')
    ax.set_title(f'Pos: {scan_position}, {defect_boolean}')
    plt.colorbar(im, ax=ax, label='Relative dB')
    plt.tight_layout()
    plt.show()

    fig.savefig(f"{folder}/{u}.0. Full Image, Pos {scan_position}.png", dpi=200, bbox_inches='tight')

    if defect_boolean == 'Defect Present':

        fig, ax = plt.subplots(figsize=(6,5))
        extent = [x_vec_c[0]*1e3, x_vec_c[-1]*1e3, y_vec_c[-1]*1e3, y_vec_c[0]*1e3]
        im = ax.imshow(image_db_c, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('Depth [mm]')
        ax.set_title(f'Cropped, Pos: {scan_position}')
        plt.colorbar(im, ax=ax, label='Relative dB')
        plt.tight_layout()
        plt.show()

        fig.savefig(f"{folder}/{u}.1. Cropped Image, Pos {scan_position}.png", dpi=200, bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(6,5))
        extent = [x_vec_c[0]*1e3, x_vec_c[-1]*1e3, y_vec_c[-1]*1e3, y_vec_c[0]*1e3]
        im = ax.imshow(image_masked, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('Depth [mm]')
        ax.set_title(f'Diagonal Mask, Pos: {scan_position}')
        plt.colorbar(im, ax=ax, label='Relative dB')
        plt.tight_layout()
        plt.show()

        fig.savefig(f"{folder}/{u}.2. Masked Image, Pos {scan_position}.png", dpi=200, bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(6,5))
        extent = [x_vec_c[0]*1e3, x_vec_c[-1]*1e3, y_vec_c[-1]*1e3, y_vec_c[0]*1e3]
        ax.imshow(image_masked, extent=extent, cmap='gray', vmin=-db_dyn, vmax=0, aspect='auto')
        ax.plot(xs_mm, ys_mm, 'r-', linewidth=1.0)
        ax.plot(peak_x_mm, peak_y_mm, 'ro', markersize=1)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('Depth [mm]')
        ax.set_title(f'Sizing, Pos: {scan_position}, Defect Size: {defect_length_mm:.2f}')
        plt.show()

        fig.savefig(f"{folder}/{u}.3. Sized Image, Pos {scan_position}.png", dpi=200, bbox_inches='tight')

    u += 1

#%%
# Create Image of side by side slices. 
# Parameters
mode = 'zn22608'
centre_freq = 2.25e6   # Hz
velocity = 6020.0     # m/s
wavelength = velocity / centre_freq

element_pitch = wavelength * 0.5
element_width = element_pitch - 0.05e-3
no_elements = 64                 # number of elements
first_element_position = 1e-3    # m from weld edge

# Imaging region
width_m  = 80e-3 
depth_m  = 70e-3 
pixel    = 0.25e-3

# Crop Area
x_min = -30e-3
x_max = 0
y_min = 30e-3
y_max = 60e-3

# Display Settings
db_dyn = 40.0 # dynamic range for image dB display
dt = 2e-8
B = 7

H_smooth = makeFilter(2414, dt, centre_freq)
scan_positions = np.arange(0, 301, 1)/1000

u = 0
diagonals = []
for scan_position in scan_positions:
    print(scan_position)

    # Ray tracing using transducer
    start = time.time()
    fmc_data, time_vec, element_positions = simScan(mode, scan_position, no_elements, element_pitch, element_width, 
                                                    first_element_position, centre_freq)
    end = time.time()
    process_time = end-start
    Tx, Rx, Nt = fmc_data.shape

    # Filtering Data
    analytic = filterData(fmc_data, H_smooth, Tx, Rx)

    # Full Image
    x_vec = np.arange(-width_m/2, width_m/2 + pixel, pixel)
    y_vec = np.arange(0.0, depth_m + pixel, pixel)
    X, Y  = np.meshgrid(x_vec, y_vec)

    image_tfm = TFM_loop(analytic, element_positions, X, Y, dt, no_elements, velocity, Rx, Nt)

    image_amp_tfm = np.abs(image_tfm.astype(np.complex64))
    image_db_tfm = 20.0 * np.log10(image_amp_tfm / (np.max(image_amp_tfm) + 1e-12))

    # Cropped Image
    x_vec_c, y_vec_c, image_db_c = image_crop(x_vec, y_vec, image_db_tfm, x_min, x_max, y_min, y_max)
    diag = extract_diagonal(image_db_c)
    diagonals.append(diag)

#%%
# Plot Diagonals
max_len = max(len(d) for d in diagonals)
diag_img = np.full((max_len, len(diagonals)), -100.0)

for i, d in enumerate(diagonals):
    diag_img[:len(d), i] = d

plt.figure(figsize=(12,6))
plt.imshow(diag_img, cmap="gray", aspect='auto', vmin=-db_dyn, vmax=0)
plt.xlabel("Scan Position [mm]")
plt.ylabel("Depth from Top of Fusion Face [mm]")
# plt.title("Weld Width Map (Diagonal Slices)")
plt.colorbar(label="Relative dB")
plt.show()

# fig.savefig(f"Images/Full Weld Length.png", dpi=300, bbox_inches='tight')

#%%
plt.rcParams['font.size'] = 14

plt.figure(figsize=(12,6))
plt.imshow(diag_img, cmap="gray", aspect='auto',
           vmin=-db_dyn, vmax=0,
           extent=[0, len(diagonals), 0, 30])  # y-axis 0→30
plt.xlabel("Scan Position [mm]")
plt.ylabel("Depth from Top of Fusion Face [mm]")
plt.colorbar(label="Relative dB")
plt.show()
