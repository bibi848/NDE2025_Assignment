#%%
# Import Functions
import numpy as np
from scipy.signal import hilbert
import NDE_functions as nde
from scipy.fft import fft, ifft, fftfreq
from skimage import measure

def simScan(mode, scan_position, no_elements, element_pitch,
            element_width, first_element_position, centre_freq):
    
    fmc_data, time_vec, element_positions = nde.fn_simulate_data_weld_v5(
                                                                        mode,
                                                                        scan_position,
                                                                        no_elements,
                                                                        element_pitch,
                                                                        element_width,
                                                                        first_element_position,
                                                                        centre_freq
                                                                        )
    return fmc_data, time_vec, element_positions

def makeFilter(Nt, dt, centre_freq):
    freqs = fftfreq(Nt, dt)

    f_lo = 0.44 * centre_freq
    f_hi = 1.56 * centre_freq

    H = np.zeros_like(freqs, dtype=float)
    H[(np.abs(freqs) >= f_lo) & (np.abs(freqs) <= f_hi)] = 1.0

    trans_frac = 0.1
    f_trans = (f_hi - f_lo) * trans_frac
    low_start = f_lo - f_trans
    low_end = f_lo + f_trans
    high_start = f_hi - f_trans
    high_end = f_hi + f_trans

    # Build smooth H
    H_smooth = np.zeros_like(H)
    for i, fval in enumerate(freqs):
        af = abs(fval)
        if af <= low_start:
            val = 0.0
        elif low_start < af < low_end:
            x = (af - low_start) / (low_end - low_start)
            val = 0.5*(1 - np.cos(np.pi * x))
        elif low_end <= af <= high_start:
            val = 1.0
        elif high_start < af < high_end:
            x = (af - high_start) / (high_end - high_start)
            val = 0.5*(1 + np.cos(np.pi * x))
        else:
            val = 0.0
        H_smooth[i] = val

    return H_smooth

def filterData(fmc_data, H_smooth, Tx, Rx):

    fmc_filtered = np.zeros_like(fmc_data, dtype=complex)
    for ii in range(Tx):
        for jj in range(Rx):
            S = fft(fmc_data[ii, jj, :])
            Sf = S * H_smooth
            fmc_filtered[ii, jj, :] = ifft(Sf)

    analytic = hilbert(np.real(fmc_filtered), axis=2)

    return analytic

def TFM_loop(analytic, elem_x, X, Y, dt, no_elements, velocity, Rx, Nt):

    Ny, Nx = X.shape

    # Precompute receiver distances
    elem_x_rx = elem_x.astype(np.float32)
    r_r_all = np.sqrt((X[None, :, :] - elem_x_rx[:, None, None])**2 + (Y[None, :, :]**2)).astype(np.float32)

    image_tfm = np.zeros((Ny, Nx), dtype=np.complex64)
    rx_indices = np.arange(Rx, dtype=np.int32)

    for ii in range(no_elements):
        x_t = float(elem_x[ii])

        # Transmit distance to every pixel
        r_t = np.sqrt((X - x_t)**2 + Y**2).astype(np.float32)

        # Two-way travel time for all Rx
        tau = (r_t[None, :, :] + r_r_all) / float(velocity)

        # Fractional sample index
        idxf = tau / float(dt)                 # float
        k0 = np.floor(idxf).astype(np.int32)   # (Rx,Ny,Nx)
        alpha = (idxf - k0).astype(np.float32) # (Rx,Ny,Nx)

        # Clamp k0 to valid range for safe indexing
        k0_clamped = np.clip(k0, 0, Nt - 2)

        # Reshape to for vectorised indexing
        k0_flat = k0_clamped.reshape(Rx, -1)
        alpha_flat = alpha.reshape(Rx, -1)
        valid_flat = ((k0.reshape(Rx, -1) >= 0) & (k0.reshape(Rx, -1) < Nt-1))

        # Analytic trace
        s_tx = analytic[ii, :, :].astype(np.complex64)

        rx_idx = rx_indices[:, None]  
        s0 = s_tx[rx_idx, k0_flat]    
        s1 = s_tx[rx_idx, k0_flat + 1]

        # linear interpolation (complex)
        alpha_c = alpha_flat.astype(np.complex64)
        s_interp = (1.0 - alpha_c) * s0 + alpha_c * s1

        # Zero-out invalid positions
        s_interp[~valid_flat] = 0+0j

        # Sum over Rx
        img_vec = np.sum(s_interp, axis=0)

        # Accumulate into full image
        image_tfm += img_vec.reshape(Ny, Nx)

    return image_tfm

def image_crop(x_vec, y_vec, image_db_tfm, x_min, x_max, y_min, y_max):

    crop_x = (x_vec >= x_min) & (x_vec <= x_max)
    crop_y = (y_vec >= y_min) & (y_vec <= y_max)

    x_vec_c = x_vec[crop_x]
    y_vec_c = y_vec[crop_y]
    image_db_c = image_db_tfm[crop_y][:, crop_x]

    return x_vec_c, y_vec_c, image_db_c

def image_mask(B, pixel, image_db_c, x_vec_c, y_vec_c):

    Xc, Yc = np.meshgrid(x_vec_c, y_vec_c)

    x0, y0 = -30e-3, 30e-3
    x1, y1 =  0e-3, 60e-3
    vx, vy = x1 - x0, y1 - y0
    t = ((Xc - x0) * vx + (Yc - y0) * vy) / (vx*vx + vy*vy)

    t = np.clip(t, 0, 1)
    X_line = x0 + t * vx
    Y_line = y0 + t * vy

    dist = np.sqrt((Xc - X_line)**2 + (Yc - Y_line)**2)
    mask = dist <= (B*pixel)

    image_masked = image_db_c.copy()
    image_masked[~mask] = -100
    return image_masked

def contour_sizing(image_masked, x_vec_c, y_vec_c):
    x0, y0 = -30e-3, 30e-3
    x1, y1 =  0e-3, 60e-3

    peak_db = np.max(image_masked)
    peak_idx = np.unravel_index(np.argmax(image_masked), image_masked.shape)
    peak_y_mm = y_vec_c[peak_idx[0]] * 1e3
    peak_x_mm = x_vec_c[peak_idx[1]] * 1e3

    level_db = peak_db - 6.0

    lin_image = 10**(image_masked / 20.0)
    lin_level = 10**(level_db / 20.0)

    contours = measure.find_contours(lin_image, lin_level)

    if len(contours) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        contour = max(contours, key=lambda x: len(x))

        ys_mm = np.interp(contour[:, 0], np.arange(len(y_vec_c)), y_vec_c*1e3)
        xs_mm = np.interp(contour[:, 1], np.arange(len(x_vec_c)), x_vec_c*1e3)

        xs = xs_mm / 1e3
        ys = ys_mm / 1e3

        vx, vy = x1 - x0, y1 - y0
        L = np.sqrt(vx*vx + vy*vy)

        t_vals = ((xs - x0) * vx + (ys - y0) * vy) / (vx*vx + vy*vy)
        t_vals = np.clip(t_vals, 0, 1)

        s_vals = t_vals * L
        defect_length_mm = (np.max(s_vals) - np.min(s_vals)) * 1000

    return defect_length_mm, peak_db, level_db, xs_mm, ys_mm, peak_x_mm, peak_y_mm

def extract_diagonal(image):
    Ny, Nx = image.shape
    diag_len = min(Ny, Nx)
    return np.array([image[i, i] for i in range(diag_len)])