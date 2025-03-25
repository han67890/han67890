from matplotlib.animation import FuncAnimation
from numpy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, lfilter
from constants import *
import pywt
from sklearn.cluster import HDBSCAN
from scipy.signal import butter, filtfilt
from scipy.signal import detrend
import warnings
from scipy.signal import stft
import hdbscan

def detrend_signals(signals):   #----去除信号的趋势项（Detrending），使信号在时间上没有全局趋势或漂移----
    """
    Detrend a 2D array of signals.

    Parameters:
    signals : array_like
        A 2D array where each row represents a signal.

    Returns:
    detrended_signals : array_like
        A 2D array of detrended signals.
    """
    detrended_signals = detrend(signals, axis=1)  # Assuming each row is a signal
    return detrended_signals


def normalize_signals(signals):
    """
    Normalize an array of signals to have zero mean and unit variance.

    Parameters:
    signals : array_like
        A 2D array where each row represents a signal.

    Returns:
    normalized_signals : array_like
        A 2D array of normalized signals.
    """
    # Subtract the mean from each signal
    signals_mean_subtracted = signals - np.mean(signals)

    # Divide by the standard deviation
    normalized_signals = signals_mean_subtracted / np.std(signals_mean_subtracted)

    return normalized_signals

def bandpass_filter(signal, fs, lowcut, highcut):
    """
    Apply a band-pass filter to a signal.

    Parameters:
    signal : array_like
        Input signal to be filtered.
    fs : float
        Sampling frequency of the signal.
    lowcut : float
        Low cutoff frequency of the filter.
    highcut : float
        High cutoff frequency of the filter.

    Returns:
    filtered_signal : array_like
        The band-pass filtered signal.
    """
    # Normalize the frequencies by the Nyquist frequency
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Generate the filter coefficients
    b, a = butter(N=4, Wn=[low, high], btype='band', analog=False)

    # Apply the filter
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def plot_chirps(data: np.ndarray, title: str):
    # fr_idx = fftfreq(n_adc_samples, d=1 / n_adc_samples)
    num_chirps = data.shape[0]
    fig, axes = plt.subplots(num_chirps, 1, figsize=(6, 3 * num_chirps))

    for chirp_idx, ax in enumerate(axes):
        # ax.plot(fr_idx, data[chirp_idx, :])
        ax.plot(data[chirp_idx, :])
        ax.set_title(f"Chirp {chirp_idx} {title}")
        ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()

def cov_matrix(x):
    """ Calculates the spatial covariance matrix (Rxx) for a given set of input data (x=inputData).
        Assumes rows denote Vrx axis.

    Args:
        x (ndarray): A 2D-Array with shape (rx, adc_samples) slice of the output of the 1D range fft

    Returns:
        Rxx (ndarray): A 2D-Array with shape (rx, rx)
    """

    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")

    if x.shape[0] > x.shape[1]:
        warnings.warn("cov_matrix input should have Vrx as rows. Needs to be transposed", RuntimeWarning)
        x = x.T

    _, num_adc_samples = x.shape
    Rxx = x @ np.conjugate(x.T)
    Rxx = np.divide(Rxx, num_adc_samples)

    return Rxx

def band_pass_filter(data, sampling_frequency: float, low_cut: float, high_cut: float):
    # Design a Butterworth band-pass filter
    b, a = butter(N=4, Wn=[low_cut, high_cut], btype='band', fs=sampling_frequency)

    filtered_data_real = np.zeros_like(data, dtype=np.float32)
    filtered_data_imag = np.zeros_like(data, dtype=np.float32)
    for frm in range(data.shape[0]):
        for chp in range(data.shape[1]):
            for lan in range(data.shape[3]):
                filtered_data_real[frm, chp, :, lan] = lfilter(b, a, data[frm, chp, :, lan].real)
                filtered_data_imag[frm, chp, :, lan] = lfilter(b, a, data[frm, chp, :, lan].imag)

    return filtered_data_real + 1j * filtered_data_imag

def HYH_butter_lowpass(data, cutoff, order=5):
    normalized_freq = max_IF / (0.5 * ADC_sample_rate)
    b, a = butter(order, normalized_freq, btype='low', analog=False)

    filtered_data_real = np.zeros_like(data, dtype=np.float32)
    filtered_data_imag = np.zeros_like(data, dtype=np.float32)

    for frm in range(data.shape[0]):
        for chp in range(data.shape[1]):
            for lan in range(data.shape[3]):
                filtered_data_real[frm, chp, :, lan] = lfilter(b, a, data[frm, chp, :, lan].real)
                filtered_data_imag[frm, chp, :, lan] = lfilter(b, a, data[frm, chp, :, lan].imag)

    return filtered_data_real + 1j * filtered_data_imag


def hann_window_1st_FFT(data):
    return data * np.hanning(data.shape[-2])[np.newaxis, np.newaxis, :, np.newaxis]

def hann_window_2nd_FFT(data):
    return data * np.hanning(data.shape[-3])[np.newaxis, :, np.newaxis, np.newaxis]

def hann_window_3rd_FFT(data):
    return data * np.hanning(data.shape[-1])[np.newaxis, np.newaxis, np.newaxis, :]

def first_fft(adc_matrix, shift: bool = True):
    range_fft = fft(adc_matrix, n=Num_range_fft_bins, axis=2)
    if shift:
        range_fft = fftshift(range_fft, axes=2)
    return range_fft

def second_fft(range_fft, shift: bool = True):
    doppler_fft = fft(range_fft, n=Num_doppler_fft_bins, axis=1)
    if shift:
        doppler_fft = fftshift(doppler_fft, axes=1)
    return doppler_fft

def third_fft(doppler_fft, shift: bool = True):
    angle_fft = fft(doppler_fft, axis=-1)
    if shift:
        angle_fft = fftshift(angle_fft, axes=-1)
    return angle_fft

def adjust_format(data):
    data = np.transpose(data, (0, 2, 1, 3))
    data = data[:, ::-1, :, :]
    return data

def cfar_ca_1d(data, num_guard_cells, num_ref_cells, rate_fa):
    num_cells = len(data)
    thresholds = np.zeros(num_cells)
    cfar_signals = np.zeros(num_cells, dtype=bool)

    num_total_cells = num_guard_cells + num_ref_cells

    for i in range(num_total_cells, num_cells - num_total_cells):
        # 取前导参考单元和后导参考单元
        leading_cells = data[i - num_total_cells:i - num_guard_cells]
        trailing_cells = data[i + num_guard_cells + 1:i + num_total_cells + 1]

        # 合并参考单元，并计算均值作为噪声估计
        ref_cells = np.concatenate((leading_cells, trailing_cells))
        noise_level = np.mean(ref_cells)
        threshold = noise_level * rate_fa
        thresholds[i] = threshold

        if data[i] > threshold:
            cfar_signals[i] = True

    return cfar_signals, thresholds


def cluster_and_find_centroids(detections, min_samples=2):     #----多个检测点可能属于同一目标。通过空间聚类找到目标的位置。----
    """
    Cluster detected peaks and find centroids of clusters.

    :param detections: 2D array of detection flags (True if target, False otherwise).
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Array of cluster centroids.
    """
    # Extract indices of detected points
    points = np.argwhere(detections)
    min_samples = min(min_samples, points.shape[0])

    if min_samples <= 1:
        return np.array([]), np.array([]), np.array([])

    # Apply DBSCAN clustering
    db = HDBSCAN(min_samples=min_samples).fit(points)
    labels = db.labels_

    # Filter out noise (-1 label)
    clustered_points = points[labels != -1]
    labels = labels[labels != -1]

    # Calculate centroids of clusters
    unique_labels = set(labels)
    centroids = np.array([clustered_points[labels == k].mean(axis=0) for k in unique_labels])

    return centroids, clustered_points, labels


def wavelet_denoise(data, thresh=0.4, wavelet="db4"):     #----小波分析去噪----
    filtered_data = np.zeros_like(data, dtype=np.complex64)

    for frm in range(data.shape[0]):
        for chp in range(data.shape[1]):
            for lan in range(data.shape[3]):
                # Thresholding
                t_real = thresh * np.nanmax(data[frm, chp, :, lan].real)
                t_imag = thresh * np.nanmax(data[frm, chp, :, lan].imag)

                coefficients_real = pywt.wavedec(data[frm, chp, :, lan].real, wavelet, mode="per")
                coefficients_real[1:] = (pywt.threshold(i, value=t_real, mode="soft") for i in coefficients_real[1:])

                coefficients_imag = pywt.wavedec(data[frm, chp, :, lan].imag, wavelet, mode="per")
                coefficients_imag[1:] = (pywt.threshold(i, value=t_imag, mode="soft") for i in coefficients_imag[1:])

                # Reconstruct the signal using the thresholded coefficients
                filtered_data[frm, chp, :, lan] = (
                    pywt.waverec(coefficients_real, wavelet, mode='per') +
                    1j * pywt.waverec(coefficients_imag, wavelet, mode='per')
                )

    return filtered_data

def estimate_distances_and_velocities(range_fft_data, doppler_fft_data):   #----得到距离和速度----
    distances_and_velocities = []

    # Distance estimation
    n_adc_samples = range_fft_data.shape[2]
    frequencies_range_fft = fftfreq(n_adc_samples, d=1 / ADC_sample_rate)
    distance_mask = np.array([(SoL * frequency) / (2 * S) for frequency in frequencies_range_fft])

    # Velocity estimation
    n_chirps = doppler_fft_data.shape[1]
    frequencies_doppler_fft = np.arange(n_chirps) * (slow_sampling_rate / n_chirps)
    velocity_mask = np.array([(SoL * f_i) / (2 * (f0 + f_i)) for f_i in frequencies_doppler_fft])

    for frame in range(range_fft_data.shape[0]):
        print(f'Processing frame: {frame + 1}', end='\r', flush=True)
        for chirp in range(range_fft_data.shape[1]):
            for lane in range(range_fft_data.shape[3]):
                peak_indexes_distances = adaptive_peak_detection(np.abs(range_fft_data[frame, chirp, :, lane]), 3)

                # TODO: Not working the velocity estimation
                for possible_object_index in peak_indexes_distances:
                    peak_indexes_velocities = adaptive_peak_detection(np.abs(doppler_fft_data[frame, :, possible_object_index, lane]), 1.5)
                    objects_velocities = velocity_mask[peak_indexes_velocities]
                    objects_distances = distance_mask[possible_object_index]
                    result = {"distance": objects_distances,
                              "velocity": objects_velocities,
                              "peak_distance_indices":peak_indexes_distances,
                              "peak_velocity_indices":peak_indexes_velocities}

                    distances_and_velocities.append(result)

    return distances_and_velocities


def adaptive_peak_detection(data, threshold_factor: float = 1.5):      #----自适应峰值检测----
    mean = np.mean(data)
    std = np.std(data)
    threshold = mean + threshold_factor * std
    peaks_indexes, _ = find_peaks(data, height=threshold)
    if len(peaks_indexes) == 0:
        max_index = np.argmax(data)
        peaks_indexes = np.array([max_index])
    return peaks_indexes


def estimate_distances(range_fft_data, threshold_factor):
    fft_bins = range_fft_data.shape[2]
    frequencies = np.arange(fft_bins) / fft_bins * ADC_sample_rate
    distance_mask = np.array([(SoL * frequency) / (2 * S) for frequency in frequencies])

    distance_matrix = np.zeros_like(range_fft_data, dtype=np.float32)

    for frame in range(range_fft_data.shape[0]):
        for chirp in range(range_fft_data.shape[1]):
            for lane in range(range_fft_data.shape[3]):
                peak_indexes = adaptive_peak_detection(np.abs(range_fft_data[frame, chirp, :, lane]), threshold_factor)
                distance_matrix[frame, chirp, peak_indexes, lane] = distance_mask[peak_indexes]

    # Average across all chirps and lanes, then detect peaks in the averaged data
    averaged_distance_matrix = np.mean(distance_matrix, axis=(1, 3))
    peak_indices = [adaptive_peak_detection(frame_distances, threshold_factor) for frame_distances in averaged_distance_matrix]

    return averaged_distance_matrix, peak_indices

def doppler_to_speed(doppler_bin, num_doppler_bins, v_max):

    return (doppler_bin - num_doppler_bins / 2) / (num_doppler_bins / 2) * v_max

def apply_CFAR_2D(matrix, guard_size, ref_size, multiplier=10, return_mask=False):
    """
    Apply a simplified CFAR detection to a 2D matrix.

    :param matrix: 2D numpy array representing the matrix.
    :param guard_size: Size of the guard cells around the COI.
    :param ref_size: Size of the reference cells around the guard cells.
    :param multiplier: Multiplier for the threshold based on noise estimate.
    :param return_mask: Boolean indicating if the function should return a mask instead of the COI value.
    :return: A 2D mask (if return_mask=True) with True indicating detected targets.
    """
    rows, cols = matrix.shape
    peaks = np.zeros_like(matrix, dtype=bool if return_mask else float)
    for target_row in range(rows):
        for target_col in range(cols):
            row_start = max(0, target_row - guard_size - ref_size)
            row_end = min(rows, target_row + guard_size + ref_size + 1)
            col_start = max(0, target_col - guard_size - ref_size)
            col_end = min(cols, target_col + guard_size + ref_size + 1)

            reference_window = matrix[row_start:row_end, col_start:col_end]

            guard_row_start = max(0, target_row - guard_size) - row_start
            guard_row_end = min(rows, target_row + guard_size + 1) - row_start
            guard_col_start = max(0, target_col - guard_size) - col_start
            guard_col_end = min(cols, target_col + guard_size + 1) - col_start

            noise_window = np.copy(reference_window)

            noise_window[guard_row_start:guard_row_end, guard_col_start:guard_col_end] = 0

            num_noise_cells = noise_window.size - (guard_row_end - guard_row_start) * (guard_col_end - guard_col_start)

            noise_estimate = np.sum(noise_window) / num_noise_cells if num_noise_cells > 0 else 0
            threshold = noise_estimate * multiplier
            COI_value = matrix[target_row, target_col]

            if COI_value > threshold:
                peaks[target_row, target_col] = True if return_mask else COI_value
    return peaks


def gen_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Generate a steering vector for AOA estimation given the theta range, theta resolution, and number of antennas

    Defines a method for generating steering vector data input --Python optimized Matrix format
    The generated steering vector will span from -angEstRange to angEstRange with increments of ang_est_resolution
    The generated steering vector should be used for all further AOA estimations (bartlett/capon)

    Args:
        ang_est_range (int): The desired span of thetas for the angle spectrum.
        ang_est_resolution (float): The desired resolution in terms of theta
        num_ant (int): The number of Vrx antenna signals captured in the RDC

    Returns:
        num_vec (int): Number of vectors generated (integer divide angEstRange/ang_est_resolution)
        steering_vectors (ndarray): The generated 2D-array steering vector of size (num_vec,num_ant)

    Example:
        >>> #This will generate a numpy array containing the steering vector with
        >>> #angular span from -90 to 90 in increments of 1 degree for a 4 Vrx platform
        >>> _, steering_vec = gen_steering_vec(90,1,4)

    """
    num_vec = (2 * ang_est_range / ang_est_resolution + 1)
    num_vec = int(round(num_vec))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex64')
    for kk in range(num_vec):
        for jj in range(num_ant):
            mag = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            real = np.cos(mag)
            imag = np.sin(mag)

            steering_vectors[kk, jj] = complex(real, imag)

    return [num_vec, steering_vectors]


def perform_hdbscan_clustering(detections, frame_idx):
    """
    Perform HDBSCAN clustering. If HDBSCAN fails, use forced clustering by computing
    the mean distance and angle of all points.

    Parameters:
    - detections: np.array, shape (N, 2), each row represents (distance, angle).
    - frame_idx: int, index of the current frame.
    - cluster_radius: float, reserved parameter (currently unused).

    Returns:
    - centroids: list, each cluster's (mean distance, mean angle).
    """
    if len(detections) == 0:
        print(f"Frame {frame_idx + 1}: No detections available.")
        return []

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5)
    labels = clusterer.fit_predict(detections)

    # If all points are noise (-1), use forced clustering: compute the mean distance and angle of all detections
    if np.all(labels == -1):
        avg_distance = np.mean(detections[:, 0])
        avg_angle = np.mean(detections[:, 1])
        forced_centroid = (avg_distance, avg_angle)
        return [forced_centroid]

    # HDBSCAN clustering succeeded; compute centroids for each cluster (ignoring noise points)
    centroids = []
    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id != -1:
            cluster_points = detections[labels == cluster_id]
            centroid_distance = np.mean(cluster_points[:, 0])
            centroid_angle = np.mean(cluster_points[:, 1])
            centroids.append((centroid_distance, centroid_angle))
    return centroids


def compute_stft(signal, fs=70, nperseg=256, noverlap=128):
    """
    Compute the Short-Time Fourier Transform (STFT) of the signal.

    Parameters:
      signal: Input radar signal (can be a complex signal).
      fs: Sampling rate (Hz).
      nperseg: Number of samples per STFT window.
      noverlap: Number of overlapping samples between windows.

    Returns:
      f: Frequency axis array.
      t: Time axis array.
      Zxx: Complex STFT result.
    """
    f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    return f, t, Zxx


def detect_micro_doppler(Zxx):
    """
    Detect micro-Doppler signals in the STFT using power thresholding.

    Parameters:
      Zxx: STFT result (complex matrix).
      power_threshold: Power threshold to distinguish significant micro-Doppler components.

    Returns:
      detection_mask: Boolean matrix of the same shape as Zxx, where True indicates detection of strong energy.
    """
    power = np.abs(Zxx) ** 2
    # Compute global average power
    avg_power = np.mean(power)
    # If needed, multiply by a factor to adjust detection sensitivity
    threshold = avg_power * 3  # The factor can be adjusted to be greater or less than 1

    # Apply threshold detection
    detection_mask = power > threshold
    return detection_mask
