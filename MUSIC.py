import numpy.linalg as LA
from constants import d_res, Measurement_rate, lambda_radar, SoL, ADC_sample_rate, S, f0, slow_sampling_rate, v_max
from scipy.signal import find_peaks
from utility import cluster_and_find_centroids, cov_matrix, adaptive_peak_detection, estimate_distances, \
    perform_hdbscan_clustering, cfar_ca_1d, apply_CFAR_2D, doppler_to_speed,compute_stft, detect_micro_doppler
from CFAR import *
import numpy as np

def _noise_subspace(covariance, num_sources):
    """helper function to get noise_subspace.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix should be a 2D square matrix.")
    if num_sources >= covariance.shape[0]:
        raise ValueError("number of sources should be less than number of receivers.")
    _, v = LA.eigh(covariance)

    return v[:, :-num_sources]


def aoa_music_1D(steering_vec, rx_chirps, num_sources):
    r"""
    Implmentation of 1D MUltiple SIgnal Classification (MUSIC) algorithm on ULA (Uniformed Linear Array).

    Current implementation assumes covariance matrix is not rank deficient and ULA spacing is half of the wavelength.
    math::
        P_{} (\\theta) = \\frac{1}{a^{H}(\\theta) \mathbf{E}_\mathrm{n}\mathbf{E}_\mathrm{n}^H a(\\theta)}
    where :math:`E_{n}` is the noise subpace and :math:`a` is the steering vector.


    Args:
        steering_vec (~np.ndarray): steering vector with the shape of (FoV/angel_resolution, num_ant).
         FoV/angel_resolution is usually 181. It is generated from gen_steering_vec() function.
        rx_chirps (~np.ndarray): Output of the 1D range FFT. The shape is (num_ant, num_chirps_per_frame).
        num_sources (int): Number of sources in the scene. Needs to be smaller than num_ant for ULA.

    Returns:
        (~np.ndarray): the spectrum of the MUSIC. Objects should be holes for the equation and thus sharp peaks.
    """
    num_antennas = rx_chirps.shape[0]
    assert num_antennas == steering_vec.shape[1], "Mismatch between number of receivers in "
    if num_antennas < num_sources:
        raise ValueError("number of sources shoule not exceed number ")

    R = cov_matrix(rx_chirps)
    noise_subspace = _noise_subspace(R, num_sources)
    v = noise_subspace.T.conj() @ steering_vec.T
    spectrum = np.reciprocal(np.sum(v * v.conj(), axis=0).real)

    return spectrum


def detect_human_and_compute_music(doppler_fft, range_fft, steering_vec, num_sources, angle_grid, debug_music=False):
    """
    For each frame, detect targets using both CFAR and range estimation, then detect micro-Doppler effects via STFT,
    compute target speed using Doppler bin conversion, and calculate the angle using the MUSIC algorithm.
    The detection results output the distance, speed, and angle information.

    Parameters:
      doppler_fft: Doppler FFT result, shape (num_frames, num_chirps, num_bins, num_antennas)
      range_fft: Range FFT result, same shape as doppler_fft
      steering_vec: Steering vector used for the MUSIC algorithm
      num_sources: Number of targets for the MUSIC algorithm
      angle_grid: Angle search grid
      debug_music: Whether to debug the MUSIC algorithm (if debugging, plot the angle spectrum and output data)
    """
    num_frames, num_chirps, num_bins, num_antennas = range_fft.shape
    angle_distance_results = []
    centroids_results = []
    threshold_factor = 8  # Threshold factor for the range estimation function

    for frame_idx in range(num_frames):
        # --- CFAR Detection ---
        # Average over the antenna dimension to obtain a 2D power map (size: num_chirps x num_bins)
        power_map = np.mean(np.abs(doppler_fft[frame_idx, :, :, :]) ** 2, axis=-1)
        cfar_detections = apply_CFAR_2D(power_map, guard_size=9, ref_size=16, multiplier=15, return_mask=True)

        # --- Range Estimation ---
        averaged_distance_matrix, peak_indices_list = estimate_distances(
            range_fft[frame_idx:frame_idx + 1, :, :, :],
            threshold_factor
        )
        # Process only one frame, take the first result
        estimated_distances = averaged_distance_matrix[0]  # shape: (num_bins,)
        peak_indices = peak_indices_list[0]  # Peak indices obtained from range estimation

        frame_results = []
        # Get 2D indices (slow time, range bin) detected by CFAR
        detection_indices = np.argwhere(cfar_detections)
        for slow_idx, range_bin in detection_indices:
            # Process only targets within the range estimation peaks
            if range_bin in peak_indices:
                # Extract the slow-time signal for the current range bin (average over the antenna dimension)
                slow_signal = np.mean(doppler_fft[frame_idx, :, range_bin, :], axis=-1)

                # Compute STFT and detect micro-Doppler effects
                f_stft, t_stft, Zxx = compute_stft(slow_signal, 70, nperseg=16, noverlap=8)
                detection_mask = detect_micro_doppler(Zxx)

                # If significant energy is detected in the STFT time-frequency plane,
                # it is assumed that a human is moving.
                if np.any(detection_mask):
                    detected_distance = estimated_distances[range_bin]
                    detected_speed = doppler_to_speed(slow_idx, num_chirps, v_max)

                    # Use all chirp data in the current range bin for angle estimation (transpose to suit the MUSIC algorithm)
                    rx_chirps = range_fft[frame_idx, :, range_bin, :].T
                    angle_spectrum = aoa_music_1D(steering_vec, rx_chirps, num_sources)
                    max_peak_idx = np.argmax(angle_spectrum)
                    best_angle = angle_grid[max_peak_idx]
                    frame_results.append((detected_distance, best_angle))
        angle_distance_results.append(frame_results)

        # Cluster the targets detected in the current frame
        if len(frame_results) > 0:
            detections = np.array(frame_results)
        else:
            detections = np.empty((0, 3))
        centroids = perform_hdbscan_clustering(detections, frame_idx)
        centroids_results.append(centroids)

    return angle_distance_results, centroids_results
















