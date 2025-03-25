import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from MUSIC import detect_human_and_compute_music
from constants import d_res, ADC_sample_rate, Num_doppler_fft_bins, v_max, SoL, Num_range_fft_bins, S
from utility import hann_window_1st_FFT, first_fft, second_fft, \
    hann_window_2nd_FFT, hann_window_3rd_FFT, third_fft, estimate_distances, adaptive_peak_detection, gen_steering_vec, \
    wavelet_denoise, plot_chirps
import warnings

# File path
file_name = "E:\\adc_data\\4_1TX4RX.pkl"

# Read radar data
adc_matrix = np.load(file_name)  # Frames x Chirps x Samples x Antennas
n_adc_samples = adc_matrix.shape[2]
# Only use the first 500 frames
num_frames_to_use = 500
adc_matrix = adc_matrix[:num_frames_to_use]

# Below is for testing the plots on SDR
plt.plot(np.arange(n_adc_samples) * 1 / ADC_sample_rate, adc_matrix[0, 0, :, 0].real, 'b')
plt.plot(np.arange(n_adc_samples) * 1 / ADC_sample_rate, adc_matrix[0, 0, :, 0].imag, 'r')

# Data preprocessing
adc_matrix = hann_window_1st_FFT(adc_matrix)
range_fft = first_fft(adc_matrix, shift=False)  # First FFT
# range_fft_denoise = wavelet_denoise(range_fft)

# Plot first FFT
plot_chirps(np.abs(range_fft[np.arange(0, 300, 100), 2, :, 0]), "Range FFT (Wavelet)")

range_fft = hann_window_2nd_FFT(range_fft)
doppler_fft = second_fft(range_fft, shift=True)  # Second FFT
# doppler_fft_denoise = wavelet_denoise(doppler_fft)
num_frames, num_chirps, num_bins, num_antennas = range_fft.shape

# Reshape Doppler FFT for visualization
doppler_fft_reshape = np.transpose(doppler_fft, (0, 2, 1, 3))
doppler_fft_reshape = doppler_fft_reshape[:, ::-1, :, :]

magnitude_dBFS = 20 * (np.log10(np.abs(doppler_fft_reshape)) - 10)
velocity_axis = np.linspace(-v_max, v_max, Num_doppler_fft_bins)
distance_axis = np.linspace(0, ADC_sample_rate * SoL / (2 * S), Num_range_fft_bins)

fig, ax = plt.subplots()

def animate(i):
    ax.clear()  # Clear the previous frame
    ax.imshow(magnitude_dBFS[i, :, :, 1],
              extent=[velocity_axis[0], velocity_axis[-1], distance_axis[0], distance_axis[-1]],
              aspect='auto', cmap='jet')
    ax.set_xlabel('Velocity - meters/sec')
    ax.set_ylabel('Distance - meters')
    ax.set_title('2D FFT amplitude profile - Frame {}'.format(i+1))

ani = FuncAnimation(fig, animate, frames=magnitude_dBFS.shape[0], interval=10)
ani.save("E:/CE301/animation.mp4", writer=animation.FFMpegWriter(fps=50))
plt.show()

# Set MUSIC parameters
num_sources = 3
num_vec, steering_vec = gen_steering_vec(90, 1, num_antennas)
angles = np.linspace(-90, 90, num_vec)

# Use the MUSIC algorithm to detect humans and compute angle, distance, etc.
results, centroids = detect_human_and_compute_music(doppler_fft, range_fft, steering_vec, num_sources, angles)

def animate_fitted_centroids(centroids_results, poly_degree=15):
    """
    Dynamically plot data after polynomial fitting.
    Here, centroids_results is a list, where the centroids for each frame are in the format [(distance, angle), ...].
    """
    frames_all = []
    distances = []
    angles = []
    for i, centroids in enumerate(centroids_results):
        if centroids:
            for d, a in centroids:
                frames_all.append(i)
                distances.append(float(d))
                angles.append(float(a))

    frames_all = np.array(frames_all)
    distances = np.array(distances)
    angles = np.array(angles)

    if len(frames_all) == 0:
        print("No data detected, unable to perform fitting.")
        return

    # Perform polynomial fitting separately for distance and angle
    warnings.simplefilter('ignore', np.RankWarning)
    coeffs_distance = np.polyfit(frames_all, distances, poly_degree)
    coeffs_angle = np.polyfit(frames_all, angles, poly_degree)

    # Generate fitted data over the entire frame range
    frame_range = np.linspace(frames_all.min(), frames_all.max(), num_frames_to_use)
    angle_gain = 3
    fitted_distance = np.polyval(coeffs_distance, frame_range)
    fitted_angle = np.polyval(coeffs_angle, frame_range)

    angles_rad = np.deg2rad(fitted_angle) * angle_gain
    x_coords = fitted_distance * np.sin(angles_rad)
    y_coords = np.clip(np.abs(fitted_distance * np.cos(angles_rad)), 1.8, 2.2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_title("Human Localization - Fitted Centroids")
    ax.legend(["Human"], loc="upper right")

    # Initialize animated scatter plot using the "rainbow" colormap
    scatter = ax.scatter([], [], c=[], cmap="rainbow", s=100, edgecolors="black", alpha=0.8)

    def update(frame_idx):
        """
        Animation update function: update the position and color of the scatter plot for each frame.
        """
        # Corresponds to a single point in the fitted data
        x = x_coords[frame_idx]
        y = y_coords[frame_idx]
        scatter.set_offsets(np.array([[x, y]]))
        # Color mapping: normalized by the frame number
        scatter.set_array(np.array([frame_idx / len(frame_range)]))
        ax.set_title(f"Human Localization - Frame {frame_idx + 1}")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(frame_range), interval=500, repeat=True)
    ani.save("E:/CE301/detection_fitted.mp4",animation.FFMpegWriter(fps=50, bitrate=6000))
    plt.show()

animate_fitted_centroids(centroids, poly_degree=30)
