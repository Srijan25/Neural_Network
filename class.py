import numpy as np
import matplotlib.pyplot as plt

# Function to generate a wave
def generate_wave(amplitude, frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, wave

# Wave parameters
amplitudes = [10, 3, 5]
frequencies = [50, 100, 150]
duration = 0.1  # Duration of the waves in seconds
sample_rate = 1000  # Number of samples per second

# Generate the three waves
waves = [generate_wave(amplitude, frequency, duration, sample_rate) for amplitude, frequency in zip(amplitudes, frequencies)]

# Add the three waves to create a single wave
t_combined = waves[0][0]  # Use the time values from the first wave
wave_combined = sum(wave for _, wave in waves)

# Plot the three individual waves and the combined wave
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.title("Three Waves with Different Amplitudes and Frequencies")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
for i, (t, wave) in enumerate(waves):
    plt.plot(t, wave, label=f"Wave {i+1} (Amp: {amplitudes[i]}, Freq: {frequencies[i]})")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.title("Combined Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t_combined, wave_combined, label="Combined Wave")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#now implememt the fourier transform

# Compute the Fast Fourier Transform (FFT)
fft_combined = np.fft.fft(wave_combined)

# Compute the sample frequencies and the corresponding values in hertz
sample_freq = np.fft.fftfreq(len(fft_combined), 1 / sample_rate)
hz_values = sample_freq[:int(len(fft_combined) / 2) + 1]

# Plot the computed FFT
plt.figure(figsize=(12, 4))
plt.title("FFT of Combined Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.plot(hz_values, np.abs(fft_combined[:int(len(fft_combined) / 2) + 1]))
plt.grid()
plt.show()
