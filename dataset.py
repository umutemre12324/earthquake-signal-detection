import numpy as np 
import matplotlib.pyplot as plt 
import random
import os

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("data", exist_ok=True)

SAMPLES = 1500
NUM_SIGNAL = 1000


def generate_noise(num_signal, noise_level=0.05):
    return np.random.normal(0, noise_level, num_signal)


def generate_earthquake(samples, num_signal):
    t = np.linspace(0, 1, num_signal)

    center = np.random.uniform(0.3, 0.7)
    width = np.random.uniform(0.05, 0.15)

    envelope = np.exp(
        -((t - center) ** 2) / (2 * width ** 2)
    )

    frequency = np.random.uniform(5, 20)
    phase = np.random.uniform(0, 2 * np.pi)

    signal = np.sin(2 * np.pi * frequency * t + phase)

    earthquake_signal = envelope * signal

    return earthquake_signal


for i in range(SAMPLES):
    earthquake_signal = generate_earthquake(SAMPLES, NUM_SIGNAL)
    noise = generate_noise(NUM_SIGNAL, noise_level=0.05)
    noisy_signal = earthquake_signal + noise

    np.save(f"data/earthquake_{i}.npy", earthquake_signal)
    np.save(f"data/noise_{i}.npy", noise)
    np.save(f"data/noisy_earthquake_{i}.npy", noisy_signal)


t = np.linspace(0, 1, NUM_SIGNAL)

earthquake_signal = np.load("data/earthquake_0.npy")
noise = np.load("data/noise_0.npy")
noisy_signal = np.load("data/noisy_earthquake_0.npy")

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, earthquake_signal)
plt.title("Synthetic Earthquake Signal")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, noise)
plt.title("Gaussian Noise")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, noisy_signal)
plt.title("Earthquake Signal with Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()