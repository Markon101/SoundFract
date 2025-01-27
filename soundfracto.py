import time
import secrets
import wave
import math
import numpy as np

# ======== GLOBAL CONTROLS ========
class Config:
    SAMPLE_RATE = 48000               # High-resolution audio rate
    DURATION = 30                     # Seconds
    OUTPUT_FILE = "neuro_noise.wav"
    
    # Frequency controls
    BASE_FREQ = 40.0                  # Hz (sub-bass range)
    FREQ_VARIANCE = 0.69              # Chaos frequency modulation depth
    FILTER_ORDER = 1                  # We'll do a simple first-order IIR
    FILTER_CUTOFF_HZ = 8000.0         # We'll manually control the lowpass cutoff
    
    # Fractal parameters
    FRACTAL_LAYERS = 8                # Number of noise octaves
    FRACTAL_PERSISTENCE = 0.314     # Amplitude decay per octave
    
    # Chaos system parameters
    CHAOS_R = 3.9999                  # Logistic map complexity
    ENTROPY_MODULATION = 0.81         # Chaos-to-sound modulation depth

# ======== ENTROPY GENERATOR ========
class HighResEntropyGenerator:
    def __init__(self, sample_rate, duration):
        self.num_samples = int(sample_rate * duration)
        self.entropy = np.zeros(self.num_samples)
        self.seed_time = time.time_ns()  # Nanosecond precision

    def _nanosecond_seed(self):
        """Create initial seed from nanosecond timestamp"""
        return (self.seed_time / 1e9) % 1.0

    def logistic_map(self, r=3.99):
        """Generate entropy values using a chaotic system with nanosecond seeding"""
        x = self._nanosecond_seed()
        for i in range(self.num_samples):
            x = r * x * (1 - x)
            self.entropy[i] = x
        return self.entropy

# ======== SECURE NOISE GENERATOR ========
class SecureNoiseGenerator:
    @staticmethod
    def _secure_float_array(num_samples):
        """Generate random float64 array in [-1, 1] using secrets."""
        raw_bytes = secrets.token_bytes(num_samples * 4)
        int32_array = np.frombuffer(raw_bytes, dtype=np.int32)
        # Convert to float range [-1,1], be mindful of zero division possibility:
        return int32_array.astype(np.float64) / (2**31 - 1)

    @classmethod
    def white_noise(cls, num_samples):
        return cls._secure_float_array(num_samples)

    @classmethod
    def blue_noise(cls, num_samples):
        """
        "Blue noise" by integrating white noise (cumulative sum),
        normalized by sqrt of sample index.
        """
        white = cls.white_noise(num_samples)
        # Avoid dividing by zero at the start
        denom = np.sqrt(np.arange(1, num_samples+1))
        return np.cumsum(white) / denom

    @classmethod
    def brownian_noise(cls, num_samples):
        """
        Brownian (red) noise: cumulative sum of white noise,
        then normalized to [-1,1].
        """
        increments = cls._secure_float_array(num_samples)
        noise = np.cumsum(increments)
        return noise / np.max(np.abs(noise))  # normalize

# ======== FRACTAL NOISE (SINGLE-THREADED) ========
def fractal_noise(num_samples):
    """
    Builds fractal noise by summing multiple 'blue noise' layers
    at different octaves, decimating each layer, and scaling amplitude.
    """
    noise_sum = np.zeros(num_samples, dtype=np.float64)
    for octave in range(Config.FRACTAL_LAYERS):
        # Generate a larger array for this octave
        scaled_len = num_samples * (2 ** octave)
        layer = SecureNoiseGenerator.blue_noise(scaled_len)
        # Decimate by factor of 2^octave
        decimated = layer[::(2**octave)]
        decimated = decimated[:num_samples]  # ensure same length
        # Scale amplitude by fractal persistence
        noise_sum += decimated * (Config.FRACTAL_PERSISTENCE ** octave)
    # Average across layers
    return noise_sum / Config.FRACTAL_LAYERS

# ======== CHAOS ENVELOPER ========
class ChaoticEnveloper:
    def __init__(self):
        self.entropy = HighResEntropyGenerator(
            Config.SAMPLE_RATE, 
            Config.DURATION
        ).logistic_map(Config.CHAOS_R)
        
    def dynamic_params(self):
        """
        Produce time-dependent parameters from chaotic entropy array.
        For simplicity, we'll just produce stereo_width & gain_env here.
        """
        # e is in [0,1], so map it suitably:
        stereo_width = 0.5 + 0.5 * np.cos(3 * self.entropy)  # vary from 0 to 1
        gain_env = 0.8 + 0.2 * (1 - self.entropy)  # vary from 0.8 to 1.0
        return stereo_width, gain_env

# ======== SIMPLE FIRST-ORDER LOWPASS FILTER ========
def lowpass_filter(data, cutoff_hz, sr):
    """
    Naive first-order low-pass filter (IIR).
    out[n] = out[n-1] + alpha*(in[n] - out[n-1])
    alpha = dt / (RC + dt), where RC = 1/(2*pi*cutoff)
    """
    dt = 1.0 / sr
    if cutoff_hz <= 0:
        return data  # no filtering if cutoff is invalid
    
    RC = 1.0 / (2 * math.pi * cutoff_hz)
    alpha = dt / (RC + dt)

    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = filtered[i-1] + alpha * (data[i] - filtered[i-1])
    return filtered

# ======== FREQUENCY SHIFT (PHASE MODULATION) ========
def frequency_shift(data, base_freq, entropy, sr):
    """
    Simple approach:
      freq_env(t) = base_freq * [1 + FREQ_VARIANCE*sin(0.5*2πt + 10*entropy)]
    Then accumulate freq_env to get a phase for each sample.
    """
    t = np.linspace(0, Config.DURATION, len(data))
    freq_env = base_freq * (
        1 + Config.FREQ_VARIANCE * np.sin(2*np.pi*0.5*t + 10*entropy)
    )
    # Numerically integrate freq_env -> phase
    phase = 2 * np.pi * np.cumsum(freq_env) / sr
    return data * np.cos(phase)

# ======== STEREO SPATIALIZATION ========
def spatialize(mono_signal, stereo_width):
    """
    Given arrays of width in [0..1], create an Nx2 stereo signal.
    Each sample n has a width(n). 
      left = mono*(0.5 + width/2)
      right= mono*(0.5 - width/2)
    """
    left = mono_signal * (0.5 + stereo_width/2.0)
    right = mono_signal * (0.5 - stereo_width/2.0)
    # Return shape (num_samples, 2)
    return np.vstack([left, right]).T

# ======== APPLY GAIN ENVELOPE ========
def apply_gain(stereo_signal, gain_env):
    """
    Multiply stereo signal by a gain envelope that changes from min->max across time.
    We'll do a straightforward linear ramp between the min and max of gain_env array.
    """
    num_samples = len(stereo_signal)
    # We'll interpret gain_env as one value per sample.
    # So just do element-wise multiply, but ensure shape alignment.
    # Also keep a small normalization to ensure we don't blow up amplitude.
    max_val = np.max(np.abs(stereo_signal))
    if max_val < 1e-12:
        max_val = 1.0  # avoid divide-by-zero
    out = np.zeros_like(stereo_signal)
    for i in range(num_samples):
        out[i] = stereo_signal[i] * gain_env[i]
    # Optionally re-normalize if needed:
    out /= np.max(np.abs(out)) if np.max(np.abs(out)) != 0 else 1.0
    return out

# ======== SAVE OUTPUT (STANDARD LIBRARY WAVE) ========
def save_output(signal, sr, filename):
    """
    Save float32 stereo signal as WAV using the built-in wave module.
    signal shape: (num_samples, 2)
    """
    # Clip to [-1,1] and convert to float32
    clipped = np.clip(signal, -1.0, 1.0).astype(np.float32)

    with wave.open(filename, 'wb') as wf:
        n_channels = 2
        sampwidth = 2  # we'll use 16-bit container for each float32,
                       # but effectively storing 32-bit raw
                       # (some wave readers might protest).
        # However, many wave readers will handle the bytes fine. 
        # If needed, store them as 16-bit PCM by scaling properly.
        
        # Instead, let's store as 32-bit PCM. That means:
        sampwidth = 4
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)  # 4 bytes per sample => 32-bit
        wf.setframerate(sr)
        wf.writeframes(clipped.tobytes())

# ======== MAIN GENERATION FUNCTION ========
def generate_neuro_noise():
    num_samples = int(Config.SAMPLE_RATE * Config.DURATION)
    
    # 1) Get chaotic envelope
    chaos = ChaoticEnveloper()
    stereo_width_array, gain_env_array = chaos.dynamic_params()
    
    # 2) Fractal noise + Brownian mixture
    frac = fractal_noise(num_samples)
    brown = SecureNoiseGenerator.brownian_noise(num_samples)
    
    # Weighted by ENTROPY_MODULATION
    modulated = frac * Config.ENTROPY_MODULATION + brown * (1 - Config.ENTROPY_MODULATION)
    
    # 3) Simple low-pass filter
    filtered = lowpass_filter(modulated, Config.FILTER_CUTOFF_HZ, Config.SAMPLE_RATE)
    
    # 4) Frequency shift
    freq_shifted = frequency_shift(filtered, Config.BASE_FREQ, chaos.entropy, Config.SAMPLE_RATE)
    
    # 5) Stereo
    # Each sample has its own stereo_width from stereo_width_array
    # We'll do this sample-by-sample:
    stereo_signal = np.zeros((num_samples, 2), dtype=np.float64)
    for i in range(num_samples):
        w = stereo_width_array[i]
        l = freq_shifted[i] * (0.5 + w/2.0)
        r = freq_shifted[i] * (0.5 - w/2.0)
        stereo_signal[i, 0] = l
        stereo_signal[i, 1] = r
    
    # 6) Apply gain envelope
    final_signal = apply_gain(stereo_signal, gain_env_array)
    
    # 7) Save as WAV
    save_output(final_signal, Config.SAMPLE_RATE, Config.OUTPUT_FILE)

if __name__ == "__main__":
    generate_neuro_noise()
    print("Neuro-noise generated and saved to:", Config.OUTPUT_FILE)

