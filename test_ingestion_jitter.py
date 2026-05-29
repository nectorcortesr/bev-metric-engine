import numpy as np

def analyze_ingestion_jitter(timestamps):
    """
    Calculate arrival jitter by analyzing the standard deviation
    of the inter-frame time deltas.
    """
    # Force safe conversion to float64 to avoid overflows or truncations
    timestamps = np.asarray(timestamps, dtype=np.float64)
    
    if len(timestamps) <= 2:
        raise ValueError("Insufficient samples to calculate temporal variance.")
    
    # Implement the inter-frame time deltas explicitly.
    deltas = np.empty(len(timestamps) - 1, dtype=np.float64)
    for i in range(1, len(timestamps)):
        deltas[i - 1] = timestamps[i] - timestamps[i - 1]
    
    # Calculate the mean of the deltas and then the population standard deviation.
    mean_delta = np.mean(deltas)
    variance = np.mean((deltas - mean_delta) ** 2)
    jitter_scratch = np.sqrt(variance)
    
    # If all frames arrive with exactly the same timestamp due to a capture software duplication error,
    # the theoretical jitter is 0.0. Ensure stability.
    if variance == 0.0:
        jitter_scratch = 0.0

    return jitter_scratch


def analyze_ingestion_jitter_library(timestamps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) <= 2:
        raise ValueError("Insufficient samples.")
    
    # Optimal implementation using NumPy vectorized primitives
    deltas = np.diff(timestamps)
    jitter_lib = np.std(deltas)
    return jitter_lib

if __name__ == "__main__":
    # Simulation of a nominal stream with slight Gaussian network noise (FPS=30, low jitter)
    np.random.seed(42)
    base_timestamps = np.linspace(0, 1000, 31) # 30 frames in 1 second
    network_noise = np.random.normal(0, 0.5, len(base_timestamps)) # Jitter introduced ~0.5ms
    nominal_stream = base_timestamps + network_noise

    try:
        j_scratch = analyze_ingestion_jitter(nominal_stream)
        j_lib = analyze_ingestion_jitter_library(nominal_stream)
        
        # Strict floating-point tolerance of 1e-7
        assert np.allclose(j_scratch, j_lib, atol=1e-7)
        print(f"[PASS] jitter metric: {j_scratch:.4f} ms")
    except NameError:
        print("[INFO] The script contains GAPs that prevent its execution. Complete the code.")