import numpy as np

def profile_sensor_snr(roi_pixels):
    """
    Characterizes the sensor variation and calculates the SNR in dB.
    Over a homogeneous region to evaluate low-light degradation.
    """
    # Force clean conversion to float64 to preserve precision of elementary operations
    img_patch = np.asarray(roi_pixels, dtype=np.float64)
    
    # Implement the required protections. If the patch is empty, throw a ValueError.
    if img_patch.size == 0:
        raise ValueError("Empty region.")
    
    # Develop the manual calculation of the arithmetic mean and population standard deviation
    # of the pixels by traversing the array in a flat or linearized manner.
    total = 0.0
    for pixel in img_patch.flat:
        total += pixel
    mean_val = total / img_patch.size

    variance = 0.0
    for pixel in img_patch.flat:
        variance += (pixel - mean_val) ** 2
    std_val = (variance / img_patch.size) ** 0.5

    if mean_val == 0.0 or std_val == 0.0:
        return 0.0

    # Calculate the SNR metric in decibels by implementing the base-10 logarithmic transformation.
    snr_scratch = 20 * np.log10(mean_val / std_val)

    return float(snr_scratch)


def profile_sensor_snr_library(roi_pixels):
    img_patch = np.asarray(roi_pixels, dtype=np.float64)
    if img_patch.size == 0:
        raise ValueError("Empty region.")
        
    mean_val = np.mean(img_patch)
    std_val = np.std(img_patch)
    
    if mean_val == 0.0 or std_val == 0.0:
        return 0.0
        
    snr_lib = 20 * np.log10(mean_val / std_val)
    return float(snr_lib)


if __name__ == "__main__":
    # Simulation of an asphalt inspection patch under low-light conditions (high ISO)
    # Nominal mean signal = 80.0 DN, with a Gaussian read noise fluctuation of sigma = 4.0
    np.random.seed(42)
    pure_signal = np.full((32, 32), 80.0)
    sensor_noise = np.random.normal(0.0, 4.0, pure_signal.shape)
    low_light_roi = pure_signal + sensor_noise

    try:
        snr_scratch = profile_sensor_snr(low_light_roi)
        snr_lib = profile_sensor_snr_library(low_light_roi)

        assert np.allclose(snr_scratch, snr_lib, atol=1e-6)
        print(f"[PASS] Calculated SNR metric: {snr_scratch:.2f} dB")
        
    except NameError:
        print("[INFO] The script contains GAPs that block the execution of the validation flow.")