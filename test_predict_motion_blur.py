import numpy as np

def predict_motion_blur_radius(velocity_kmh, exposure_time, focal_length, distance_z, pixel_size):
    """
    Analytically predicts the radius of the motion blur kernel induced in pixels
    to validate the potential localization error in moving vehicles.
    """
    v_kmh = float(velocity_kmh)
    t_e = float(exposure_time)
    f = float(focal_length)
    z = float(distance_z)
    p_s = float(pixel_size)
    
    # Prevent division by zero or physically impossible values in the projective space.
    if z <= 0.0 or p_s <= 0.0:
        raise ValueError("Invalid geometric parameters.")

    # Convert velocity from km/h to m/s and calculate the metric displacement delta_x.
    v_ms = v_kmh / 3.6
    delta_x = v_ms * t_e

    # Compute the blur kernel radius based on the derived pinhole projective equation.
    blur_radius_scratch = (f * delta_x) / (2.0 * z * p_s)

    return float(blur_radius_scratch)


def predict_motion_blur_radius_library(velocity_kmh, exposure_time, focal_length, distance_z, pixel_size):
    if distance_z <= 0.0 or pixel_size <= 0.0:
        raise ValueError("Invalid geometric parameters.")
    
    # Vectorized resolution using structured native operations
    v_ms = np.divide(velocity_kmh, 3.6)
    delta_x = np.multiply(v_ms, exposure_time)
    
    # Total length in pixels on the focal plane
    l_blur = (focal_length * delta_x) / (distance_z * pixel_size)
    return float(l_blur / 2.0)


if __name__ == "__main__":
    # Real parameters extracted from the industrial camera installed in the logistics yard
    cam_velocity = 20.0       # Truck moving at 20 km/h (according to phase specifications)
    shutter_speed = 0.016     # Shutter open for 16ms (1/60s standard in low light)
    lens_focal = 0.008        # 8mm lens
    target_range = 12.0       # Truck crossing at 12 meters distance
    sensor_pitch = 4.5e-6     # Pixel size of 4.5 micrometers

    try:
        r_scratch = predict_motion_blur_radius(cam_velocity, shutter_speed, lens_focal, target_range, sensor_pitch)
        r_library = predict_motion_blur_radius_library(cam_velocity, shutter_speed, lens_focal, target_range, sensor_pitch)
        
        assert np.allclose(r_scratch, r_library, atol=1e-7)
        print(f"[PASS] Metric Blur Kernel Radius: {r_scratch:.4f} pixels")
        
    except NameError:
        print("[INFO] The script contains GAPs that block the execution flow. Complete the code.")