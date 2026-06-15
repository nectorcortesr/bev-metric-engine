import numpy as np
import cv2

def calculate_compression_psnr(img_raw, img_h264):
    """
    Manually calculate the PSNR to quantify the edge degradation
    induced by H.264 compression macroblocks.
    """
    # Ensure transformation to high-precision floats
    r = np.asarray(img_raw, dtype=np.float64)
    c = np.asarray(img_h264, dtype=np.float64)
    
    if r.shape != c.shape:
        raise ValueError("Image dimensions must match exactly.")
    if r.size == 0:
        raise ValueError("Input image contains no valid pixels.")

    # If the images are identical (MSE == 0), the formula would result in division by zero.
    if np.array_equal(r, c):
        return float("inf")

    # Implement the calculation of the Mean Squared Error (MSE) by iterating over the flattened matrices.
    mse = np.sum((r - c) ** 2) / r.size

    # Apply the logarithmic transformation formula to obtain the PSNR in dB.
    psnr_scratch = 20 * np.log10(255.0 / np.sqrt(mse))

    return float(psnr_scratch)


def calculate_compression_psnr_library(img_raw, img_h264):
    return cv2.PSNR(img_raw, img_h264)


if __name__ == "__main__":
    np.random.seed(42)
    h, w = 64, 64
    
    raw_block = np.zeros((h, w), dtype=np.uint8)
    raw_block[:, w//2:] = 180 
    
    h264_block = raw_block.copy()
    noise_contour = np.random.randint(-15, 15, size=(h, 8), dtype=np.int16)
    h264_block[:, (w//2 - 4):(w//2 + 4)] = np.clip(
        h264_block[:, (w//2 - 4):(w//2 + 4)].astype(np.int16) + noise_contour, 0, 255
    ).astype(np.uint8)

    try:
        psnr_scratch = calculate_compression_psnr(raw_block, h264_block)
        psnr_lib = calculate_compression_psnr_library(raw_block, h264_block)
        
        assert np.allclose(psnr_scratch, psnr_lib, atol=1e-5)
        print(f"[PASS] PSNR metric against RAW: {psnr_scratch:.4f} dB")
        
    except NameError:
        print("[INFO] The script contains GAPs that block the execution flow. Complete the code.")