import numpy as np

def calculate_frame_drop_rate(sequence_ids):
    """
    Calculate the frame drop rate by analyzing
    discontinuities in the received sequence identifiers.
    """
    # Ensure integer data type for sequence identifiers
    seq = np.asarray(sequence_ids, dtype=np.int64)
    
    # If the sequence list is empty, we must throw an explicit exception.
    if seq.size == 0:
        raise ValueError("Empty sequence.")

    # Find the minimum and maximum values to derive N_expected.
    seq_min = seq[0]
    seq_max = seq[0]
    for num in seq:
        if num < seq_min:
            seq_min = num
        if num > seq_max:
            seq_max = num

    n_expected = seq_max - seq_min + 1

    # Determine the number of unique IDs (N_received) and calculate the FDR fraction.
    unique_ids = []
    for num in seq:
        if num not in unique_ids:
            unique_ids.append(num)
    n_received = len(unique_ids)
    fdr_scratch = (n_expected - n_received) / n_expected

    return float(fdr_scratch)


def calculate_frame_drop_rate_library(sequence_ids):
    seq = np.asarray(sequence_ids, dtype=np.int64)
    if seq.size == 0:
        raise ValueError("Empty sequence.")
        
    n_expected = np.max(seq) - np.min(seq) + 1
    n_received = len(np.unique(seq))
    
    fdr_lib = (n_expected - n_received) / n_expected
    return float(fdr_lib)


if __name__ == "__main__":
    # Simulation of an industrial UDP network burst with packet loss and reordering
    corrupted_sequence = [500, 501, 502, 503, 506, 507, 508, 508, 509, 511, 512, 513, 514]

    try:
        fdr_scratch = calculate_frame_drop_rate(corrupted_sequence)
        fdr_lib = calculate_frame_drop_rate_library(corrupted_sequence)
        
        assert np.allclose(fdr_scratch, fdr_lib, atol=1e-7)
        print(f"[PASS] Frame Drop Rate metric: {fdr_scratch * 100:.2f}%")
        
    except NameError:
        print("[INFO] The script contains active GAPs. Complete Step 1 to execute.")