import numpy as np
from scipy.stats import skew as calc_skew
from scipy.stats import kurtosis as calc_kurtosis
from scipy.signal import welch as calc_psd
from scipy.integrate import simpson as calc_integral
from warnings import catch_warnings, filterwarnings
import matplotlib.pyplot as plt

def segment_array(array, stride=256, n_strides=4):
    # Segment the signal into overlapping windows
    # 4 strides of 256 for a 1024 channel signal
    signal_segments = []
    for i in range(int(len(array) / stride) - n_strides):
        if stride*i + n_strides >= len(array):
            break
        signal_segments.append(array[stride*i:stride*(i+n_strides)])
    return signal_segments

def calc_rel_power_ppg(signal, time_seg):
    """
    Calculate relative power for PPG (Photoplethysmogram) signals.
    
    This function computes the relative power in the physiologically relevant frequency band
    for PPG signals (1-10 Hz) compared to the total power below 30 Hz.
    
    Parameters:
    -----------
    signal : array-like
        Input PPG signal amplitude values
    time_seg : array-like
        Time points corresponding to the signal samples
        
    Returns:
    --------
    float
        Relative power ratio (signal_power / total_power)
        Range: [0, 1] where higher values indicate stronger signal in the relevant band
        
    Notes:
    ------
    - Signal band: 1-10 Hz (typical heart rate range: 60-600 BPM)
    - Total power computed up to 30 Hz (Nyquist consideration)
    - Uses Welch's method for power spectral density estimation
    """
    fs = 1 / (time_seg[1] - time_seg[0])
    freqs, psd = calc_psd(signal, fs=fs)
    signal_freqs = (freqs > 1) * (freqs < 10)
    total_freqs = freqs < 30
    signal_power = calc_integral(psd[signal_freqs], freqs[signal_freqs])
    total_power = calc_integral(psd[total_freqs], freqs[total_freqs])
    return signal_power / total_power



def calc_rel_power_ii(signal, time_seg):
    """
    Calculate relative power for II (Impedance) signals.
    
    This function computes the relative power in the physiologically relevant frequency band
    for impedance-based signals (5-15 Hz) compared to the total power below 30 Hz.
    
    Parameters:
    -----------
    signal : array-like
        Input impedance signal amplitude values
    time_seg : array-like
        Time points corresponding to the signal samples
        
    Returns:
    --------
    float
        Relative power ratio (signal_power / total_power)
        Range: [0, 1] where higher values indicate stronger signal in the relevant band
        
    Notes:
    ------
    - Signal band: 5-15 Hz (respiratory and cardiac impedance changes)
    - Total power computed up to 30 Hz
    - Uses Welch's method for power spectral density estimation
    """
    fs = 1 / (time_seg[1] - time_seg[0])
    freqs, psd = calc_psd(signal, fs=fs)
    signal_freqs = (freqs > 5) * (freqs < 15)
    total_freqs = freqs < 30
    signal_power = calc_integral(psd[signal_freqs], freqs[signal_freqs])
    total_power = calc_integral(psd[total_freqs], freqs[total_freqs])
    return signal_power / total_power


def test_segments(signal_segments, times_segments, signal_type="ppg"):
    """
    Compute signal quality metrics for multiple signal segments.
    
    This function analyzes signal quality by computing skewness, relative power,
    and kurtosis for each segment. It returns both clean (valid) and dirty (all) results.
    
    Parameters:
    -----------
    signal_segments : list of array-like
        List of signal amplitude arrays, one per segment
    times_segments : list of array-like
        List of time arrays corresponding to each signal segment
    signal_type : str, optional
        Type of signal processing ("ppg" or "ii"), default="ppg"
        
    Returns:
    --------
    tuple
        Two tuples containing:
        - clean_results: (skew_array, rel_power_array, kurtosis_array) - only valid values
        - dirty_results: (skew_array, rel_power_array, kurtosis_array) - all values (including NaN)
        
    Raises:
    -------
    ValueError
        If signal_type is not "ppg" or "ii"
        
    Notes:
    ------
    - Skewness: Measures asymmetry of signal distribution
    - Relative Power: Signal strength in relevant frequency band
    - Kurtosis: Measures tail heaviness of signal distribution
    - Clean arrays contain only successfully computed values
    - Dirty arrays contain NaN for failed computations
    """
    clean_skew_list = []
    clean_rel_power_list = []
    clean_kurtosis_list = []
    dirty_skew_list = []
    dirty_rel_power_list = []
    dirty_kurtosis_list = []

    if signal_type == "ppg":
        calc_rel_power = calc_rel_power_ppg
    elif signal_type == "ii":
        calc_rel_power = calc_rel_power_ii
    else:
        raise ValueError("Invalid value for signal_type")
    
    for i, (signal_seg, time_seg) in enumerate(zip(signal_segments, times_segments)):
        #print(signal_seg)
        #print(type(signal_seg))
        #print(time_seg)
        #print(type(time_seg))
        if np.any(np.isnan(signal_seg)):
            skew = np.nan
            rel_power = np.nan
            kurtosis = np.nan
            dirty_skew_list.append(skew)
            dirty_rel_power_list.append(rel_power)
            dirty_kurtosis_list.append(kurtosis)
            continue
            
        try:
            with catch_warnings():
                filterwarnings('error', category=RuntimeWarning)
                skew = calc_skew(signal_seg)
                rel_power = calc_rel_power(signal_seg, time_seg)
                kurtosis = calc_kurtosis(signal_seg)
            if np.any(np.isnan([skew, rel_power, kurtosis])):
                raise RuntimeError("Found nan")
            clean_skew_list.append(skew)
            clean_rel_power_list.append(rel_power)
            clean_kurtosis_list.append(kurtosis)
        except RuntimeWarning as e:
            print(f"Caught a runtime warning: {e}")
            skew = np.nan
            rel_power = np.nan
            kurtosis = np.nan
        except Exception as e:
            print(f"Caught a runtime Exception: {e}")
            skew = np.nan
            rel_power = np.nan
            kurtosis = np.nan
        
        dirty_skew_list.append(skew)
        dirty_rel_power_list.append(rel_power)
        dirty_kurtosis_list.append(kurtosis)
    
    return (
        (np.array(clean_skew_list), np.array(clean_rel_power_list), np.array(clean_kurtosis_list)),
        (np.array(dirty_skew_list), np.array(dirty_rel_power_list), np.array(dirty_kurtosis_list)),
    )

def filter_segments(
    signal_segments, times_segments, ranges_dict=None, test_results=None, signal_type="ppg"
):
    """
    Filter signal segments based on quality metrics thresholds.
    
    This function identifies valid signal segments by applying thresholds to
    skewness, relative power, and kurtosis metrics. Thresholds can be specified
    as absolute values or percentiles.
    
    Parameters:
    -----------
    signal_segments : list of array-like
        List of signal amplitude arrays, one per segment
    times_segments : list of array-like
        List of time arrays corresponding to each signal segment
    ranges_dict : dict, optional
        Dictionary containing filtering criteria:
        - "skew": (min, max) tuple for skewness range
        - "rel_power": (min, max) tuple for relative power range
        - "kurtosis": (min, max) tuple for kurtosis range
        - "percentiles": bool, if True interpret ranges as percentiles
        Default: {"skew": (0.0, 1.0), "rel_power": (0.0, 1), "kurtosis": (0.0, 1.0), "percentiles": True}
    test_results : tuple, optional
        Pre-computed results from test_segments() to avoid recomputation
    signal_type : str, optional
        Type of signal processing ("ppg" or "ii"), default="ppg"
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating valid segments (True) and invalid segments (False)
        
    Notes:
    ------
    - When percentiles=True, ranges are interpreted as quantiles (0.0-1.0)
    - When percentiles=False, ranges are absolute threshold values
    - A segment is valid only if ALL three metrics fall within their respective ranges
    - Prints summary of filtering results
    
    Examples:
    ---------
    # Use percentile-based filtering (default)
    valid_idx = filter_segments(signals, times, {"skew": (0.1, 0.9), "percentiles": True})
    
    # Use absolute threshold filtering
    valid_idx = filter_segments(signals, times, {"skew": (-1, 1), "percentiles": False})
    """
    if test_results is None:
        test_results = test_segments(signal_segments, times_segments, signal_type)

    dirty_skew_arr, dirty_rel_power_arr, dirty_kurtosis_arr = test_results[1]

    if ranges_dict is None:
        ranges_dict = {
            "skew": (0.0, 1.0),
            "rel_power": (0.0, 1),
            "kurtosis": (0.0, 1.0),
            "percentiles": True,
        }
    print(f"Skew stats: min={np.nanmin(dirty_skew_arr)}, max={np.nanmax(dirty_skew_arr)}")
    print(f"Rel Power stats: min={np.nanmin(dirty_rel_power_arr)}, max={np.nanmax(dirty_rel_power_arr)}")
    print(f"Kurtosis stats: min={np.nanmin(dirty_kurtosis_arr)}, max={np.nanmax(dirty_kurtosis_arr)}")
    if ranges_dict.get("percentiles", False):
        skew_min, skew_max = np.nanquantile(dirty_skew_arr, ranges_dict["skew"], method="nearest")
        rel_power_min, rel_power_max = np.nanquantile(
            dirty_rel_power_arr, ranges_dict["rel_power"], method="nearest"
        )
        kurtosis_min, kurtosis_max = np.nanquantile(
            dirty_kurtosis_arr, ranges_dict["kurtosis"], method="nearest"
        )
    else:
        skew_min, skew_max = ranges_dict["skew"]
        rel_power_min, rel_power_max = ranges_dict["rel_power"]
        kurtosis_min, kurtosis_max = ranges_dict["kurtosis"]
    print(f"Skew range: ({skew_min}, {skew_max})")
    print(f"Rel power range: ({rel_power_min}, {rel_power_max})")
    print(f"Kurtosis range: ({kurtosis_min}, {kurtosis_max})")

    valid_skew_idxs = (dirty_skew_arr >= skew_min) * (dirty_skew_arr <= skew_max)
    print(f" valid_skew_idxs={valid_skew_idxs}")
    valid_rel_power_idxs = (
        (dirty_rel_power_arr >= rel_power_min) * (dirty_rel_power_arr <= rel_power_max)
    )
    valid_kurtosis_idxs = (dirty_kurtosis_arr >= kurtosis_min) * (dirty_kurtosis_arr <= kurtosis_max)
    print(f" valid_kurtosis_idxss={ valid_kurtosis_idxs}")
    valid_idxs = valid_skew_idxs * valid_rel_power_idxs * valid_kurtosis_idxs
    print(
        f"{np.count_nonzero(valid_idxs)} out of {len(signal_segments)} {signal_type.upper()} are valid"
    )

    return valid_idxs

