import wfdb
import numpy as np 
import torch
from torch.utils.data import Dataset
class Preprocessor(Dataset):
    """
    Preprocessor Dataset class to load a physiological record, extract PPG and ECG signals,
    segment them into overlapping windows, and provide PyTorch tensors for model training.

    Attributes:
        record (wfdb.Record): The loaded record object.
        ppg (np.ndarray): 1D array of the PPG signal.
        ecg (np.ndarray): 1D array of the ECG signal.
        ppg_segments (list of np.ndarray): List of segmented PPG windows.
        ecg_segments (list of np.ndarray): List of segmented ECG windows.
    """
    def __init__(self,record_path):
        super().__init__()
        self.record = wfdb.rdrecord(record_path)
        # Extract signals
        self.ppg = self._get_ppg(self.record)
        self.ecg = self._get_ecg(self.record)

        # Segment signals
        self.ppg_segments = self._segment(self.ppg)
        self.ecg_segments = self._segment(self.ecg)
    def _get_ppg(self,record):
        """
        Extract the PPG signal as a 1D numpy array from the record.

        inputs:
            record (wfdb.Record): The record object.

        Returns:
            np.ndarray: PPG signal array.
        """
        ppg_idx=record.sig_name.index("PLETH,")
        return record.p_signal[:, ppg_idx]
    def _get_ecg(self,record):
        """
        Extract the ECG signal as a 1D numpy array from the record.

        inputs:
            record (wfdb.Record): The record object.

        Returns:
            np.ndarray: ECG signal array.
        """
        ii_idx = record.sig_name.index("II,")
        return record.p_signal[:, ii_idx]
    def get_times(self,record):
        #Generates time points (in seconds) for each signal sample based on the sampling frequency.
        return np.arange(0, (record.p_signal.shape[0] / record.fs), 1.0 / record.fs)
    def _segment(self,signal: np.ndarray):
        """
        Segment the signal into overlapping windows of length 1024 (4*256).

        inputs:
            signal (np.ndarray): 1D input signal.

        Returns:
            List[np.ndarray]: List of signal segments.
        """
        signal_segments = []
        for i in range(int(len(signal) / 256) - 4):
            if 256*i + 4 >= len(signal):
                break
            signal_segments.append(signal[256*i:256*(i+4)])
        return signal_segments
    def _transform(self, segment):
        """
        Transform a signal segment into a PyTorch tensor with shape (segment_length, 1).

        inputs:
            segment (np.ndarray): Signal segment array.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        t_signal = segment.astype(np.single)
        return torch.tensor(t_signal).unsqueeze(1)  # Shape: (1024, 1)

    def __getitem__(self, index):
        """
        Retrieve the transformed PPG and ECG segments at the specified index.

        inputs:
            index (int): Segment index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: PPG and ECG tensors.
        """
        ppg_seg = self._transform(self.ppg_segments[index]).T  # [1, 1024]
        ecg_seg = self._transform(self.ecg_segments[index]).T  # [1, 1024]
        return ppg_seg, ecg_seg

    def __len__(self):
        """
        Get the number of signal segments.

        Returns:
            int: Number of segments.
        """
        return len(self.ppg_segments)


            
            
        