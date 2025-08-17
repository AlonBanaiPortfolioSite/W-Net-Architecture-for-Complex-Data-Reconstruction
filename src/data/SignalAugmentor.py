import torch
import numpy as np

class SignalAugmentor:
    """
    A class for performing data augmentation on 1D physiological signals
    such as ECG and PPG. Supports jittering, permutation, window cropping
    with warping, MixUp, and CutMix.

    Parameters:
    - jitter_std (float): Standard deviation of Gaussian noise for jittering.
    - permute_segments (int): Number of segments to divide signal for permutation.
    - crop_ratio (float): Ratio of the window to crop before warping back.
    - mixup_alpha (float): Alpha parameter for Beta distribution in MixUp.
    - cutmix_alpha (float): Alpha parameter for Beta distribution in CutMix.
    - seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        jitter_std=0.02,
        permute_segments=6,
        crop_ratio=0.9,
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
        seed=42,
    ):
        self.jitter_std = jitter_std
        self.permute_segments = permute_segments
        self.crop_ratio = crop_ratio
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.rng = np.random.default_rng(seed)

    def jitter(self, signal,y=None):
        """
        Add small random Gaussian noise to the signal to simulate sensor noise.
        """
        noise = torch.tensor(self.rng.normal(0, self.jitter_std, size=signal.shape), dtype=signal.dtype)
        return signal + noise

    def permutation(self, signal,y=None):
        """
        Divide the signal into segments and randomly shuffle them to alter local structure
        while preserving overall content.
        """
        seg_len = signal.shape[0] // self.permute_segments
        remainder = signal.shape[0] % self.permute_segments

        segments = [
        signal[i * seg_len: (i + 1) * seg_len]
        for i in range(self.permute_segments)
        ]

        if remainder > 0:
            # Add remainder to last segment
            segments[-1] = torch.cat([segments[-1], signal[-remainder:]])

        self.rng.shuffle(segments)
        return torch.cat(segments)

    def window_crop_warp(self, signal,y=None):
        """
        Randomly crop a window from the signal and warp it back to original length
        using interpolation, simulating temporal distortions.
        Args:
            signal (torch.Tensor): Tensor of shape [1, T] or [C, T] where T is the time dimension.
            y (optional): Not used.

        Returns:
            torch.Tensor: Warped signal of the same shape as input.
        """
        #print("input:", signal.shape)  # should be [1, 1024] or similar

        orig_len = signal.shape[1]
        crop_len = max(2, int(orig_len * self.crop_ratio))  # must be >= 2 for interpolation
        start = self.rng.integers(0, orig_len - crop_len)

        cropped = signal[:, start:start + crop_len]  # crop along time
        #print("cropped:", cropped.shape)

        cropped_np = cropped.squeeze().detach().cpu().numpy()  # (crop_len,)
        #print("cropped_np:", cropped_np.shape)

        # Interpolate back to original length
        warped_np = np.interp(
            np.linspace(0, crop_len - 1, orig_len),
            np.arange(crop_len),
            cropped_np
        )
        warped = torch.from_numpy(warped_np).type(signal.dtype).unsqueeze(0)  # restore [1, T]
        #print("warped:", warped.shape)

        return warped

    def mixup(self, x1, y1):
        """
        Blend two signals and their labels using a weighted average. Helps regularize the model.
        """
        lam = self.mixup_alpha
        indices = torch.randperm(len(x1))
        x2 = x1[indices]
        y2 = y1[indices]
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y

    def cutmix(self, x1, y1):
        """
        Apply CutMix augmentation by replacing a random segment of each signal in the batch 
        with a segment from another signal in the same batch. Output has same shape as input.
        
        Args:
            x1 (torch.Tensor): Input signals, shape [B, L]
            y1 (torch.Tensor): Target signals, shape [B, L] or [B]
            
        Returns:
            x (torch.Tensor): CutMixed signals, same shape as x1
            y (torch.Tensor): Mixed labels, same shape as y1
        """
        B, L = x1.shape
        indices = torch.randperm(B)
        x2 = x1[indices]
        y2 = y1[indices]
    
        x = x1.clone()
        lam = self.cutmix_alpha
    
        cut_len = int(L * lam)
        cut_starts = self.rng.integers(0, L - cut_len + 1, size=B)
    
        for i in range(B):
            start = cut_starts[i]
            end = start + cut_len
            x[i, start:end] = x2[i, start:end]
    
        y = lam * y1 + (1 - lam) * y2  # assumes y is shape [B, ...]
        return x, y

    def apply(self, x, y, method="jitter"):
        """
        Apply augmentation method(s) to a single input (x, y) pair.

        Supported methods:
        - "jitter"
        - "permutation"
        - "window_crop"
        - "mixup" (uses jitter(x) as second sample)
        - "cutmix" (uses jitter(x) as second sample)
        - "all" (applies all deterministic transforms in sequence)
        """
        if method == "jitter":
            return self.jitter(x), y
        elif method == "permutation":
            return self.permutation(x), y
        elif method == "window_crop":
            return self.window_crop_warp(x), y
        elif method == "mixup":
            x2, y2 = self.jitter(x), self.jitter(y)
            return self.mixup(x, y, x2, y2)
        elif method == "cutmix":
            x2, y2 = self.jitter(x), self.jitter(y)
            return self.cutmix(x, y, x2, y2)
        elif method == "all":
            x_aug = self.jitter(x)
            x_aug = self.permutation(x_aug)
            x_aug = self.window_crop_warp(x_aug)
            return x_aug, y  # mixup and cutmix require 2 samples; skip them here
        else:
            raise ValueError(f"Unknown method: {method}")