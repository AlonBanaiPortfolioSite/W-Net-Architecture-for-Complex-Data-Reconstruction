import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,random_split 
import torch.nn as nn
import torchviz
import torch.nn.functional as F
class WNet1D(nn.Module):
    """
    A 1D W-Net architecture for signal reconstruction tasks (e.g., ECG from PPG).

    This model consists of two stacked 1D U-Net-like networks. Each U-Net includes:
    - An encoder: a sequence of 1D convolutional blocks with downsampling via MaxPooling.
    - A bottleneck (center) block.
    - A decoder: a sequence of upsampling layers and skip connections from the encoder.
    
    Skip connections are designed with custom channel combinations defined by `dec_skip_configs`.

    The first U-Net produces an intermediate reconstruction. The second U-Net refines this output
    using residual learning (by adding the input signal to the first output).

    inputs:
        in_channels (int): Number of input channels (e.g., 1 for a single PPG signal).
        out_channels (int): Number of output channels (e.g., 1 for a single ECG signal).

    Components:
        - conv_block: Conv1D → BatchNorm1D → LeakyReLU.
        - _make_enc_blocks: Builds encoder from a list of channel sizes.
        - _make_dec_blocks: Builds decoder from a list of (skip_input_channels, output_channels).
        - _unet_pass: Runs a single U-Net forward pass with skip connections.
        - forward: Applies two sequential U-Nets with residual correction in between.

    Notes:
        - Uses kernel size 15 for all convolutions.
        - Uses nearest-neighbor upsampling and max pooling with stride 2.
    """

    def __init__(self,in_channels=1,out_channels=1):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.pool=nn.MaxPool1d(kernel_size=2,stride=2)
        self.upsample=nn.Upsample(scale_factor=2,mode='nearest')
         # Define encoder channels
        self.enc_channels = [in_channels, 24, 48, 72, 96, 120, 144, 168, 192]
        # Define decoder skip connection structure
        self.dec_skip_configs = [(192 + 192, 168),(168 + 168, 192),(144 + 192, 144),(144 + 120, 120),(120 + 96, 96),(96 + 72, 72),(72 + 48, 72),(72 + 24, 24),]

        # Build encoder/decoder for both UNets
        self.enc1_blocks = self._make_enc_blocks(self.enc_channels)
        self.center1 = self._conv_block(192, 192)
        self.dec1_blocks = self._make_dec_blocks(self.dec_skip_configs)
        self.final1 = nn.Conv1d(24, out_channels, 1)

        self.enc2_blocks = self._make_enc_blocks([out_channels] + self.enc_channels[1:])
        self.center2 = self._conv_block(192, 192)
        self.dec2_blocks = self._make_dec_blocks(self.dec_skip_configs)
        self.final2 = nn.Conv1d(24, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=15,stride=1, padding=7),
            nn.BatchNorm1d(out_ch),
            self.leaky_relu,
        )

    def _make_enc_blocks(self, channels):
        return nn.ModuleList([
            self._conv_block(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

    def _make_dec_blocks(self, skip_configs):
        return nn.ModuleList([
            self._conv_block(in_ch, out_ch)
            for in_ch, out_ch in skip_configs
        ])

    def _unet_pass(self, x, enc_blocks, center_block, dec_blocks):
        skips = []
        for block in enc_blocks:
            x = block(x)
            skips.append(x)
            #print(f"Encoder output shape: {x.shape}")
            x = self.pool(x)
        #print(f"Center input shape: {x.shape}")
        x = center_block(x)
        #print(f"Center output shape: {x.shape}")

        # reverse the encoder outputs for skip connections
        skips = skips[::-1]

        for i, block in enumerate(dec_blocks):
            x = self.upsample(x)
            #print(f"Upsampled decoder input shape at block {i}: {x.shape}")
            #print(f"Corresponding skip connection shape at block {i}: {skips[i].shape}")
            # Check if shapes match along length dimension before concat
            if x.shape[2] != skips[i].shape[2]:
                print(f"WARNING: Shape mismatch at decoder block {i} - upsampled length {x.shape[2]} != skip length {skips[i].shape[2]}")
            x = torch.cat([x, skips[i]], dim=1)
            #print(f"After concatenation at decoder block {i}: {x.shape}")
            x = block(x)
            #print(f"After conv block at decoder block {i}: {x.shape}")

        return x

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] != 1:
            x = x.permute(0, 2, 1)

        u1 = self._unet_pass(x, self.enc1_blocks, self.center1, self.dec1_blocks)
        out1 = self.final1(u1)

        residual = out1 + x
        u2 = self._unet_pass(residual, self.enc2_blocks, self.center2, self.dec2_blocks)
        out2 = self.final2(u2)

        return out2




