import torch
import torch.nn as nn
import torch.nn.functional as F 

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2 block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True) )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    """Downscaling block: MaxPool => Specified Conv Block"""
    def __init__(self, in_channels, out_channels, block=DoubleConv):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), block(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling block using the 'factor' logic for bottleneck compatibility."""
    def __init__(self, in_ch_below, skip_ch, out_ch, bilinear=True, block=DoubleConv):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upsampled_ch = in_ch_below
        else:
            self.up = nn.ConvTranspose2d(in_ch_below, in_ch_below // 2, kernel_size=2, stride=2)
            upsampled_ch = in_ch_below // 2
        concat_ch = skip_ch + upsampled_ch
        self.conv = block(concat_ch, out_ch)
    def forward(self, x1, x2): # x1=from below, x2=skip connection
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1); return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 Convolution layer"""
    def __init__(self, in_channels, out_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_classes, kernel_size=1)
    def forward(self, x): return self.conv(x)

# --- Attention Gate Module ---
class AttentionGate(nn.Module):
    """
    Additive Attention Gate module as described in the paper.
    Filters features passed through the skip connection based on context from a gating signal.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): Number of channels in the gating signal (from lower layer).
            F_l (int): Number of channels in the input feature map (skip connection).
            F_int (int): Number of intermediate channels.
        """
        super(AttentionGate, self).__init__()

        # Gating signal transformation (1x1 Conv)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Input feature map transformation (1x1 Conv)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Final 1x1 Conv to get attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Sigmoid to get coefficients between 0 and 1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from the lower decoder layer.
            x: Skip connection signal from the corresponding encoder layer.
        Returns:
            Attention-weighted skip connection signal.
        """
        # Transform gating signal
        g1 = self.W_g(g) # (Batch, F_int, H/2, W/2) -> assuming g comes from coarser scale

        # Transform skip connection signal
        x1 = self.W_x(x) # (Batch, F_int, H, W)

        # Upsample g1 to match spatial dimensions of x1 before adding
        # align_corners=True is often recommended with 'bilinear'
        g1_upsampled = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)

        # Add transformed signals and apply ReLU
        psi_input = self.relu(g1_upsampled + x1) # (Batch, F_int, H, W)

        # Compute attention coefficients (alpha)
        alpha = self.psi(psi_input) # (Batch, 1, H, W)

        # Apply attention coefficients to original skip connection features
        return x * alpha # Element-wise multiplication


# --- Attention U-Net ---
class AttentionUNet(nn.Module):
    """
    U-Net architecture with integrated Attention Gates in skip connections.
    Based on the UNetFactor structure (using factor logic for bottleneck).
    """
    def __init__(self, in_channels, out_classes, bilinear=True, block=DoubleConv): # Allow specifying block
        super(AttentionUNet, self).__init__()
        if not isinstance(in_channels, int) or in_channels < 1: raise ValueError("in_channels must be > 0")
        if not isinstance(out_classes, int) or out_classes < 1: raise ValueError("out_classes must be > 0")

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.bilinear = bilinear

        # Determine bottleneck channels using factor logic
        factor = 2 if bilinear else 1
        bottleneck_channels = 1024 // factor # 512 if bilinear=T, 1024 if bilinear=F

        # Define channel sizes
        ch_inc, ch_down1, ch_down2, ch_down3 = 64, 128, 256, 512
        ch_down4 = bottleneck_channels
        ch_up1, ch_up2, ch_up3, ch_up4 = 512, 256, 128, 64

        # Intermediate channels for Attention Gates (can be tuned, e.g., F_l / 2)
        F_int_d3 = ch_down3 // 2
        F_int_d2 = ch_down2 // 2
        F_int_d1 = ch_down1 // 2
        F_int_inc = ch_inc // 2

        # --- Encoder ---
        self.inc = block(in_channels, ch_inc)
        self.down1 = Down(ch_inc, ch_down1, block=block)
        self.down2 = Down(ch_down1, ch_down2, block=block)
        self.down3 = Down(ch_down2, ch_down3, block=block)
        self.down4 = Down(ch_down3, ch_down4, block=block)

        # --- Attention Gates ---
        # AG input: F_g (channels from below), F_l (channels from skip), F_int
        self.Att1 = AttentionGate(F_g=ch_down4, F_l=ch_down3, F_int=F_int_d3)
        self.Att2 = AttentionGate(F_g=ch_up1,   F_l=ch_down2, F_int=F_int_d2)
        self.Att3 = AttentionGate(F_g=ch_up2,   F_l=ch_down1, F_int=F_int_d1)
        self.Att4 = AttentionGate(F_g=ch_up3,   F_l=ch_inc,   F_int=F_int_inc)

        # --- Decoder ---
        self.up1 = Up(ch_down4, ch_down3, ch_up1, bilinear, block=block) # Below=bottleneck, Skip=down3(512), Out=512
        self.up2 = Up(ch_up1,   ch_down2, ch_up2, bilinear, block=block) # Below=up1(512), Skip=down2(256), Out=256
        self.up3 = Up(ch_up2,   ch_down1, ch_up3, bilinear, block=block) # Below=up2(256), Skip=down1(128), Out=128
        self.up4 = Up(ch_up3,   ch_inc,   ch_up4, bilinear, block=block) # Below=up3(128), Skip=inc(64), Out=64

        # Final Output Layer
        self.outc = OutConv(ch_up4, out_classes)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)    # Skip connection 4 (channels: ch_inc)
        x2 = self.down1(x1) # Skip connection 3 (channels: ch_down1)
        x3 = self.down2(x2) # Skip connection 2 (channels: ch_down2)
        x4 = self.down3(x3) # Skip connection 1 (channels: ch_down3)
        x5 = self.down4(x4) # Bottleneck (channels: ch_down4)

        # --- Decoder with Attention Gates ---
        # Level 1 (deepest)
        x4_att = self.Att1(g=x5, x=x4)     # Apply attention to x4 using x5 as gating signal
        d1 = self.up1(x5, x4_att)          # Pass bottleneck (x5) and attended skip (x4_att) to Up block

        # Level 2
        x3_att = self.Att2(g=d1, x=x3)     # Apply attention to x3 using d1 as gating signal
        d2 = self.up2(d1, x3_att)          # Pass d1 and attended skip (x3_att) to Up block

        # Level 3
        x2_att = self.Att3(g=d2, x=x2)     # Apply attention to x2 using d2 as gating signal
        d3 = self.up3(d2, x2_att)          # Pass d2 and attended skip (x2_att) to Up block

        # Level 4 (shallowest)
        x1_att = self.Att4(g=d3, x=x1)     # Apply attention to x1 using d3 as gating signal
        d4 = self.up4(d3, x1_att)          # Pass d3 and attended skip (x1_att) to Up block

        # --- Output ---
        logits = self.outc(d4)
        return logits