import torchaudio
import torch
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

MAX_EPOCH = 100  # Maximum number of epochs to train for

def create_model():
    return DiffusionModel(
        net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
        in_channels=2,  # U-Net: number of input/output (audio) channels
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
        attention_heads=8,  # U-Net: number of attention heads per attention item
        attention_features=64,  # U-Net: number of attention features per attention item
        diffusion_t=VDiffusion,  # The diffusion method used
        sampler_t=VSampler,  # The diffusion sampler used
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    epoch = 0
    step = 0

    model.train()
    while epoch < MAX_EPOCH:
