
from torch.utils.data import Dataset


class DrumAudioDataset(Dataset):
    """Drum audio dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Path to all the audio files.
        """
        self.root_dir = root_dir

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

