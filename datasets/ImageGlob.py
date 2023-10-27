from PIL import Image
from torch.utils.data import Dataset
import glob


class ImageGlob(Dataset):
    
    def __init__(self, img_glob, transform=None):
        
        self.samples = [f for f in sorted(glob.glob(img_glob))]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
