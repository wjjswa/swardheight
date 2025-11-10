import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.common_transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        transformed_data = self.common_transform(self.data[index]).float().to(device)
        targets = self.targets[index].float().to(device)  # assuming targets are already of correct type
        return transformed_data, targets
