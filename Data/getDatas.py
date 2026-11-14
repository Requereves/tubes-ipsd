import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

DATASET = "/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20"
SIZE = 128                 

prep = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def find_dir(fold, split, aug=False):
    
    for dp, dn, _ in os.walk(DATASET):
        low = dp.lower().replace("_", "").replace(" ", "")
        if os.path.basename(dp).lower() == split.lower() and f"fold{fold}" in low \
           and ("aug" in low) == aug and dn:
            return dp
    raise FileNotFoundError(f"Folder {split} fold {fold} tidak ketemu")

def get_train(fold, batch_size=32):
    
    train = ConcatDataset([datasets.ImageFolder(find_dir(fold, "Train", False), prep),
                           datasets.ImageFolder(find_dir(fold, "Train", True), prep)])
    valid = datasets.ImageFolder(find_dir(fold, "Valid", False), prep)
    return (DataLoader(train, batch_size, shuffle=True, num_workers=2),
            DataLoader(valid, batch_size, shuffle=False, num_workers=2))

def get_test(fold, batch_size=32):
    test = datasets.ImageFolder(find_dir(fold, "Test", False), prep)
    return DataLoader(test, batch_size, shuffle=False, num_workers=2), test.classes