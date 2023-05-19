from models import get_mtcnn
from utils import collate_pil
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import *

def preprocess_data(args):
    """
    Detect faces in CASIA dataset.
    """
    batch_size = args.batch_size
    # Get MTCNN for face detection
    mtcnn = get_mtcnn()
    data = datasets.ImageFolder(CASIA_PATH, transform=transforms.Resize((512, 512)))
    data.samples = [
        (p, p.replace(CASIA_PATH, casia_cropped_path))
        for p, _ in data.samples
    ]

    loader = DataLoader(
        data,
        batch_size=batch_size,
        collate_fn=collate_pil,
    )

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print(f'\rBatch {i+1}/{len(loader)}', end='')

    print('\nFinish Preprocessing')
    print('='*20)

    del mtcnn