import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


transform = transforms.Compose([
    transforms.ToTensor()
])

# train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

def generate_data():
    train_dataset = datasets.MNIST(root='data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=False, transform=transform)
    return train_dataset, test_dataset

def MNIST_tagging_function(dataset):
    possible_tags = ["rotate60", "rotate45", "shrink", "expand", "none"]

    shrink_and_pad_transform = transforms.Compose([
    transforms.Resize((14, 14)),          # Resize to shrink the content
    transforms.Pad(padding=7, fill=1)   # Pad with white pixels (fill=255 for white)
        ])
    expand_and_clip_transform = transforms.Compose([
    transforms.Resize((56, 56)),          # Expand the content to a larger size
    transforms.CenterCrop((28, 28))       # Crop back to 28x28, clipping the edges
        ])

    tag_transforms = [transforms.RandomRotation(60), transforms.RandomRotation(45), 
                      shrink_and_pad_transform, expand_and_clip_transform, lambda x: x]
    
    tags_dict = {}

    for i in range(len(possible_tags)):
        tags_dict[i] = possible_tags[i]

    tags = np.random.choice([0,1,2,3,4], size=dataset.data.shape[0],
                             p=[0.15, 0.15, 0.15, 0, 0.55])
    PIL = transforms.ToPILImage()
    TENSOR = transforms.ToTensor()

    transformed_data = torch.zeros_like(dataset.data, dtype=torch.uint8)

    for i in tqdm(range(dataset.data.shape[0])):
        
        tag_idx = tags[i]
        transform = tag_transforms[tag_idx]
        image, _ = dataset[i]
        
        # Convert to PIL, apply transformation, and back to tensor
        transformed_image = transform(PIL(image))
        transformed_image = TENSOR(transformed_image) * 255  # Scale back to 0-255
        transformed_data[i] = transformed_image.byte()       # Store as 8-bit integers
    

    return transformed_data, tags, tags_dict

