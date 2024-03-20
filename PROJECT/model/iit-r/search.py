import faiss
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np


import os
os.environ['http_proxy'] = 'http://'
os.environ['https_proxy'] = 'https://'

# Define a dataset class
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

# Define a transformation to preprocess the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# image_dir = 'upload_folder' 
dataset = ImageDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load the pre-trained resnet101 model
resnet101 = models.resnet101(pretrained=True)
resnet101.eval()  # Set the model to evaluation mode

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet101.to(device)

# Function to extract features
def extract_features(dataloader, model, device):
    features = []
    paths = []
    with torch.no_grad():
        for inputs, image_paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            paths.extend(image_paths)
    return np.vstack(features), paths

# Extract features from the image dataset
image_features, image_paths = extract_features(dataloader, resnet101, device)

# Save the features and image paths for later use
np.save('image_features.npy', image_features)
with open('image_paths.txt', 'w') as f:
    for path in image_paths:
        f.write("%s\n" % path)

# Number of features and dimension
num_features = image_features.shape[0]
feature_dim = image_features.shape[1]

# Normalize the features (L2 normalization is important for cosine similarity)
faiss.normalize_L2(image_features)

# Build a FAISS index
index = faiss.IndexFlatL2(feature_dim)  # Using L2 distance for similarity
index.add(image_features)  # Add the image features to the index

# Save the index for later use
faiss.write_index(index, 'image_features.index')

# Function to perform a query
def search(query_image_path, transform, model, index, top_k=10):
    # Load and transform the query image
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = transform(query_image)
    query_image = query_image.unsqueeze(0).to(device)  # Add batch dimension and send to device

    # Extract features from the query image
    model.eval()
    with torch.no_grad():
        query_features = model(query_image).cpu().numpy()
    faiss.normalize_L2(query_features)  # L2 normalization

    # Perform the search
    distances, indices = index.search(query_features, top_k)
    scores = 1 / (1 + distances)

    

    # Return the paths of the top_k closest images
    return [image_paths[i] for i in indices[0]], scores[0]

# Load the index and image paths (assuming they were saved previously)
index = faiss.read_index('image_features.index')
with open('image_paths.txt', 'r') as f:
    image_paths = f.read().splitlines()
    



def runner2(query_image_path,image_dir):

    # Perform the search
    similar_image_paths, similarity_scores = search(query_image_path, transform, resnet101, index)
    res = {}
    # Print the paths of the top 30 similar images
    for path, score in zip(similar_image_paths, similarity_scores):
        print(f"Image path: {path}, Similarity score: {(score-0.80)/0.2}")
        score = (score-0.80)/0.2
        res[path] = score

    return res