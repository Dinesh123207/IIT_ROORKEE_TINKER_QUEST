import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from elasticsearch import Elasticsearch

model = models.resnet101(weights=True)
model.eval()  # Set the model to evaluation mode
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.es = Elasticsearch(
        cloud_id="jello:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDJiMzljMGI3ZDE2MjQxNmViYWQ2NWExZmRhMzkwNWU5JGNjNzk0M2E4YjI3YjQ0YzdhNmZjMDIzZDZlZTU5Zjdk",
    basic_auth=( "elastic","qvLuM6QXURj9xf729jVjydTx")
    )
        self.index_name = "wewewe"
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

    def indices_create(self, index_name):
        # Define the index settings and mappings
        settings = {
            "settings": {
                "number_of_shards": 1,  # Adjust based on your cluster setup
                "number_of_replicas": 0  # Adjust for production
            },
            "mappings": {
                "properties": {
                    "feature_vector": {
                        "type": "dense_vector",
                        "dims": 1000  # The dimensionality of the ResNet101 feature vector
                    },
                    "image_path": {
                        "type": "keyword"
                    }
                }
            }
        }

        # Create the index
        self.es.indices.create(index=index_name, body=settings)
 
    def Load_the_dataset(self):
        image_dir = 'D:\Shashank_SIH\SIH_eval (2)\SIH_eval'  # Change to your image directory
        dataset = ImageDataset(image_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        return dataloader

    # Function to extract features
    def extract_features(self,dataloader, model, device):
        features = []
        paths = []
        with torch.no_grad():
            for inputs, image_paths in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                features.append(outputs.cpu().numpy())
                paths.extend(image_paths)
        return np.vstack(features), paths
    
    

    def index_images(self,features, image_paths):
        for i in range(len(features)):
            document = {
                "feature_vector": features[i].tolist(),  # Convert numpy array to list
                "image_path": image_paths[i]
            }
            self.es.index(index=self.index_name, body=document)


    def search_similar_images(self,query_image_path, transform, model):
        # Load and transform the query image
        query_image = Image.open(query_image_path).convert('RGB')
        query_image = transform(query_image)
        query_image = query_image.unsqueeze(0).to(device)  # Add batch dimension and send to device

        # Extract features from the query image
        model.eval()
        with torch.no_grad():
            query_features = model(query_image).cpu().numpy()
        test_search_body = {
        "size": 30,
        "_source":["image_path","_score"],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script":{
                "source": "cosineSimilarity(params.query_vector, 'feature_vector') + 1.0",
                    "params": {"query_vector": query_features[0]}
                }
            }
        }
        }
        l=[]

        response = self.es.search(index=self.index_name, body=test_search_body)
        for hit in response['hits']['hits']:
            l.append([hit['_source']['image_path'],hit['_score']/2])
        return l



# Define a transformation to preprocess the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


query_image_path = "aks_sample.jpeg"
image_dir = "D:\Shashank_SIH\SIH_eval (2)\SIH_eval"
    
p=ImageDataset(image_dir,transform)
features,paths=p.extract_features(DataLoader,model,device)
index_images(features,paths)
l=p.search_similar_images(query_image_path,transform,model)
print(l)
    
