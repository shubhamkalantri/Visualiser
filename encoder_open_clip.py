import torch
import numpy as np
from tqdm import tqdm
import open_clip
from PIL import Image
import os
from random import shuffle

class encoder():
    def __init__(self, model_name, pretrained=None):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
    
    def process_images(self, image_folder, categories, num_samples):
        images_processed = []
        for category in tqdm(categories, desc="Processing images"):
            flist = os.listdir(os.path.join(image_folder, category))
            shuffle(flist)
            for image in tqdm(flist[:num_samples], desc=f"Processing folder {category}", leave=False):
                images_processed.append(self.preprocess(Image.open(os.path.join(image_folder, category, image))).unsqueeze(0))
        return images_processed

    def get_image_features(self, images_processed, proj=True):
        image_features = []

        with torch.no_grad():
            for image in tqdm(images_processed, desc="Getting image features"):
                image.to("cuda")
                if proj:
                    image_features.append(self.model.encode_image(image).cpu().detach().numpy())
                else:
                    image_features.append(self.model.visual.trunk(image).cpu().detach().numpy())
                image.to("cpu")

        image_features = np.squeeze(np.array(image_features))
        
        return image_features
    
    def process_and_encode(self, image_folder, categories, num_samples, proj=True):
        images_processed = self.process_images(image_folder, categories, num_samples)
        image_features = self.get_image_features(images_processed, proj)
        return image_features