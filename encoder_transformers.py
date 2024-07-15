import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import os
from random import shuffle

class encoder():
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to("cuda")
        self.model.eval()
    
    def process_images(self, image_folder, categories, num_samples):
        images_processed = []
        for category in tqdm(categories, desc="Processing images"):
            flist = os.listdir(os.path.join(image_folder, category))
            shuffle(flist)
            for image in tqdm(flist[:num_samples], desc=f"Processing folder {category}", leave=False):
                images_processed.append(self.processor(images=Image.open(os.path.join(image_folder, category, image)), return_tensors="pt"))
        return images_processed

    def get_image_features(self, images_processed, proj=True):
        image_features = []

        with torch.no_grad():
            for image in tqdm(images_processed, desc="Getting image features"):
                image.to("cuda")
                if proj:
                    image_features.append(self.model.get_image_features(**image).cpu().detach().numpy())
                else:
                    image_features.append(np.squeeze(self.model.vision_model(**image)[0].cpu().detach().numpy())[0])
                image.to("cpu")

        image_features = np.squeeze(np.array(image_features))

        return image_features
    
    def get_text_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs).cpu().detach().numpy()
        return text_features
    
    def process_and_encode(self, image_folder, categories, num_samples, proj=True):
        images_processed = self.process_images(image_folder, categories, num_samples)
        image_features = self.get_image_features(images_processed, proj)
        return image_features
    