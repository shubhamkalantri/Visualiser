import os
import encoder_transformers
import encoder_open_clip
from plotter import plotter

hf_model = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
text = True
image_folder = ""
threeD = False
num_samples = 100

categories = [x[1] for x in os.walk(image_folder)][0]

t = encoder_transformers.encoder(hf_model)
o = encoder_open_clip.encoder(hf_model)
p = plotter(threeD)


image_features = o.process_and_encode(image_folder, categories, num_samples, proj=text)
text_features = o.get_text_features(text)

prefix = hf_model.replace("hf-hub:", "").replace("/", "-")

fname = f"{prefix}_tsne_3d.png" if threeD else f"{prefix}_tsne.png"


p.plot(image_features, text_features, fname, num_samples, categories)