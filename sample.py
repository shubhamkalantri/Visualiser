import os
import encoder_transformers
# import encoder_open_clip
from plotter import plotter

hf_model = ""
text = False
image_folder = ""
threeD = True
num_samples = 23

categories = [x[1] for x in os.walk(image_folder)][0]

texts = ["" for _ in range(len(categories))]

for i in range(len(categories)):
    texts[i] = categories[i].replace("", "")

t = encoder_transformers.encoder(hf_model)
# o = encoder_open_clip.encoder(hf_model)
p = plotter(threeD)


image_features = t.process_and_encode(image_folder, categories, num_samples, proj=text)
if text:
    text_features = t.get_text_features(texts)
else:
    text_features = None

prefix = hf_model.replace("hf-hub:", "").replace("/", "-")

fname = f"{prefix}_tsne_3d.png" if threeD else f"{prefix}_tsne.png"


p.plot(image_features, text_features, fname, num_samples, texts)