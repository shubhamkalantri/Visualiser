import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class plotter():
    def __init__(self, threeD):
        self.threeD = threeD
        self.tsne = TSNE(n_components=3 if self.threeD else 2)

    def plot(self, image_features, text_features, fname, num_samples, categories):
        results = self.tsne.fit_transform(image_features + text_features)
        fig = plt.figure()
        numCats = len(categories)
        if self.threeD:
            ax = fig.add_subplot(projection='3d')
            for i in range(len(categories)):
                plt.scatter(results[-(numCats -i), 0], results[-(numCats -i), 1], results[-(numCats -i), 2], label="text" + categories[-(numCats -i)])
            for i, category in enumerate(categories):
                ax.scatter(results[i*num_samples:(i+1)*num_samples, 0], results[i*num_samples:(i+1)*num_samples, 1], results[i*num_samples:(i+1)*num_samples, 2], label=category)
                plt.legend()
                plt.savefig(fname)
        else:
            for i in range(len(categories)):
                plt.scatter(results[-(numCats -i), 0], results[-(numCats -i), 1], label="text" + categories[-(numCats -i)])
            for i, category in enumerate(categories):
                plt.scatter(results[i*num_samples:(i+1)*num_samples, 0], results[i*num_samples:(i+1)*num_samples, 1], label=category)
                plt.legend()
                plt.savefig(fname)
