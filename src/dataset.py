# Question 1

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from wandb_setup import setup_wandb

wandb = setup_wandb(run_name="dataset")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Collect one image per class
sample_images = []
for class_id in range(10):
    indices = np.where(y_train == class_id)[0]
    if len(indices) > 0:
        sample_images.append(x_train[indices[0]])

# Plot the images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Fashion-MNIST Dataset Sample Images", fontsize=14)

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.set_title(class_labels[i])
    ax.axis("off")

plt.show()

np.savez("fashion-mnist.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
wandb.log({"sample_images": [wandb.Image(img, caption=f"Class {class_labels[i]}") for i, img in enumerate(sample_images)]})
wandb.finish()