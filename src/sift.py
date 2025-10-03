
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import SIFT, match_descriptors, plot_matched_features
from skimage import transform
from PIL import Image
import numpy as np

# Load the datafiles and make them grayscale
def load_image(path):
    return rgb2gray(np.array(Image.open(path)))

# Extract SIFT keypoints and descriptors
def extract_sift_features(image):
    sift = SIFT()
    sift.detect_and_extract(image)
    return sift.keypoints, sift.descriptors

# Visualize the matches
def match_and_plot(img1, img2, keypoints1, descriptors1, keypoints2, descriptors2, title):
    matches = match_descriptors(descriptors1, descriptors2, max_ratio=0.6, cross_check=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.gray()
    plot_matched_features(img1, img2, keypoints1, keypoints2, matches, ax=ax)
    ax.axis('off')
    ax.set_title(title)
    plt.show()

# 🧪 Exemple d'utilisation avec tes images
img1 = load_image('data/images/city_hall/city_hall1.jpg')
img2 = load_image('data/images/city_hall/city_hall2.jpg')

keypoints1, descriptors1 = extract_sift_features(img1)
keypoints2, descriptors2 = extract_sift_features(img2)

match_and_plot(img1, img2, keypoints1, descriptors1, keypoints2, descriptors2, "Image 1 vs Image 2")