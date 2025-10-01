import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# %%  
def convert_to_grayscale(im):
    if im.ndim == 3:
        im = np.mean(im, axis=2)
    return im

# %%  
def compute_harris_response(im, sigma=3):
    """ Compute the Harris corner detector response function
    for each pixel in a graylevel image. """
    # derivatives
    imx = np.zeros(im.shape)
    gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = np.zeros(im.shape)
    gaussian_filter(im, (sigma,sigma), (1,0), imy)
    # compute components of the Harris matrix
    Wxx = gaussian_filter(imx*imx, sigma)
    Wxy = gaussian_filter(imx*imy, sigma)
    Wyy = gaussian_filter(imy*imy, sigma)
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    return Wdet / Wtr

# %%  
def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """
    # find top corner candidates above a threshold
    coords = np.array((harrisim > harrisim.max()*threshold).nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    # sort candidates
    index = np.argsort(candidate_values)
    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                              (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return filtered_coords

# %%  
def get_descriptors(image, filtered_coords, wid=5):
    """ For each point return pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid). """
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                      coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    return desc

# %%  
def preprocess_harris_dataset(data_dir, feature_dir, harris_threshold=0.3, wid=5):
    """
    Parcours toutes les images dans data_dir, standardise, détecte Harris et extrait les descripteurs.
    Sauvegarde coords + descriptors dans un fichier .txt par image dans feature_dir.
    
    data_dir : dossier contenant les dossiers de landmarks
    feature_dir : dossier où stocker les features
    """
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    for landmark in os.listdir(data_dir):
        landmark_path = os.path.join(data_dir, landmark)
        if not os.path.isdir(landmark_path):
            continue

        out_landmark_path = os.path.join(feature_dir, landmark)
        if not os.path.exists(out_landmark_path):
            os.makedirs(out_landmark_path)

        for img_name in os.listdir(landmark_path):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(landmark_path, img_name)
            im = np.array(Image.open(img_path).convert('L'))
            
            # Standardisation : mettre les pixels centrés autour de 0 avec écart-type 1
            im_std = (im - np.mean(im)) / (np.std(im) + 1e-8)
            
            # Harris
            harrisim = compute_harris_response(im_std)
            coords = get_harris_points(harrisim, min_dist=wid+1, threshold=harris_threshold)
            descriptors = get_descriptors(im_std, coords, wid)

            # Sauvegarde dans .txt
            out_file = os.path.join(out_landmark_path, img_name.split('.')[0] + '.txt')
            with open(out_file, 'w') as f:
                for c, d in zip(coords, descriptors):
                    line = ','.join(map(str, [c[0], c[1]] + list(d)))
                    f.write(line + '\n')

            print(f"Processed {img_name} for {landmark}, {len(coords)} points saved.")
