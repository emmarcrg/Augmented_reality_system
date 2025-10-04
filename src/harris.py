import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def preprocess_image(img_path, target_size=None, sigma=1):
    """
    Standard preprocessing:
    - grayscale
    - resize (optional)
    - normalize (zero mean, unit variance)
    - gaussian smoothing
    """
    im = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
    if target_size:
        im = np.array(Image.fromarray(im).resize(target_size, Image.BICUBIC), dtype=np.float32)
    im = (im - np.mean(im)) / (np.std(im) + 1e-8)
    return gaussian_filter(im, sigma=sigma)


def compute_harris_response(im, sigma=3):
    """ Harris corner response. """
    imx = np.zeros_like(im)
    imy = np.zeros_like(im)
    gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    Wxx = gaussian_filter(imx * imx, sigma)
    Wxy = gaussian_filter(imx * imy, sigma)
    Wyy = gaussian_filter(imy * imy, sigma)

    Wdet = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy
    return Wdet / (Wtr + 1e-8)


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Extract Harris points above threshold with min distance. """
    coords = np.array((harrisim > harrisim.max() * threshold).nonzero()).T
    values = harrisim[coords[:, 0], coords[:, 1]]
    index = np.argsort(values)[::-1]  # tri décroissant

    allowed = np.zeros(harrisim.shape, dtype=bool)
    allowed[min_dist:-min_dist, min_dist:-min_dist] = True

    selected = []
    for i in index:
        y, x = coords[i]
        if allowed[y, x]:
            selected.append((y, x))
            allowed[y - min_dist:y + min_dist + 1, x - min_dist:x + min_dist + 1] = False
    return selected


def get_descriptors(image, coords, wid=5):
    """ Extract patch descriptors around each point. """
    desc = []
    for y, x in coords:
        patch = image[y - wid:y + wid + 1, x - wid:x + wid + 1]
        if patch.shape == (2 * wid + 1, 2 * wid + 1):  # ignore borders
            desc.append(patch.flatten())
    return np.array(desc)


def plot_harris_points(image, coords):
    plt.figure(figsize=(6, 6))
    plt.gray()
    plt.imshow(image)
    if coords:
        plt.plot([x for _, x in coords], [y for y, _ in coords], 'c*')
    plt.title(f"Harris Points: {len(coords)}")
    plt.axis('off')
    plt.show()


def preprocess_harris_dataset(data_dir, feature_dir, harris_threshold=0.3, wid=5):
    os.makedirs(feature_dir, exist_ok=True)

    for landmark in os.listdir(data_dir):
        landmark_path = os.path.join(data_dir, landmark)
        if not os.path.isdir(landmark_path):
            continue

        out_path = os.path.join(feature_dir, landmark)
        os.makedirs(out_path, exist_ok=True)

        for img_name in os.listdir(landmark_path):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(landmark_path, img_name)
            im = preprocess_image(img_path)
            harrisim = compute_harris_response(im)
            coords = get_harris_points(harrisim, min_dist=wid + 1, threshold=harris_threshold)
            descriptors = get_descriptors(im, coords, wid)

            plot_harris_points(im, coords)

            out_file = os.path.join(out_path, img_name.rsplit('.', 1)[0] + '.txt')
            with open(out_file, 'w') as f:
                for c, d in zip(coords, descriptors):
                    line = ','.join(map(str, [c[0], c[1]] + list(d)))
                    f.write(line + '\n')

            print(f"Processed {img_name} ({landmark}), {len(coords)} points.")


def match(desc1, desc2, locs1, locs2, ncc_thresh=0.6, dist_thresh=200, ratio=0.8):
    """ Match descriptors using NCC with distance & ratio tests. """
    matches = []
    n = desc1.shape[1]

    for i, d1 in enumerate(desc1):
        d1n = (d1 - np.mean(d1)) / (np.std(d1) + 1e-8)
        scores = []

        for j, d2 in enumerate(desc2):
            if np.linalg.norm(np.array(locs1[i]) - np.array(locs2[j])) > dist_thresh:
                continue
            d2n = (d2 - np.mean(d2)) / (np.std(d2) + 1e-8)
            ncc = np.sum(d1n * d2n) / (n - 1)
            if ncc > ncc_thresh:
                scores.append((j, 1 - ncc))

        if len(scores) >= 2:
            scores.sort(key=lambda x: x[1])
            best, second = scores[:2]
            if best[1] / (second[1] + 1e-8) < ratio:
                matches.append((i, best[0]))

    return matches


def load_features(feature_dir):
    """ Load coords + descriptors from txt files. """
    features = {}
    for landmark in os.listdir(feature_dir):
        l_path = os.path.join(feature_dir, landmark)
        if not os.path.isdir(l_path):
            continue
        features[landmark] = {}
        for f in os.listdir(l_path):
            if not f.endswith('.txt'):
                continue
            coords, desc = [], []
            with open(os.path.join(l_path, f)) as file:
                for line in file:
                    vals = list(map(float, line.strip().split(',')))
                    coords.append([int(vals[0]), int(vals[1])])
                    desc.append(np.array(vals[2:]))
            features[landmark][f] = (coords, np.array(desc))
    return features


def classify_image(new_img_path, features, ncc_thresh=0.6, harris_threshold=0.2, wid=5, dist_thresh=500, ratio=0.8):
    im = preprocess_image(new_img_path)
    harrisim = compute_harris_response(im)
    coords_new = get_harris_points(harrisim, min_dist=wid + 1, threshold=harris_threshold)
    desc_new = get_descriptors(im, coords_new, wid)

    plot_harris_points(im, coords_new)

    best_score, best_landmark, best_image = -1, None, None
    for landmark, images in features.items():
        for img_name, (coords_ref, desc_ref) in images.items():
            if len(desc_ref) == 0 or len(desc_new) == 0:
                continue
            matches = match(desc_new, desc_ref, coords_new, coords_ref,
                            dist_thresh=dist_thresh, ratio=ratio, ncc_thresh=ncc_thresh)
            num = len(matches)
            print(f"Comparing {img_name} ({landmark}): {num} matches")

            if num > best_score:
                best_score, best_landmark, best_image = num, landmark, img_name

    return best_landmark, best_image, best_score


def plot_image_matches(img1_path, img2_path, ncc_thresh=0.6, harris_threshold=0.3, wid=5, dist_thresh=200, ratio=0.8):
    """
    Affiche deux images avec leurs points Harris détectés et les correspondances (matches).
    """
    # Prétraitement
    im1 = preprocess_image(img1_path)
    im2 = preprocess_image(img2_path)

    # Harris + descripteurs
    harrisim1 = compute_harris_response(im1)
    harrisim2 = compute_harris_response(im2)
    coords1 = get_harris_points(harrisim1, min_dist=wid + 1, threshold=harris_threshold)
    coords2 = get_harris_points(harrisim2, min_dist=wid + 1, threshold=harris_threshold)
    desc1 = get_descriptors(im1, coords1, wid)
    desc2 = get_descriptors(im2, coords2, wid)

    # Match
    matches = match(desc1, desc2, coords1, coords2,
                    ncc_thresh=ncc_thresh, dist_thresh=dist_thresh, ratio=ratio)

    # Plot côte à côte
    plt.figure(figsize=(12, 6))
    plt.gray()

    # Concaténer horizontalement
    h1, w1 = im1.shape
    h2, w2 = im2.shape
    h = max(h1, h2)
    concat_img = np.zeros((h, w1 + w2))
    concat_img[:h1, :w1] = im1
    concat_img[:h2, w1:w1 + w2] = im2
    plt.imshow(concat_img)

    # Points Harris
    for y, x in coords1:
        plt.plot(x, y, 'ro', markersize=3)
    for y, x in coords2:
        plt.plot(x + w1, y, 'ro', markersize=3)

    # Lignes de correspondance
    for i, j in matches:
        y1, x1 = coords1[i]
        y2, x2 = coords2[j]
        plt.plot([x1, x2 + w1], [y1, y2], 'c-', linewidth=0.7)

    plt.title(f"{len(matches)} matches trouvés")
    plt.axis('off')
    plt.show()

    return matches

if __name__ == "__main__":
    data_dir = r"data\images"
    feature_dir = r"data\features"

    ncc_thresh = 0.5
    ratio = 0.5
    dist_thresh = 200
    harris_treshold = 0.2
    #preprocess_harris_dataset(data_dir, feature_dir, harris_threshold=harris_treshold, wid=5)
    features = load_features(feature_dir)

    new_img = r"fram_test.avif"
    landmark, ref_img, score = classify_image(new_img, features, ncc_thresh=ncc_thresh, dist_thresh=dist_thresh, ratio=ratio)

    print(f"Nouvelle image classée comme : {landmark}")
    print(f"Meilleure référence : {ref_img}")
    print(f"Nombre de correspondances : {score}")
    
    img1 = r"data\images\opera\opera1.jpg"
    img2 = r"data\images\opera\opera2.jpg"
    #matches = plot_image_matches(img1, img2, ncc_thresh=ncc_thresh, ratio=ratio, dist_thresh=dist_thresh,harris_threshold=harris_treshold, wid=5)

