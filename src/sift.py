import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('data/images/city_hall/city_hall1.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()

img2 = cv2.imread('data/images/city_hall/city_hall2.jpg', cv2.IMREAD_GRAYSCALE)

keypoints_im1, descriptors_im1 = sift.detectAndCompute(img1, None)

filtered_keypoints = [kp for kp in keypoints_im1 if kp.response > 0.05]
#print(len(filtered_keypoints))
img_with_keypoints = cv2.drawKeypoints(img1, filtered_keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Affichage avec matplotlib
'''plt.figure(figsize=(10, 8))
plt.imshow(img_with_keypoints, cmap='gray')
plt.title("Keypoints détectés avec SIFT")
plt.axis('off')
plt.show()'''

keypoints_im2, descriptors_im2 = sift.detectAndCompute(img2, None)

filtered_keypoints = [kp for kp in keypoints_im2 if kp.response > 0.05]
#print(len(filtered_keypoints))
img_with_keypoints = cv2.drawKeypoints(img2, filtered_keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
'''
# Affichage avec matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(img_with_keypoints, cmap='gray')
plt.title("Keypoints détectés avec SIFT")
plt.axis('off')
plt.show()'''

def match(desc1,desc2):
    """Foreachdescriptorinthefirstimage,
    selectitsmatchinthesecondimage.
    input:desc1(descriptorsforthefirstimage),
    desc2(sameforsecondimage)."""
    desc1=np.array([d/np.linalg.norm(d) for d in desc1])
    desc2=np.array([d/np.linalg.norm(d) for d in desc2])
    dist_ratio=0.5
    desc1_size=desc1.shape
    matchscores=np.zeros((desc1_size[0],1),'int')
    desc2t=desc2.T#precomputematrixtranspose
    for i in range (desc1_size[0]):
        dotprods=np.dot(desc1[i,:],desc2t)#vectorofdotproducts
        dotprods=0.9999*dotprods
        #inversecosineandsort,returnindexforfeaturesinsecondimage
        indx=np.argsort(np.arccos(dotprods))
        #checkifnearestneighborhasanglelessthandist_ratiotimes2nd
        if np.arccos(dotprods)[indx[0]]<dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i]=int(indx[0])
    return matchscores

def match_twosided(desc1,desc2):
    """ Two-sided symmetric version of match(). """
    matches_12 = match(desc1,desc2)
    matches_21 = match(desc2,desc1)
    ndx_12 = matches_12.nonzero()[0]
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    return matches_12
    

matches = match_twosided(descriptors_im1, descriptors_im2)

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1,im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matchscores):
    """Affiche les deux images côte à côte avec des lignes bleues entre les keypoints appariés."""

    # Combine les deux images horizontalement
    im3 = appendimages(im1, im2)

    # Affiche l'image combinée
    plt.figure(figsize=(12, 8))
    plt.imshow(im3, cmap='gray')
    plt.title("Correspondances entre points clés (SIFT)")
    plt.axis('off')

    # Décale horizontalement les points de la deuxième image
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            pt1 = (int(locs1[i].pt[0]), int(locs1[i].pt[1]))
            pt2 = (int(locs2[m[0]].pt[0] + cols1), int(locs2[m[0]].pt[1]))
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b', linewidth=1)

    plt.show()

print(f"Total of matches : {len(matches)}")
plot_matches(img1,img2, keypoints_im1, keypoints_im2, matches)