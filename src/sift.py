import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/images/city_hall/city_hall_1.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
if img is None:
    raise FileNotFoundError("L’image n’a pas été trouvée ou n’a pas pu être chargée.")
else : 
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Affichage avec matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("Keypoints détectés avec SIFT")
    plt.axis('off')
    plt.show()

img = cv2.imread('data/images/city_hall/city_hall_2.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
if img is None:
    raise FileNotFoundError("L’image n’a pas été trouvée ou n’a pas pu être chargée.")
else :
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Affichage avec matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("Keypoints détectés avec SIFT")
    plt.axis('off')
    plt.show()