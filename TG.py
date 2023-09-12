import cv2
from matplotlib import pyplot as plt

# Load image
img = cv2.VideoCapture(0)
ret, img = img.read()
# Check for loading
if img is None:
    print("Erro ao carregar a imagem.")
else:
    chave = img # Making copies for later usage
    original = img 

    # Applying for less noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Grayscale convertion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Segmentation of the image to show only the box
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Canny method
    edges = cv2.Canny(gray, 20, 80)
    # Closing borders to make a closed figure 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Countours finding
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Finding the biggest countour
    largest_contour = max(contours, key=cv2.contourArea)

    # Area calcule
    area = cv2.contourArea(largest_contour)
    print("Área da caixa: ", area)

    # Image checking
    cv2.drawContours(img, [largest_contour], 0, (0, 255, 0), 2)

    # Keypoint start
    detector = cv2.ORB_create()
    # Keypoints detection
    keypoints = detector.detect(gray, None)
    # Drawing the keypoints
    img_with_keypoints = cv2.drawKeypoints(chave, keypoints, None, color=(0, 255, 0))

    fig = plt.figure(figsize=(10, 7)) # Plotting

    # Setting values to rows and column variables
    rows = 2
    columns = 2 
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    # Showing image
    plt.imshow(original)
    plt.axis('off')
    plt.title("Original")
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    # Showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("Bordas")
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    # Showing image
    plt.imshow(img_with_keypoints)
    plt.axis('off')
    plt.title("Keypoints")
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 4)
    # Showing image
    plt.imshow(img_with_keypoints)
    plt.axis('off')
    plt.title("Keypoints")

    plt.show() # Final plot