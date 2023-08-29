import cv2
from matplotlib import pyplot as plt

# Carregar a imagem
img = cv2.VideoCapture(0)
ret, img = img.read()

# Verificar se a imagem foi carregada corretamente
if img is None:
    print("Erro ao carregar a imagem.")

chave = img
original = img

# Reduzir o ruído
img = cv2.GaussianBlur(img, (5, 5), 0)

# Converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Segmentar a imagem para isolar a caixa
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Segmentar a imagem usando o método de Canny
edges = cv2.Canny(gray, 20, 80)

# Fechar as bordas para formar uma forma fechada take 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Encontrar os contornos da caixa
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encontrar o maior contorno
largest_contour = max(contours, key=cv2.contourArea)

# Calcular a área do maior contorno
area = cv2.contourArea(largest_contour)

# Exibir a área
print("Área da caixa: ", area)

# Exibir a imagem com o contorno da caixa destacado
cv2.drawContours(img, [largest_contour], 0, (0, 255, 0), 2)

# Inicializar o detector de keypoints (SURF, ORB, etc.)
detector = cv2.ORB_create()

# Detectar keypoints
keypoints = detector.detect(gray, None)

# Desenhar keypoints na imagem
img_with_keypoints = cv2.drawKeypoints(chave, keypoints, None, color=(0, 255, 0))

fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 2
columns = 2 

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(original)
plt.axis('off')
plt.title("Original")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Bordas")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(img_with_keypoints)
plt.axis('off')
plt.title("Keypoints")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(img_with_keypoints)
plt.axis('off')
plt.title("Keypoints")

plt.show()