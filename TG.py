import cv2

# Carregar a imagem
img = cv2.VideoCapture(0)
ret, img = img.read()

# Verificar se a imagem foi carregada corretamente
if img is None:
    print("Erro ao carregar a imagem.")

cv2.imshow("Imagem original", img)
cv2.waitKey(0)

# Reduzir o ruído
img = cv2.GaussianBlur(img, (5, 5), 0)

# Converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Segmentar a imagem para isolar a caixa
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Segmentar a imagem usando o método de Canny
edges = cv2.Canny(gray, 50, 150)

# Fechar as bordas para formar uma forma fechada
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
cv2.imshow("Imagem com contorno", img)
cv2.waitKey(0)
cv2.destroyAllWindows()