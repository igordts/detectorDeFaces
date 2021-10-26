import cv2
 
def redim(img, largura):
  #alt = int(img.shape[0] / img.shape[1] * largura)
  #img = cv2.resize(img, (largura, alt), interpolation = cv2.INTER_AREA)
  return img
 
#Criar o classificador a partir de parametros ja balanceados
df = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Abre o video gravado em disco
camera = cv2.VideoCapture('ednaldo.mp4')
#camera = cv2.VideoCapture(0)
 
while True:
  (sucesso, frame) = camera.read()
  if not sucesso: #falha na leitura ou video acabou
    break
  frame = redim(frame, 320)
  frame_tons_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = df.detectMultiScale(frame_tons_cinza, scaleFactor = 1.3, minNeighbors=3)
  frame_temp = frame.copy()
  for(x, y, lar, alt) in faces:
    cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0, 0, 255), 2)
  cv2.imshow("Detector de faces", redim(frame_temp, 640))
  if cv2.waitKey(1) & 0xFF == ord("s"):
    break
 
camera.release()
cv2.destroyAllWindows()