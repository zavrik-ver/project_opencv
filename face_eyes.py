import cv2


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eye = cv2.CascadeClassifier('open_cv\eye.xml')
    
    results2 = eye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

    
   
    for (x,y,w,h) in results2:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), thickness=1)

    

    if not ret:
        break    

    cv2.imshow("result", img)
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break

cap.release()
cv2.destroyAllWindows()