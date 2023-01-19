from numpy import* 
import cv2
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
def faceExtractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceClassifier.detectMultiScale(gray, 1.3, 5)
    if face is None:
        return None
    for (x,y,w,h) in face:
        croppedImage = img[y:y+h+50, x:x+w+50]
        return croppedImage
        
cap = cv2.VideoCapture(0)
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if faceExtractor(frame) is not None:
        faces = cv2.resize(faceExtractor(frame), (300,300))
        count += 1
        if (count<400):
            faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
            fileName = "C:/Users/asus/Desktop/vision/face_recognition/train/arpit/"+str(count)+".jpg"
            cv2.imwrite(fileName, faces)
            cv2.imshow("FaceCropper", faces)
        else:
            faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
            fileName = "C:/Users/asus/Desktop/vision/face_recognition/test/arpit/"+str(count)+".jpg"
            cv2.imwrite(fileName, faces)
            cv2.imshow("FaceCropper", faces)
    else:
        print("Face not found")
    if(cv2.waitKey(1)==13 or count==500 ):
        break
cap.release()
cv2.destroyAllWindows()
print("Your dataset is collected now.")