import cv2
import matplotlib.pyplot as plt

# Get a pointer to the devides 
camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detectFaces(bgrImg):
    gray = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)    

def drawFaceBB(bgrImg, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(bgrImg, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return bgrImg

def getFrame():
    return_value, image = camera.read()
    return image
    

def displayIm(img):   
    cv2.imshow("Tracking", img)

if __name__ == "__main__":
    while True:
        image = cv2.resize(getFrame(),None,fx=0.5,fy=0.5)
        faces = detectFaces(image)
        output = drawFaceBB(image,faces)
        displayIm(output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()