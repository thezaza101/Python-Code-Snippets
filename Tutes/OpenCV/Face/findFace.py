import cv2
import matplotlib.pyplot as plt

# Get a pointer to the devides 
camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detectFaces(rgbImg):
    gray = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)    

def drawFaceBB(rgbImg, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(rgbImg, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return rgbImg

def getFrame():
    return_value, image = camera.read()
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

def displayIm(img):   
    plt.clf()
    plt.imshow(img)  
    plt.pause(.05)

if __name__ == "__main__":
    while True:
        try:
            image = cv2.resize(getFrame(),None,fx=0.5,fy=0.5)
            faces = detectFaces(image)
            output = drawFaceBB(image,faces)
            displayIm(output)
        except:
            pass

    camera.release()