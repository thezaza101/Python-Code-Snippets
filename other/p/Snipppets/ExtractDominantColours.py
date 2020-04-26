import imageio as imio # Reading images
import numpy as np # Matrix operations
import pandas as pd # Matrix operations
import matplotlib.pyplot as plt # for visualisation
from sklearn.cluster import KMeans # Kmeans 
import math # For math
import cv2 # Webcam feed

# Get a pointer to the devides 
camera = cv2.VideoCapture(0)

def getFrame(file=None):
    # This function retuens a webcam feame if a path is not specified, 
    # otherwise returns the image
    if (file is None):
        return_value, image = camera.read()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return imio.imread(file)
    

def displayIm(imList, nCols = 5, hideAxis=False):
    # This function displays multiple images in a grid with with <= nCols
    # The input must be a list of images

    numIm = len(imList)
    numRows = math.ceil(numIm/nCols)
    
    if ((numIm < nCols) and not (numIm > nCols)):
        ncols = numIm
    else:
        ncols = nCols       
    
    if (numIm >1):
        fig, axes = plt.subplots(nrows=numRows, ncols=ncols) 
        ax = axes.ravel()    
        [ax[i].imshow(imList[i]) for i in range(numIm)]
        if (hideAxis):
            [a.set_axis_off() for a in ax]
        ax[0].imshow(imList[0])        
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.imshow(imList[0])  
        if (hideAxis):
            plt.axis('off')

def rgb_to_hsv(r, g, b):
    # Convert RGB values to HSV
    # https://www.w3resource.com/python-exercises/math/python-math-exercise-77.php
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def updateOutput(img,cols):
    # Update the UI
    plt.clf()
    plt.subplot(211)
    displayIm([img],hideAxis=True)
    plt.subplot(212)
    displayIm(cols, hideAxis=True)
    plt.pause(.05)

def GridIm(im, n_x, n_y):
    # break the image into n_x x n_y chunks (figure 2, step 1)
    x = np.array_split(im, n_y,axis=0)
    x = [np.array_split(x[i], n_x,axis=1) for i in range(len(x))]
    return [val for sublist in x for val in sublist] 

def averageImList(imList):
    # average a list of images to single pixel values and reduces dimention (figure 2, step 2 and 3)
    return [[[np.mean(np.mean(imList[i],axis=0,dtype=np.int),axis=0,dtype=np.int)]] for i in range(len(imList))]

def processIm(Im, k=3,s=0,p=5):
    size_y = Im.shape[0]/p
    dat = averageImList(GridIm(Im,math.ceil(Im.shape[1]/size_y),math.ceil(size_y)))
    deet = np.concatenate([dat[i][0] for i in range(len(dat))])
    # Ucomment this line to also filter by saturation.. highter value of s will result in more vivid colors 
    #deet = deet[[rgb_to_hsv(val[0],val[1],val[2])[1] >= s for val in deet]]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(deet)
    xx = pd.DataFrame(kmeans.cluster_centers_).astype('int32').T
    return [[[list(xx[:][i]) for i in range(k)]]]

while True:
    image = getFrame()
    cols = sorted(processIm(image,10,0,4))
    updateOutput(image,cols)

camera.release()
