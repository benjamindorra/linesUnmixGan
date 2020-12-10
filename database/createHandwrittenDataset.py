"""
Create a random database of overlapping lines using the iah words database.
"""
import numpy as np
from glob import glob
import cv2
import os
import random

#TODO: randomize number of words in each line (not really needed)
# add random small rotations to words (may be way too hard to be worth it)

#Paths
baseDir = os.path.dirname(__file__)
rootTrain = "trainSetHWLines"
rootValid = "validSetHWLines"
inputDir = "input"
targetDir = "target"
resultDir = "result"
imagesDir = "wordImages"

#Create the directories if needed
if not os.path.exists(os.path.join(baseDir, rootTrain)):
    os.mkdir(os.path.join(baseDir, rootTrain))
    os.mkdir(os.path.join(baseDir, rootValid))
    os.mkdir(os.path.join(baseDir, rootTrain, inputDir))
    os.mkdir(os.path.join(baseDir, rootTrain, targetDir))
    os.mkdir(os.path.join(baseDir, rootValid, inputDir))
    os.mkdir(os.path.join(baseDir, rootValid, targetDir))
    os.mkdir(os.path.join(baseDir, rootValid, resultDir))

#Open the handwritten words dataset
#http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database
imageNames = glob(os.path.join(baseDir, imagesDir, "*", "*", "*"))
random.shuffle(imageNames)

def adaptFg(bg, fg, y, x):
    """Limit the fg image if it gets out of the allocated space"""
    if y+fg.shape[0]>bg.shape[0]:
        fg=fg[:bg.shape[0]-y,:,:]
    if x+fg.shape[1]>bg.shape[1]:
        fg=fg[:,:bg.shape[1]-x,:]
    return fg, fg.shape[0], fg.shape[1]

def blendImages(bg, fg, y, x):
    """
    Cool alpha blending method inspired by
    https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/

    This is you from the future. It does not work well. Mixed text parts look
    unnatural.
    """
    fg, height, width = adaptFg(bg, fg, y, x)
    alpha = fg/255
    try:
        bg[y:y+height,x:x+width]=fg*(1-alpha)+bg[y:y+height,x:x+width]*alpha
    except:
        print("fg :", fg.shape)
        print("bg :", bg.shape)
        print("fg(1-alpha) :", (fg*(1-alpha)).shape)
        print("bg[y:y+height,x:x+width] :",(bg[y:y+height,x:x+width]).shape)
        print("y :", y)
        print("x :", x)
        print("height :", height)
        print("width :", width)
    return bg, fg, height, width

def blendImages2(bg, fg, y, x):
    """
    Just.... use the minimum ?
    """
    fg, height, width = adaptFg(bg, fg, y, x)
    try:
        bg[y:y+height,x:x+width]=np.minimum(fg,bg[y:y+height,x:x+width])
    except:
        print("fg :", fg.shape)
        print("bg :", bg.shape)
        print("fg(1-alpha) :", (fg*(1-alpha)).shape)
        print("bg[y:y+height,x:x+width] :",(bg[y:y+height,x:x+width]).shape)
        print("y :", y)
        print("x :", x)
        print("height :", height)
        print("width :", width)
    return bg, fg, height, width

#Start of the loop !
nTrain = 0
nVal = 0
i=0
countTrain=0
while i<11000:
    #Every 1 in 10 images is put in the validation set
    trainOrVal = countTrain>=9
    if trainOrVal:
        setDir = rootValid
        nVal+=1
        countTrain=0
        imgName =  f'{nVal:05}'+'.png'
    else:
        setDir = rootTrain
        nTrain+=1
        countTrain+=1
        imgName =  f'{nTrain:05}'+'.png'

    #Create the input and target
    imgShape=[256,512,3]
    inputImg = np.empty(imgShape, np.float)
    inputImg.fill(255)
    targetImg = np.empty(imgShape, np.float)
    targetImg.fill(0)

    #Open words to make the top line and the bottom line
    try:
        topLine = cv2.imread(imageNames[2*i]).astype(float)
    except:
        print(imageNames[2*i])
    botLine = cv2.imread(imageNames[2*i+1]).astype(float)

    #Randomly select the text position
    botY=imgShape[0]
    botX=imgShape[1]
    counter = 0
    while botY+botLine.shape[0]/2>=imgShape[0] or botX+botLine.shape[1]/3>=imgShape[1]:
        if counter >=3:
            #If the words can't fit correctly in 3 tries, load the next pair
            i += 1 #increment the big loop counter
            topLine = cv2.imread(imageNames[2*i]).astype(float)
            botLine = cv2.imread(imageNames[2*i+1]).astype(float)
        topY=random.randint(0, 128)
        topX=random.randint(0, 256)
        botY=topY+topLine.shape[0]-random.randint(int(topLine.shape[0]/10),int(topLine.shape[0]/2))
        botX=max(0, topX+random.randint(-64,64))
        counter += 1

    """
    #Combine the words to make an example and a target
    #Also use gaussian blur, thresholding and contrast enhancement
    #to improve the result
    inputImg, _, _, _ = blendImages(inputImg, topLine, topY, topX)
    blur = cv2.GaussianBlur(inputImg[:,:,0].astype(np.uint8),(5,5),0)
    contrast  = cv2.addWeighted(blur, 1, blur, 0, 1)
    ret,th = cv2.threshold(contrast,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.GaussianBlur(th,(5,5),0)
    inputImg = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    targetImg[:,:,2]=255-th#inputImg[:,:,0]

    blur = cv2.GaussianBlur(botLine[:,:,0].astype(np.uint8),(5,5),0)
    contrast  = cv2.addWeighted(blur, 1, blur, 0, 1)
    ret,th = cv2.threshold(contrast,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.GaussianBlur(th,(5,5),0)
    botLine = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    inputImg, botLine, height, width = blendImages(inputImg, botLine, botY, botX)
    targetImg[botY:botY+height, botX:botX+width, 1] = 255-botLine[:,:,0]

    blur = cv2.GaussianBlur(inputImg[:,:,0].astype(np.uint8),(5,5),0)
    contrast  = cv2.addWeighted(blur, 1, blur, 0, 1)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.GaussianBlur(th,(5,5),0)
    targetImg[:,:,0] = th#inputImg[:,:,0]
    """

    #Combining the words, with blending method 2

    blurVal = random.randint(-1,3)*2+1#random smoothing value

    inputImg, _, _, _ = blendImages2(inputImg, topLine, topY, topX)
    ret,th = cv2.threshold(inputImg[:,:,0].astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inputImg = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    #blur = cv2.GaussianBlur(inputImg[:,:,0].astype(np.uint8),(5,5),0)
    #contrast  = cv2.addWeighted(blur, 1, blur, 0, 1)
    if blurVal>0:
        th = cv2.GaussianBlur(th,(blurVal,blurVal),0)
    targetImg[:,:,2]=255-th#inputImg[:,:,0]

    #blur = cv2.GaussianBlur(botLine[:,:,0].astype(np.uint8),(5,5),0)
    #contrast  = cv2.addWeighted(blur, 1, blur, 0, 1)
    ret,th = cv2.threshold(botLine[:,:,0].astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    botLine = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    inputImg, botLine, height, width = blendImages2(inputImg, botLine, botY, botX)
    blank = np.empty(imgShape, np.float)
    blank.fill(255)
    blank, _, _, _ = blendImages2(blank, botLine, botY, botX)
    if blurVal>0:
        th = cv2.GaussianBlur(blank[:,:,0],(blurVal,blurVal),0)
    else:
        th = blank[:,:,0]
    #targetImg[botY:botY+height, botX:botX+width, 1] = 255-th#botLine[:,:,0]
    targetImg[:,:,1] = 255-th

    #blur = cv2.GaussianBlur(inputImg[:,:,0].astype(np.uint8),(5,5),0)
    #contrast  = cv2.addWeighted(blur, 1, blur, 0, 1)
    #ret,th = cv2.threshold(contrast,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = inputImg[:,:,0].astype(np.uint8)
    if blurVal>0:
        th = cv2.GaussianBlur(th,(blurVal,blurVal),0)
    targetImg[:,:,0] = 255-th#inputImg[:,:,0]
    inputImg = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    #Save the input and target !
    cv2.imwrite(os.path.join(baseDir, setDir, inputDir, imgName),
            inputImg.astype(np.uint8))
    cv2.imwrite(os.path.join(baseDir, setDir, targetDir, imgName),
            targetImg.astype(np.uint8))
    #cv2.imshow("test", targetImg.astype(np.uint8))
    #cv2.waitKey()
    i += 1
