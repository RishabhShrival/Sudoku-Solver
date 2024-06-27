import cv2
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer


def imagePreProcess(imagepath,pixel_size=64*9):
    #image must be in greyscale
    #resize image with reduced pixel
    image = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    image=cv2.resize(image,[pixel_size,pixel_size])
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)
    image=cv2.GaussianBlur(image,(3,3),1)
    return image



def findContourPoints(image):
    image = cv2.bitwise_not(image)
    contours, hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max=0
    for i in contours:
        if(cv2.contourArea(i)>50):
            peri=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            if(len(approx)==4):
                if(cv2.contourArea(approx)>max):
                    bigcontour=approx
                    
    bigcontour=bigcontour.T
    bigcontour[0]=bigcontour[0][0]
    bigcontour[1]=bigcontour[1][0]
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    return np.array(list(zip(list(bigcontour[0][0]),list(bigcontour[1][0]))))




def imageWrape(image,contour):
    contour = np.array(contour, dtype=np.float32)
    dest_points = np.array([(0, 0), (0, len(image)),(len(image), len(image)),(len(image),0)],dtype=np.float32)
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(contour,dest_points)
    sudoku_img_bit= cv2.warpPerspective(image, matrix,(len(image),len(image)))

    #cliping
    clipped_image = sudoku_img_bit[4:-4, 4:-4]
    return clipped_image



def imageSplit(image, cliping_margin=8):
    image = cv2.resize(image, (64*9,64*9)) # +4 for clipping margin
    # Split the image into 9 vertical strips
    sudoku_img_vsplit = np.vsplit(image, 9)
    sudoku_img_vhsplit = []   
    for i in range(9):
        # Split each vertical strip into 9 horizontal strips
        row_splits = np.hsplit(sudoku_img_vsplit[i], 9)
        clipped_row_splits = np.array([img[cliping_margin:-cliping_margin, cliping_margin:-cliping_margin] for img in row_splits])  
        clipped_row_splits = np.array([cv2.copyMakeBorder(img,cliping_margin,cliping_margin,cliping_margin,cliping_margin,cv2.BORDER_CONSTANT,None,255) for img in clipped_row_splits])
        # clipped_row_splits = np.array([cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1) for image in clipped_row_splits])
        clipped_row_splits = np.array([cv2.GaussianBlur(img,(3,3),1) for img in clipped_row_splits])  
        # clipped_row_splits = np.array([cv2.GaussianBlur(img,(3,3),1) for img in clipped_row_splits])  
        sudoku_img_vhsplit.append(clipped_row_splits)

    sudoku_img_vhsplit=np.array(sudoku_img_vhsplit)
    return sudoku_img_vhsplit


def loadmodel(modelPath):
    # modelPath='./model/best_mnist_resnet50_model'
    # Load the model as an inference-only layer
    tfsm_layer = TFSMLayer(modelPath, call_endpoint='serving_default')
    return tfsm_layer


def prediction(sudoku_img_vhsplit,loaded_modal):
    thresh=10*255
    output=np.zeros(shape=(9,9))
    output2=np.zeros(shape=(9,9))
    for row in range(9):
        for column in range(9):
            tempImage=cv2.resize(sudoku_img_vhsplit[row][column],(32,32))
            tempImage=cv2.cvtColor(tempImage, cv2.COLOR_GRAY2RGB)
            tempImage=cv2.bitwise_not(tempImage)
            if(np.sum(tempImage[5:-5,5:-5])>100):
                tempImage2 = np.expand_dims(tempImage, axis=0)
                predict=loaded_modal(tempImage2)
                result=loaded_modal(tempImage2)
                predict=tf.argmax(result['dense_1'],axis=1)
                output2[row][column]=tf.reduce_max(result['dense_1'])
                if(tf.reduce_max(result['dense_1'])>0.4):
                    output[row][column]=int(predict)
                else:
                    output[row][column]=0
            else:
                output[row][column]=0
            # print(f"{row} |  {column} | sum is {np.sum(tempImage[5:-5,5:-5])} |  predicted value is {output[row][column]}")
    return output


#compilation of all preprocess and prediction
def compiled(imagePath,modelPath='./best_mnist_resnet50_model'):
    image=imagePreProcess(imagePath)
    contourPoints=findContourPoints(image)
    image=imageWrape(image,contourPoints)
    imageList=imageSplit(image)
    model=loadmodel(modelPath)
    output=prediction(imageList,model)
    return output

