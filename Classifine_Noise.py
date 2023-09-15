import cv2
import numpy as np
import os

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def reorder(myPoints):
 
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
 
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
 
    return myPointsNew


def warp_Perspective(img):
    heightImg, widthImg, channels = img.shape
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    threshold = cv2.threshold(imgGray, 0, 255, cv2.THRESH_OTSU)[0]
    imgThreshold = cv2.Canny(imgBlur, threshold, 255)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxArea = biggestContour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
    else:
        imgWarpColored = img
    return imgWarpColored

def apply_noise(image, noise_type):
    if noise_type == 'gauss':
        denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
        return denoised_image
    
    elif noise_type == 'poisson':
        denoised_image = cv2.fastNlMeansDenoising(image, None, h=5)
        return denoised_image
    else:
        noisy_image = image
    
    return noisy_image

def unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened


def PrePocessing(input_path, listPreprocessing):
    #Load Image
    input_path = input_path
    image = cv2.imread(input_path)
    if len(listPreprocessing) == 0 :
        display_image = image.copy()
        return display_image, image
    else: 
        if "geometric" in listPreprocessing:
            image = warp_Perspective(image)
            print('Done geometric')
            
        # copy anh image sau do return ra image do
        display_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if "contrast" in listPreprocessing:
            #Improve Contrast
            image[image > 175] = 255
            image[image < 125] = 0
            print('Done Improve Contrast')

        #Noise Reduction  
        if "gauss" in listPreprocessing:
            noisy_image = apply_noise(image, "gauss")
            image = noisy_image
            print('Done Gauss noise reduction')

        if "poisson" in listPreprocessing:
            noisy_image = apply_noise(image, "poisson")
            image = noisy_image
            print('Done Poisson noise Reduction')

            # Sharpening Image
        if "sharpen" in listPreprocessing:
            image = unsharp_masking(image, kernel_size=(9, 9), sigma=1, amount=1, threshold=0)
            print('Done sharpening')
    
        return display_image, image

