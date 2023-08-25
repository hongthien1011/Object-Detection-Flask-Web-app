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









def is_noisy(image):
    # Gaussian noise check
    diff_gauss = cv2.absdiff(image, cv2.GaussianBlur(image, (5, 5), 0))
    mean_diff_gauss = np.mean(diff_gauss)
    is_noisy_gauss = mean_diff_gauss > 10  # Adjust this threshold as needed
    
    # Salt & Pepper noise check
    num_black_pixels = np.sum(image == 0)
    num_white_pixels = np.sum(image == 255)
    total_pixels = image.shape[0] * image.shape[1]
    ratio_black = num_black_pixels / total_pixels
    ratio_white = num_white_pixels / total_pixels
    is_noisy_sp = ratio_black > 0.02 or ratio_white > 0.02
    
    # Poisson noise check
    poisson_var = np.var(image)
    is_noisy_poisson = poisson_var > 5  # Adjust this threshold as needed
    
    # Speckle noise check
    diff_speckle = cv2.absdiff(image, cv2.GaussianBlur(image, (5, 5), 0))
    mean_diff_speckle = np.mean(diff_speckle)
    is_noisy_speckle = mean_diff_speckle > 10  # Adjust this threshold as needed
    
    return {
        'gauss': is_noisy_gauss,
        's&p': is_noisy_sp,
        'poisson': is_noisy_poisson,
        'speckle': is_noisy_speckle
    }

def apply_noise(image, noise_type):
    if noise_type == 'gauss':
        denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
        return denoised_image
    elif noise_type == 's&p':
        # Áp dụng Median blur để loại bỏ nhiễu muối tiêu
        denoised_image = cv2.medianBlur(image, 5)
        return image
    elif noise_type == 'speckle':
        # Áp dụng bộ lọc trung bình
        denoised_image = cv2.medianBlur(image, 5)  # Thay số 5 bằng kích thước kernel bạn muốn sử dụng
        return denoised_image
    elif noise_type == 'poisson':
        denoised_image = cv2.fastNlMeansDenoising(image, None, h=10)
        return denoised_image
    else:
        noisy_image = image
    
    return noisy_image

def unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened


def PrePocessing(input_path):
    #Load Image
    input_path = input_path
    image = cv2.imread(input_path)
    image = warp_Perspective(image)
   # copy anh image sau do return ra image do
    org_image = image.copy()


    #Improve Contrast
    white_mask = image >= 200
    dark_mask = image <= 100
    image = np.where(white_mask, 255, np.where(dark_mask, 0, image))
    print('Done Improve Contrast')

    #Check Noise
    noise_dict = is_noisy(image)
    print(noise_dict)

    #Noise Reduction
    for noise_type, apply in noise_dict.items():
        if apply:
            noisy_image = apply_noise(image, noise_type)
            image = noisy_image
    print('Done noise Reduction')

    #Resize Image
    H, W, c = image.shape

    if max(H, W) > 1280:
        if H > W:
            image = cv2.resize(image, (int(W * 1280 / H), 1280))
        else:
            image = cv2.resize(image, (1280, int(H * 1280 / W)))
        # Save the resized image
    print('Done Resize image')

    # Sharpening Image
    image = unsharp_masking(image, kernel_size=(3, 3), sigma=0.5, amount=0.5, threshold=0)
    return org_image,image

