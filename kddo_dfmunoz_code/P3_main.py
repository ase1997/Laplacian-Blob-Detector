"""
Created on Wed Nov 17 20:37:39 2021
Author: Khoa Dang Do, Derik Munoz Solis
Instructor: Dr. Tianfu Wu
ECE 558-01 (Fall 2021)
Project 3: Laplacian Blob Detector
File Description: Main Code
**License** © 2021 - 2021 Khoa Do, Deirk Munoz. All rights reserved.
"""










""" import necessary packages """
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt










""" blobs plotting function """
def plot_blob(img, exLoc, k):
    
    
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap="gray")
    
    for blob in exLoc:
        y,x,r = blob                                        # (x,y,radius)
        c = plt.Circle((x, y), r*k, color='red', linewidth=0.5, fill=False)
        ax.add_patch(c)
        
    ax.plot()  
    plt.show()










""" non-max supression function """
def NMS(scaleSpace):

    
    scaleLayers= scaleSpace.shape[0]
    iRows, iCols = scaleSpace[0].shape
    
    
    """ 2-D NMS """
    nms2D = np.zeros((scaleLayers, iRows, iCols), dtype='float32')
    
    for i in range(scaleLayers):
        octaveImg = scaleSpace[i, :, :]
        [octR,octC] = octaveImg.shape
        for j in range(octR):
            for k in range(octC):
                #nms2D[i, j-1, k-1] = np.amax(octaveImg[j-1:j+2 , k-1:k+2])
                nms2D[i, j, k] = np.amax(octaveImg[j:j+2 , k:k+2])
    
    
    """ 3-D NMS """
    nms3D = np.zeros((scaleLayers, iRows, iCols), dtype='float32')
    
    for j in range(1, np.size(nms2D,1)-1):
            for k in range(1, np.size(nms2D,2)-1):
                nms3D[:, j, k] = np.amax(nms2D[:, j-1:j+2 , k-1:k+2])
                
    nms3D = np.multiply((nms3D == nms2D), nms3D)
    
    return nms3D                                            # return new scale space after NMS to log_scale_space










""" blobs detecting function """
def blob_detector(logImg, k, sigma):
    
    
    exLocation = []                                         # list of extremas' coordinates
    
    w,l = logImg[0].shape
    
    for i in range(w):
        for j in range(l):
            slicedImg = logImg[:, i:i+2 , j:j+2]            # 10x3x3 slice (can do with different 10xMxN slice)
            extrema = np.max(slicedImg)                     # find maximum
            if extrema >= 0.025:                            # threshold manually changed
                z,x,y = np.unravel_index(slicedImg.argmax(), slicedImg.shape)           # find locations of max values, convert linear indexes back to row-col indexes of the sliced img
                exLocation.append((i+x, j+y, np.power(k,z)*sigma))                      # convert back to locations on the original image corresponding to that sigma
    
    return exLocation                                       # return extremas' location to main function










""" 2-D FFT function """
def DFT2(fw):

    
    fw_2D = np.zeros(fw.shape, dtype=complex)               # create image same size as the original image, filled with 0's

    """ 2-D FFT algorithm """
    for i in range(fw.shape[0]):
        fw_2D[i, :] = np.fft.fft(fw[i, :])                  # do 1-D FFT on rows of original image
    for i in range(fw.shape[1]):                            # do 1-D FFT on columns of image after being 1-D FFT-ed 
        fw_2D[:, i] = np.fft.fft(fw_2D[:, i])

    return fw_2D                                            # return transformed image to conv2_fft










""" kernel padding function """
def pad_kernel(w, imgR, imgC):
    
    
    kR, kC = w.shape
    
    paddedKernel = np.zeros((imgR,imgC)).astype(np.float32)
    paddedKernel[int((imgR/2)-(kR/2)):int((imgR/2)+(kR/2)) , int((imgC/2)-(kC/2)):int((imgC/2)+(kC/2))] = w
    paddedKernel = np.fft.ifftshift(paddedKernel)           # shift spectrum to center
    
    return paddedKernel                                     # return padded kernel to conv2_fft










""" zero padding function for img"""
def zero_pad_img(image,padW):
    
    
    """ zero-padding algorithm """
    paddedImg = np.zeros((image.shape[0] + (padW * 2), image.shape[1] + (padW * 2)))    # create (pad_w)-zero arrays around the image
    paddedImg[int(padW):-int(padW) , int(padW):-int(padW)] = image                      # assume padding width is even all around
    
    return paddedImg                                        # return padded image to conv2_fft










""" 2-D FFT convolution function """
def conv2_fft(f, w):
    
    
    padW = 1                                                # assume padding width and stride = 1

    """ convolution algorithm for grayscale """
    padded_f = zero_pad_img(f,padW)                         # call zero_pad to do pad image with zero padding
    imgR, imgC = padded_f.shape
    padded_w = pad_kernel(w, imgR, imgC)                    # pad the kernel
    
    padded_f_fft = DFT2(padded_f)                           # transform padded image to freg. domain
    padded_w_fft = DFT2(padded_w)                           # transform padded kernel to freg. domain
    
    mul_fft = np.multiply(padded_f_fft, padded_w_fft)       # convolution in freg. domain

    output = np.abs(np.conj(DFT2(np.conj(mul_fft))) / (imgR*imgC))                      # invert freg. domain to spatial domain and calculate the magnitude
                                                                                        # imaginary parts will be automatically discarded if np.abs is not used
    
    """ strip off extra padding to make the convoled img's size equal the og. img's size """
    newOutput = np.delete(output, np.s_[:1], axis=0)
    newOutput = np.delete(newOutput, np.s_[-1:], axis=0)
    newOutput = np.delete(newOutput, np.s_[:1], axis=1)
    newOutput = np.delete(newOutput, np.s_[-1:], axis=1)
        
    return newOutput                                        # return convolution result to log_scale_space

    








""" LoG filter generation function """
def log_kernel(scaleSigma):
    

    n = np.ceil(scaleSigma*6)
    
    x,y = np.ogrid[(-n//2):(n//2+1) , (-n//2):(n//2+1)]     # generate a grid 
    xW = np.exp(-(x*x/(2.*scaleSigma**2)))
    yW = np.exp(-(y*y/(2.*scaleSigma**2)))
    w = (-(2*scaleSigma**2) + (x*x + y*y) ) * (xW*yW) * (1/(2*np.pi*scaleSigma**4))     # simply with get to standard LoG filter formula
    
    return w                                                # return Gaussian kernel to log_scale_space










""" Laplacian scale space function """
def log_scale_space(img, k, sigma, numLayers):
    
    
    logImg = []                                             # store list of LoG images
    
    for i in range(numLayers):
        scaleSigma = sigma * np.power(k, i)
        logFilter = log_kernel(scaleSigma)                  # call log_kernal function to generate LoG filters
        filteredImg = conv2_fft(img, logFilter)             # convolution in freg. domain
        squareImg = np.square(filteredImg)                  # square fitlered image
        logImg.append(squareImg)
        
    scaleSpace = np.array([i for i in logImg])
    newSpace = NMS(scaleSpace)                              # call NMS to perform non-max supression

    return newSpace                                         # return new LoG space scale after performing NMS to main function
    
    








""" main function """
def main_function():
    
    
    """ some constants """
    k = 1.5
    sigma = 2
    numLayers = 7                                           # no. LoG filters, manually changed
    
    img = cv2.imread("computervision.jpg", 0) / 255               # read image as grayscale and scale to between 0 and 1
    
    
    clockS = time.time()                                    # start timer
    
    logImg = log_scale_space(img, k, sigma, numLayers)      # call log_scale_space function to convolve image with different LoG filters, perform NMS, and return list of DoG img's
    exLoc =  list(set(blob_detector(logImg, k, sigma)))     # call blob_detector to find extrema locations, return locations, and store them in a list of set of coordinates 
    plot_blob(img, exLoc, k)                                # plot detected blobs on grayscale image

    clockE = time.time()                                    # end timer
    
    print("Runtime is:", clockE - clockS)                   # calculate runtime = performance
    
      







    
if __name__ == "__main__":
    main_function()