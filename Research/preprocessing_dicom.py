import numpy as np
import pandas as pd
import os
import glob
import cv2
import dicom

def preprocess_patient(dicom_root, patient_id):
    slices = load_patient(dicom_root=dicom_root, patient_id=patient_id)
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    for i in range(len(slices)):
        img = slices[i].image.copy()

        # threshold HU > -300
        img[img > -300] = 255
        img[img < -300] = 0
        img = np.uint8(img)

        # find surrounding torso from the threshold and make a mask
        im2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)

        # apply mask to threshold image to remove outside. this is our new mask
        img = ~img
        img[(mask == 0)] = 0  # <-- Larger than threshold value

        # apply closing to the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

        # apply mask to image
        img2 = slices[i].image.copy()
        img2[(img == 0)] = -2000  # <-- Larger than threshold value
        image[i] = img2.copy()
    return lumTrans(image)

# DICOM rescale correction
def rescale_correction(s):
    s.image = s.pixel_array * s.RescaleSlope + s.RescaleIntercept

# Returns a list of images for that patient_id, in ascending order of Slice Location
# The pre-processed images are stored in ".image" attribute
def load_patient(dicom_root, patient_id):
    files = glob.glob(dicom_root + '/{}/*.dcm'.format(patient_id))
    slices = []
    for f in files:
        dcm = dicom.read_file(f)
        rescale_correction(dcm)
        slices.append(dcm)

    slices = sorted(slices, key=lambda x: x.SliceLocation)
    return slices

def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg
