import streamlit as st
import pytesseract
import io
from keras.utils import load_img
import model

#pytesseract.pytesseract.tesseract_cmd = None

import cv2
#from pytesseract import Output
from PIL import Image
import pytesseract
#import sys
import numpy as np
from cv2 import *
#import time
import os
#import threading
#import imutils
import re

def get_string(img_path):
    # Read image using opencv
    img = cv2.imread(img_path)
   # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]
    # Create a directory for outputs
    output_path = os.path.join('output_path', "ocr")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)

    #Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)#increases the white region in the image
    img = cv2.erode(img, kernel, iterations=1) #erodes away the boundaries of foreground object

    #Apply blur to smooth out the edges
    #img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #Save the filtered image in the output directory
    save_path = os.path.join(output_path, file_name + "_filter_" + str('as') + ".png")
    cv2.imwrite(save_path, img)
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="ara")
    return result


st.title('Transliterasi Teks Arab ke Latin')
# Header
st.header('Membaca Gambar berisi Teks Arab')

# Subheader
st.subheader('Upload Gambar')

# Input type - File
uploaded_file = st.file_uploader('File Uploader', type=["png", "jpg", "jpeg"])

# Prediction for upload file
if uploaded_file is not None:
    image = load_img(io.BytesIO(uploaded_file.read()))
    # Showing image that was upload
    st.image(image, caption="Uploaded Image", use_column_width=True)
    #img = Image.open(image)
    #img = img.save("img.jpg")

    text = pytesseract.image_to_string(image, lang='ara')
    #a = get_string("img.jpg")
    y = re.sub(r"[\n\t\s]+", " ", text)

    # Print
    #st.text_area("Teks Prediksi:", text)
    #texts = st.text_input("Teks Prediksi:", text)
    # Subheader
    st.subheader('Teks Arab Prediksi')
    texts = st.text_area("Teks Prediksi:", y)

    #modeltransliterasi.decoder_output = modeltransliterasi.generate(text)
    #prediction = modeltransliterasi.decode(modeltransliterasi.output_decoding, modeltransliterasi.decoder_output[0])
#    prediction = model2.predict_output(text)
    # Print
#    st.text_area("Hasil Prediksi  Transliterasi:", prediction)

#text = st.text_input
    with st.form(key='my_form'):
        #texts = st.text_input(text, label='Tulisan Arab bisa diedit')
        submit_button = st.form_submit_button(label='Prediksi')
        prediction = model.predict_output(texts)
    st.subheader('Teks Prediksi Transliterasi')
    st.text_area("Hasil Prediksi  Transliterasi:", prediction)
