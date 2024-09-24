import numpy as np
import imutils
import cv2
import pandas as pd
import streamlit as st
from imutils.perspective import four_point_transform
from imutils import contours

# Load the answer key from the CSV file
df = pd.read_csv('answer.csv')
ANSWER_KEY = {row[0]: row[1] for row in df.values}

# Streamlit interface
st.title("OMR Reader")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Resize the image to a larger size if it's too small
    image = imutils.resize(image, width=800)
    
    # Process the image as in your original script
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Debug: Show edged image
    st.image(edged, caption='Edged Image', use_column_width=True)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    # Debug: Show all contours
    image_contours = image.copy()
    cv2.drawContours(image_contours, cnts, -1, (0, 255, 0), 2)
    st.image(image_contours, caption='All Contours', use_column_width=True)

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is not None:
        paper = four_point_transform(image, docCnt.reshape(4, 2))
        warped = four_point_transform(gray, docCnt.reshape(4, 2))

        thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)

        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        correct = 0

        for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
            cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
            bubbled = None

            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            color = (0, 0, 255)
            k = ANSWER_KEY[q + 1]
            if k == bubbled[1] + 1:
                color = (0, 255, 0)
                correct += 1
            cv2.drawContours(paper, [cnts[k - 1]], -1, color, 3)

        score = (correct / len(ANSWER_KEY)) * 100
        cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the original and processed images in Streamlit
        st.image(image, caption='Original Image', use_column_width=True)
        st.image(paper, caption='Processed Image', use_column_width=True)
        st.write(f"Score: {score:.2f}%")
    else:
        st.write("Could not find the document contour. Please make sure the bubble sheet is clearly visible in the image.")
        st.image(image, caption='Uploaded Image', use_column_width=True)
