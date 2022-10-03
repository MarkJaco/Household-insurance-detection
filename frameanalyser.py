"""
Module for analysing frames of uploaded video with yolov6 for OD
"""
import streamlit as st
import cv2
import tempfile
import time
import numpy as np
import pandas as pd
import yolov6.yoloInterface as yoloInterface


class FrameAnalyser:
    def __init__(self, video_bytes, prices, inferer):
        self.video_bytes = video_bytes
        self.prices = prices
        self.inferer = inferer

    def display(self):
        """
        main function
        :param video_bytes: the video bytes from reading the uploaded video
        """
        # handle opencv
        vf, length = self.compute_cv2(self.video_bytes)
        
        # create slider
        x = st.slider("Frame Slider", 0, length)
        
        # get current frame as opencv image
        vf.set(cv2.CAP_PROP_POS_FRAMES, x - 1)
        ret, frame = vf.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # object detection
        information = self.object_detection(image)
        
        # display current frame including bounding boxes
        st.image(image)
        
        # create table for objects
        self.create_table(information)
        
    def compute_cv2(self, video_bytes):
        """
        create opencv video from uploaded file
        :param video_bytes: bytes of the uploaded file
        :return: opencv video, length in frames
        """
        # make temporary file for opencv
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_bytes)
        
        # read with opencv
        vf = cv2.VideoCapture(tfile.name)
        length = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return vf, length
        
    def object_detection(self, image):
        """
        detect objects on given image and draw bounding boxes
        :param image: cv2 image to draw on
        :return: tuple (labels, amounts, prices)
        """
        # object detection
        detections = yoloInterface.classify(image, self.inferer)
        
        # draw rectangles
        confidence_threshold = 0.5
        labels = []
        amounts = []
        price_li = []
        for d in detections:
            if d[2] <= confidence_threshold:
                continue
            # get class from index
            current_class = yoloInterface.CLASSES[int(d[0])]
            # keep track of label, amount and price of detected object
            current_index = labels.index(current_class) if current_class in labels else None
            if current_index:
                amounts[current_index] += 1
                price_li[current_class] += self.prices[current_class]
            else:
                labels.append(current_class)
                amounts.append(1)
                price_li.append(self.prices.get(current_class, 0))
            # draw rectangle on image
            p1 = (int(d[1][0]), int(d[1][1]))
            p2 = (int(d[1][2]), int(d[1][3]))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 3)
        return labels, amounts, price_li
        
    def create_table(self, information):
        """
        create table to display detected objects and their amounts
        :param information: the detected labels etc. as tuple as outputted by 
                       object_detection function
        :return: None
        """
        st.write("Current Frame Information")
        st.dataframe(
            pd.DataFrame({
                'label': information[0],
                'amount': information[1],
                'total price in €': information[2]
            })
        )