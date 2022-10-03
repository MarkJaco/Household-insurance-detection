"""
Module to analyse entire video 
"""
import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import PIL
import imagehash
import yolov6.yoloInterface as yoloInterface


class VideoAnalyser:
    def __init__(self, video_bytes, prices, inferer):
        self.video_bytes = video_bytes
        self.prices = prices
        self.inferer = inferer
        
    def display(self):
        """
        main function to execute
        """
        # handle opencv
        vf, length = self.compute_cv2(self.video_bytes)
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        
        # find different frames
        relevant_frames = self.find_relevant_frames(vf, length, 80, fps)
        
        d = self.general_object_detection(relevant_frames, vf)
        
        # display price estimation on screen
        st.header("Choose Objects")
        self.choose_objects(d)
        
    @st.cache(hash_funcs={cv2.VideoCapture: lambda _: None})
    def find_relevant_frames(self, vf, length, threshold, skip_frames):
        """
        find all frames that differ to a certain threshold percentage
        :param vf: the opencv video
        :param length: amount of frames of opencv video
        :param threshold: the threshold percentage
        :param skip_frames: amount of frames to skip before analysing next
        :return: list of integers as the list of relevant frames
        """
        relevant_frames = []
        last_relevant_frame = None
        for x in range(0, length, skip_frames):
            # load frame image
            vf.set(cv2.CAP_PROP_POS_FRAMES, x - 1)
            ret, image = vf.read()
            
            # first relevant frame 
            if last_relevant_frame is None:
                relevant_frames.append(x)
                last_relevant_frame = image
                continue
            
            # compare frames
            res = cv2.absdiff(last_relevant_frame, image)
            res = res.astype(np.uint8)
            # percentage of difference
            percentage = (np.count_nonzero(res) * 100) / res.size
            
            if percentage > threshold:
                relevant_frames.append(x)
                last_relevant_frame = image
            
        print("this: ", relevant_frames)
        return relevant_frames
        
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
    
    @st.cache(hash_funcs={cv2.VideoCapture: lambda _: None})
    def general_object_detection(self, relevant_frames, vf):
        """
        detect objects on all relevant frames
        :param relevant_frames: list of integers of the frames
        :return: dictionary of detections
        """
        d = {}
        # detection_progress = st.progress(0)
        for i in range(len(relevant_frames)):
            # keep track of progress
            # detection_progress.progress(int((i / (len(relevant_frames) - 1)) * 100))
            # get current frame as opencv image
            vf.set(cv2.CAP_PROP_POS_FRAMES, relevant_frames[i] - 1)
            ret, frame = vf.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # object detection
            information = self.object_detection(image)
            for key in information:
                d[key] = d.get(key, [])
                d[key] += information[key]
        return d
    
    def object_detection(self, image):
        """
        detect objects on given image
        :param image: cv2 image to draw on
        :return: dict {'label': amount}
        """
        # object detection
        detections = yoloInterface.classify(image, self.inferer)
        if detections is None:
            detections = []
        acceptable = {"tv", "laptop", "chair", "mouse", "bottle", "cell phone", "vase", "mouse", "couch", "dining table"}
        
        # draw rectangles
        confidence_threshold = 0.5
        labels =  {}
        for d in detections:
            # filter out below confidence threshold
            if d[2] <= confidence_threshold:
                continue
            # get class from index
            current_class = yoloInterface.CLASSES[int(d[0])]
            if current_class not in acceptable:
                continue
            # cut detection out of image
            p1 = (int(d[1][0]), int(d[1][1]))
            p2 = (int(d[1][2]), int(d[1][3]))
            crop_img = image[p1[1]:p2[1], p1[0]:p2[0]]
            # resize image to standard
            height = 200
            width = int((height / crop_img.shape[0]) * crop_img.shape[1])
            crop_img = cv2.resize(crop_img, (width, height))
            # keep track of label and add corresponding image
            labels[current_class] = labels.get(current_class, [])
            labels[current_class].append(crop_img)
        return labels
        
    def display_price_evaluation(self):
        """
        display the calculated amount of objects and prices
        give option of modifying
        :return: None
        """
        st.header("Price Evaluation")
        overall_price = 0
        for l in st.session_state['considered_objects']:
            p = self.prices.get(l, 0)
            overall_price += p
            st.write(f"Adding {p}€ for {l}")
        st.markdown(f"### Overall Price: {overall_price}€")

    def choose_objects(self, objects):
        """
        give user the option of choosing the correct objects
        :param objects: the dictionary of detected objects with cv2 images
        :return: 
        """
        # getting keys as list from dict
        keys = list(objects.keys())
        # stop if no more images
        if st.session_state['key_index'] >= len(keys):
            st.write("All objects chosen for")
            self.display_price_evaluation()
            return
        # display first image
        img_list = objects[keys[st.session_state['key_index']]]
        current_img = img_list[st.session_state['current_index']]
        obj_class = keys[st.session_state['key_index']]
        st.write(f"Choose for this {obj_class}")
        st_img = st.image(current_img)
        # display next image if button is pressed
        col1, col2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])
        with col1:
            if st.button("Consider Object"):
                self.next_image(img_list, keys)
                st.session_state['considered_objects'].append(obj_class)
        with col2:
            if st.button("Discard Object"):
                self.next_image(img_list, keys)
            
    def next_image(self, img_list, keys):
        """
        cycle to next choice image
        """
        st.session_state['current_index'] += 1
        # switch object type
        if st.session_state['current_index'] == len(img_list):
            st.session_state['current_index'] = 0
            st.session_state['key_index'] += 1
        # compare images
        else:
            # from cv2 to pillow
            current_pil = PIL.Image.fromarray(img_list[st.session_state['current_index']])
            previous_pil = PIL.Image.fromarray(img_list[st.session_state['current_index'] - 1])
            # check for similarity
            hash0 = imagehash.average_hash(current_pil)
            hash1 = imagehash.average_hash(previous_pil)
            cutoff = 7
            if hash0 - hash1 < cutoff:
                self.next_image(img_list, keys)
        
                