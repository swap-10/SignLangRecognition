import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import av
import asyncio

st.title("Hello")

res1 = []



class VideoTransformer(VideoProcessorBase):

    def __init__(self):

        self.res = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        model = tf.keras.models.load_model('My_Model3.h5')
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        img = frame.to_ndarray(format="bgr24")
        img1 = img
        img = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)

        prediction = probability_model.predict(np.expand_dims(img, 0))
        class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        self.res = [class_names[np.argmax(prediction[0])], 100*np.max(prediction)]
        # cv2.putText(img, self.res, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 3, 2)
        return av.VideoFrame.from_ndarray(img1, format="bgr24")


ctx = webrtc_streamer(key="Recognized", video_processor_factory=VideoTransformer, async_transform=True)


# Replace with autorefresh later

# You need to move this slider whose only purpose is to refresh the component so that the predictions displayed
# can be updated to that of the image being displayed to the camera at that time
confidence_threshold = st.slider(
   "Confidence threshold", 0.0, 100.0, 0.5, 0.05
   )

if ctx.video_processor:

    st.write(ctx.video_processor.res)
