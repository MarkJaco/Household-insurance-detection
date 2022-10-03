import streamlit as st
import frameanalyser
import videoanalyser
import yolov6.yoloInterface as yoloInterface

PRICES = {
    "bottle": 5,
    "laptop": 800,
    "mobile phone": 400,
    "oven": 800,
    "refrigerator": 500,
    "microwave": 90,
    "tv": 600
}

def main():
    # give user option of selecting analysis mode
    add_selectbox = st.sidebar.selectbox(
        "Select Mode",
        ("Frame Analysis", "Video Analysis")
    )
    st.header(add_selectbox)
    
    inferer = load_model_inferer()
    
    # give option of uploading file
    f = st.file_uploader("Upload file")
    
    # reset state variables if new files was uploaded
    if "file" not in st.session_state or f != st.session_state['file']:
        st.session_state['file'] = f
        st.session_state['key_index'] = 0
        st.session_state['current_index'] = 0
        st.session_state['considered_objects'] = []
    
    # if the file is uploaded activate other functionality
    if f is not None:
        # clear from previous run
        st.experimental_singleton.clear()
        # display video
        video_bytes = f.read()
        st.video(video_bytes)
        
        # analyse based on user input
        if add_selectbox == "Frame Analysis":
            analyser = frameanalyser.FrameAnalyser(video_bytes, PRICES, inferer)
            analyser.display()
        elif add_selectbox == "Video Analysis":
            analyser = videoanalyser.VideoAnalyser(video_bytes, PRICES, inferer)
            analyser.display()
        
@st.cache
def load_model_inferer():
    return yoloInterface.get_inferer()        

if __name__ == "__main__":
    main()


