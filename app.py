import cv2
import face_recognition as f
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp

mp_drawings=mp.solutions.drawing_utils
mp_face_mesh=mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

model_face_mesh=mp_face_mesh.FaceMesh()
st.title("OPEN CV TUTORIAL")
st.subheader("TUTORIAL")
selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("ABOUT","FACE RECOGNITION", "FACE DETECTION", "SELFIE SEGMENTATION")
)
if selectbox=="ABOUT":
    st.write("This is a part of Lets Upgrade tutorial helped me to build an app")
    st.write("This app performs various rendering on images try by uploading an image")
elif selectbox=="FACE RECOGNITION":
    image_path=st.sidebar.file_uploader("Upload an Image")
    if image_path is not None:
        image=np.array(Image.open(image_path))
        st.sidebar.image(image)

        shiva=f.load_image_file("shiva.jpg")
        shiva_encode=f.face_encodings(shiva)[0]
        shiva_locate=f.face_locations(shiva)[0]

        mahesh=f.load_image_file("C:\\Users\\Shiva\\Desktop\\LETS UPGRADE\\pichai.jpg")
        mahesh_encode=f.face_encodings(mahesh)[0]
        mahesh_locations=f.face_locations(mahesh)[0]

        known_encode=[shiva_encode,mahesh_encode]
        known_faces=["shiva","pichai"]

        image_encode=f.face_encodings(image)
        image_location=f.face_locations(image)
        face_name=[]

        for encode in image_encode:
            match=f.compare_faces(known_encode,encode)
            name="Unkown"
            face_dst=f.face_distance(known_encode,encode)
            idx=np.argmin(face_dst)
            if match[idx]:
                name=known_faces[idx]
            face_name.append(name)
        for (top,right,bottom,left),name in zip(image_location,face_name):
            cv2.rectangle(image,(left,top),(bottom,right),(114,20,245))
            font=cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(image,name,(left+10,bottom-10),font,0.75,(0,0,0),2)
        st.image(image)
elif selectbox=="FACE DETECTION":
    image_path=st.sidebar.file_uploader("Upload an Image")
    if image_path is not None:
        image=np.array(Image.open(image_path))
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            results=face_detection.process(image)
            annotated_image = image.copy()
            for detection in results.detections:
                print('Nose tip:')
                print(mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawings.draw_detection(annotated_image, detection)
            st.image(annotated_image)
elif selectbox=="SELFIE SEGMENTATION":
    button=st.sidebar.radio("Color",("None","red","green","IRON MAN BACKGROUND","AVENGERS BACKGROUND"))
    image_path=st.sidebar.file_uploader("Upload an Image")
    if button=="None":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            st.image(image)
            st.write("Choose any method to segment your image by chossing on respective button")
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")
    elif button=="red":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            r,g,b=cv2.split(image)
            zeros=np.zeros(image.shape[:2],dtype="uint8")
            st.image(cv2.merge([r,zeros,zeros]))
            st.write("Given Image's background is changed to red")
    elif button=="green":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            r,g,b=cv2.split(image)
            zeros=np.zeros(image.shape[:2],dtype="uint8")
            st.image(cv2.merge([zeros,g,zeros]))
            st.write("Given Image's background is changed to red")
    elif button=="IRON MAN BACKGROUND":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            ironman=cv2.imread("iron.jpg")
            ironman=cv2.resize(ironman,(image.shape[1],image.shape[0]))
            ironman=cv2.cvtColor(ironman,cv2.COLOR_BGR2RGB)
            image_intensity=st.sidebar.select_slider("MAIN IMAGE INTENSITY",options=[0.5,0.6,0.7,0.8,0.9,1])
            background=st.sidebar.select_slider("Background Image Intensity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            alpha=st.sidebar.select_slider("opcaity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            blended=cv2.addWeighted(image,image_intensity,ironman,background,alpha)
            st.image(blended)
    elif button=="AVENGERS BACKGROUND":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            ironman=cv2.imread("avengers.jpg")
            ironman=cv2.resize(ironman,(image.shape[1],image.shape[0]))
            ironman=cv2.cvtColor(ironman,cv2.COLOR_BGR2RGB)
            image_intensity=st.sidebar.select_slider("MAIN IMAGE INTENSITY",options=[0.5,0.6,0.7,0.8,0.9,1])
            background=st.sidebar.select_slider("Background Image Intensity",options=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            alpha=st.sidebar.select_slider("opcaity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            blended=cv2.addWeighted(image,image_intensity,ironman,background,alpha)
            st.image(blended)