
from io import BytesIO
import base64
import cv2
import numpy as np
from keras.models import load_model
import cvlib as cv
import matplotlib.pyplot as plt
import os
import requests
from PIL import Image
import time
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
# get_ipython().run_line_magic('matplotlib', 'inline')
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  


id = '1yQPsSzVLYKPh3NA3v2eeFnhieVaaBYIb'
destination = 'weight.h5'


download_file_from_google_drive(id, destination)
# time.sleep(10)   # 20 sec sleep for letting whole model to load




def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href




def predict_class(img):
	loaded_model = load_model(destination)
	img=cv2.resize(img, (100,100))
	img=img/255
	Img=img.reshape(1,100,100,3)
	for i in loaded_model.predict(Img):
		for j in i:
			if i >= 0.5:
				return 'Rotten'
			else:
				return 'Fresh'


def classify_fruit(img):  # input as openCV image
    # converting openCV to PIL for cropping

    Img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    PIL_img = Image.fromarray(Img)
    
    coor, label, conf = cv.detect_common_objects(img)    # detectimg fruit
    for i in coor:
        for j in label:
            if j == 'orange' or j == 'apple' or j == 'banana':      # checking for our 3 classes apple, banana, oranges
                x,y,w,h = i
                roi = PIL_img.crop((i))                            # cropping Image to find roi (region of interest)
                roi = np.asarray(roi)
                if predict_class(roi) == 'Rotten':
                    cv2.rectangle(img, (x,y), (w,h), (255,0,0), 2)
                    cv2.putText(img, 'Rotten', (x+5, y+18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)
                elif predict_class(roi) == 'Fresh':
                    cv2.rectangle(img, (x,y), (w,h), (0,255,0), 2)
                    cv2.putText(img, 'Fresh', (x+5, y+18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
                    
    return img










def about():
	st.write('''
    This web app is for classification of fresh and rotten fruits.
    This app is for 3 fruit categories, i.e. Apples, Oranges and Banana.
    
    I used two basic algorithm for this project one is object detection using YOLO object detection,
    other is Convolutional Neural Network classifier for Image classification.''')





def main():
	st.title('Fresh and Rotten Fruit Classifier WebApp :apple::banana:🍊')
	st.write('Using YOLO object detection and CNN Classification')
    
	activities =['Home', 'About']
	choice = st.sidebar.selectbox('Pick your choice', activities)
    
	if choice == 'Home':
		st.write('Go to the About section on the sidebar to learn more about it.')
        
		file_type =['Image', 'Video']
		file_choice = st.sidebar.radio('Choose file type', file_type)

		if file_choice == 'Image':

			image_file=st.file_uploader('Upload Image', type=['jpeg', 'png', 'jpg', 'webp'])
            
			if image_file is not None:
				image = Image.open(image_file)
				image = np.array(image)
                
				if st.button('Process'):
					result_image= classify_fruit(image)
					st.image(result_image, use_column_width=True)
					pil_img = Image.fromarray(result_image)
					st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)


		elif file_choice == 'Video':
			video_file = st.file_uploader('Upload Video', type=['mp4'])
			if video_file is not None:
				tfile = tempfile.NamedTemporaryFile(delete=False)
				tfile.write(video_file.read())

				if st.button('Process'):
					vf = cv2.VideoCapture(tfile.name)
					fps=vf.set(cv2.CAP_PROP_FPS,30)
					stframe = st.empty()

					while vf.isOpened():
						ret, frame = vf.read()
						if not ret:
							print("Can't receive frame (stream end?). Exiting ..." )
							break
						frame = classify_fruit(frame)
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						stframe.image(frame)
						# st.markdown(get_binary_file_downloader_html('C:\\Users\\Priyam Srivastava\\Downloads\\', 'Video'), unsafe_allow_html=True)
			else:
				st.warning('No File Uploaded!!!')  

	elif choice == 'About':
		about()





if __name__ == "__main__":
	main()







