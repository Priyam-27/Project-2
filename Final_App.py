


from keras.models import load_model
import pickle
import cv2
import numpy as np
import cvlib as cv
from PIL import Image
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

loaded_model=load_model('FruitClassifier.h5') 

# def predict_class(img):                     # this function will predict our class label
#     img=cv2.resize(img, (100,100))
#     img=img/255
#     Img=img.reshape(1,100,100,3)
#     prob= loaded_model.predict_proba(Img)
#     for i in loaded_model.predict(Img):
#         for j in i:
#             if i >= 0.5:
#                 label= 'Rotten'
#             else:
#                 label= 'Fresh'
#     return(label)
def class_prediction(img):                     # this function will predict our class label
	img=cv2.resize(img, (100,100))
	img=img/255
	Img=img.reshape(1,100,100,3)
	prob= loaded_model.predict_proba(Img)
	for i in loaded_model.predict(Img):
		for j in i:
			if i >= 0.5:
				return ('Rotten')
			else:
				return ('Fresh')


def final_model(img):
    
    """This function will classify fruit into fresh & rotten drawing boundary box and label on it"""
    
    # loaded_model=load_model('FruitClassifier.h5')      # this will load our pre-trained model
    
    def predict_class(img):                     # this function will predict our class label
        img=cv2.resize(img, (100,100))
        img=img/255
        Img=img.reshape(1,100,100,3)
        prob= loaded_model.predict_proba(Img)
        for i in loaded_model.predict(Img):
            for j in i:
                if i >= 0.5:
                    return ('Rotten')
                else:
                    return ('Fresh')
            
    ###converting openCV to PIL for cropping
    Img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    PIL_img = Image.fromarray(Img)
    
    coor, label, conf = cv.detect_common_objects(img)    # detectimg fruit
    for i in coor:
        for j in label:
            if j == 'orange' or j == 'apple' or j == 'banana':      # checking for our 3 classes apple, banana, oranges
                x,y,w,h = i
                roi = PIL_img.crop((i))                            # cropping Image to find roi (region of interest)
                roi = np.asarray(roi)
                prediction =predict_class(roi)
                # probability=loaded_model.predict_proba(roi)
                if prediction == 'Rotten':
                    img=cv2.rectangle(img, (x,y), (w,h), (255,0,0), 2)
                    img=cv2.putText(img, 'Rotten', (x+5, y+18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
                elif prediction == 'Fresh':
                    img=cv2.rectangle(img, (x,y), (w,h), (0,255,0), 2)
                    img=cv2.putText(img, 'Fresh', (x+5, y+18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
                    
    return (img)





# ### Model Deployment



import streamlit as st
import tempfile




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
					result_image= final_model(image)
					st.image(result_image, use_column_width=True)
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
						frame = final_model(frame)
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						stframe.image(frame)
                    # st.info(f'FPS : {fps}')
			else:
				st.warning('No File Uploaded!!!')  

	elif choice == 'About':
		about()





if __name__ == "__main__":
	main()







