import cv2
import numpy as np
video= cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

name=input("enter the name:")

skip=0
face_training=[]
path='C:/Users/Shreya/Desktop/machine learning/'

while True:
	boolean ,frame = video.read()

	if boolean == False:
		continue

	
	face = classifier.detectMultiScale(frame, 1.3, 5)
	if len(face)==0:
		continue

	#arrange in decsending order of faces incase of multiple faces
	face = sorted(face, key = lambda l:l[2]*l[3], reverse = True)
	for (x,y,b,h) in face:
		cv2.rectangle(frame, (x,y), (x+b,y+h), (0,0,225),3)

		#cropping face part
		offset=10
		cropped_face = frame[y-offset:y+h+offset,x-offset:x+b+offset]

		#resize image spthat all training data is of same size
		cropped_face = cv2.resize(cropped_face,(100,100))

		
		#storing every 10th face
		skip+=1
		if (skip%10==0):
			face_training.append(cropped_face)
			print(len(face_training))

	cv2.imshow('video stream', frame)
	cv2.imshow("face", cropped_face)


	#if user press e then exit
	press = cv2.waitKey(1) & 0xff
	if press == ord('e'):
		break

# Converting our face_training list array into a numpy array
face_training = np.asarray(face_training)
face_training = face_training.reshape((face_training.shape[0],-1))
print(face_training.shape)

# Saving training data into file system
np.save(path+name+'.npy',face_training)
print("Data Successfully saved at "+path+name+'.npy')

video.release()
cv2.destroyAllWindows()