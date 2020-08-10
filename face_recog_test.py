import cv2
import numpy as np 
import os 

#KNN
def distance(v1, v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]

#video streaming
video= cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


skip=0
face_train=[]
labels=[]
path='C:/Users/Shreya/Desktop/machine learning/'

class_id = 0 #labels
names = {} #maping of names with labels



# loading training data
for t in os.listdir(path):
	if t.endswith('.npy'):
		#Create a mapping btw class_id and name
		names[class_id] = t[:-4]
		print("Loaded "+t)
		data_item = np.load(path+t) #giving file name plus path to load
		face_train.append(data_item)

		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_train,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


#testing

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

		pred = knn(trainset, cropped_face.flatten())

		pred_name = names[int(pred)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+b,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('e'):
		break

video.release()
cv2.destroyAllWindows()