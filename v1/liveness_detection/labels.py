import pandas as pd 
import cv2

images = []
labels = []
for i in range(400):
	img = cv2.imread("img-live/" + str(i)+".jpg")
	img = cv2.resize(img, (120,120))
	images.append(img)
	labels.append(0)

for i in range(400):
	img = cv2.imread("img-not-live/" + str(i)+".jpg")
	img = cv2.resize(img, (120,120))
	images.append(img)
	labels.append(1)



d = {'images': images, 'labels': labels}
csv = pd.DataFrame(data=d)

csv.to_csv('liveness_data.csv', index=False)
