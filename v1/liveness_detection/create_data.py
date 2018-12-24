import cv2
import numpy as np 


cap = cv2.VideoCapture(0)
xx = 0
while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if xx < 100:
		cv2.imwrite('img-not-live/' +str(900+ xx) + ".jpg", img )
	else:
		break
	cv2.imshow('img',img)
	k = cv2.waitKey(30) % 0xff
	xx+=1
	if k == 27:
		break


