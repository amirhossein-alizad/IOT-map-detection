import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#loading the picture
image = cv2.imread('8.jpg') 

#set contrast and brightness
"""
new_image = np.zeros(image.shape, image.dtype)
alpha = 1.0
beta = 0
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
"""

#make gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#getting the treshhold picture *********
ret,thresh = cv2.threshold(gray,127,255,0)

#bluring the picture for decreasing the noise
#blur = cv2.blur(gray,(5,5))

#extracting edges of the shape
#edged = cv2.Canny(gray, 30, 200) 

#extrcting the contours numpy
_,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

"""
print(contours)
for i in range (len(contours)):
  for j in range (len(contours[i])):
    for k in range (j+1, len(contours[i])):
      for l in range (k+1, len(contours[i])):
        a1 = (contours[i][j,0,1] - contours[i][k,0,1])
        a2 = (contours[i][j,0,0] - contours[i][k,0,0])
        a3 = (contours[i][k,0,1] - contours[i][l,0,1])
        a4 = (contours[i][k,0,0] - contours[i][l,0,0])
        b1 = a1*a4 - a2*a3
        b2 = a2*a4 + a1*a3
        if not (b2 == 0):
          #print(b1/b2)
          if (b1/b2)*10 in range (-1, 1):
            np.delete(contours[i], k)
            break
"""
"""
#min point and max point
	perimeter = cv2.arcLength(contours[i], False)
	#print(perimeter)
	if perimeter < 100 :
	    contours[i] = np.array([[[0, 0]]])
	if cv2.contourArea(contours[i]) < 20:
	    contours[i] = np.array([[[0, 0]]])
	
"""
#find if close fuction
def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 5 :
                return True
            elif i==row1-1 and j==row2-1:
                return False
#merging the near contours
LENGTH = len(contours)
status = np.zeros((LENGTH,1))
for i,cnt1 in tqdm(enumerate(contours)):
	x = i
	if i != LENGTH-1:
		for j,cnt2 in enumerate(contours[i+1:]):
			x = x+1
			dist = find_if_close(cnt1,cnt2)
			if dist == True:
				val = min(status[i],status[x])
				status[x] = status[i] = val
			else:
				if status[x]==status[i]:
					status[x] = i+1
unified = []
maximum = int(status.max())+1
for i in range(maximum):
    pos = np.where(status==i)[0]
    if pos.size != 0:
        cont = np.vstack(contours[i] for i in pos)
        hull = cv2.convexHull(cont)
        unified.append(hull)
#idea of going into contours one by one

print(unified)
print(len(unified))
cv2.drawContours(image, unified, -1, (0, 255, 0), 1) 
cv2.imshow('Contours', image) 
cv2.waitKey(0) 