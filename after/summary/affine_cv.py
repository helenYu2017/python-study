import cv2 
import numpy as np 
img = cv2.imread("img/jin.jpg") 
cols, rows, c = np.shape(img)
theta = np.pi / 4
Ab = np.float32([[np.cos(theta), np.sin(theta), 0], 
                 [-np.sin(theta), np.cos(theta), 0]])
img2 = cv2.warpAffine(img,Ab,(cols,rows))
cv2.imshow(' ', img2)
cv2.waitKey(0)


np.savez("img.image", U=U, V=V, A=A)