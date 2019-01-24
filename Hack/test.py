import cv2
import sys
sys.path.append('build')
import bv
 
im = cv2.imread("./data/FORM - FREE TEXT INOUT BOXES_1-1.png", 0)
l = []
# imc = im.copy()
a = 5

b = bv.run(im)


for i in range(len(b)):
	cv2.imshow("Ad", b[i])
	cv2.waitKey(0)
print(len(b))

print(len(l))	
