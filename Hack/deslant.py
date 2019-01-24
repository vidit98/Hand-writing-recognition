import cv2 
import numpy as np

class result:
	def __init__(self):
		self.sum_alpha = 0.0
		self.transform = np.zeros((2,3), dtype = np.float)
		self.height = 0
		self.width = 0

	def __lt__(self, another_result):
		return self.sum_alpha < another_result.sum_alpha


class deslantImg:

	def __init__(self, image):
		self.img = image

	def deslant(self):
		
		retval, threshold = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
		# threshold = self.img
		alphaVals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

		results = []

		for i in range(len(alphaVals)):
			res = result()
			alpha = alphaVals[i]
			shiftX = np.fmax(-alpha*threshold.shape[0], 0)
			res.height = threshold.shape[0]
			res.width = threshold.shape[1] + np.ceil(abs(alpha*threshold.shape[0]))
			res.transform[0, 0] = 1;
			res.transform[0, 1] = alpha;
			res.transform[0, 2] = shiftX;
			res.transform[1, 0] = 0;
			res.transform[1, 1] = 1;
			res.transform[1, 2] = 0;

			imgSheared = cv2.warpAffine(threshold, res.transform, (int(res.width), int(res.height)), cv2.INTER_NEAREST)

			h_alpha = np.sum(imgSheared > 0, axis = 0)
			temp = (((imgSheared>0).T)*range(1, imgSheared.shape[0]+1)).T
			maxx = np.argmax(temp, axis = 0)
			temp2 = np.max(maxx)
			temp[temp==0] = temp2
			minn = np.argmin(temp, axis = 0)
			delta_y_alpha = maxx - minn + 1
			h_alpha[h_alpha!=delta_y_alpha] = 0

			res.sum_alpha = np.sum(h_alpha*h_alpha)
			results.append(res)

		results.sort()
		imgDeslant = cv2.warpAffine(threshold, results[-1].transform, (int(results[-1].width),int(results[-1].height)), cv2.INTER_LINEAR)

		return imgDeslant

