import cv2
import numpy as np
from matplotlib import pyplot as plt
  
# 读取输入图片
new_image_path = '5.png'
img = cv2.imread(new_image_path)
# 将彩色图片灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用Canny边缘检测 
edges = cv2.Canny(gray,50,200,apertureSize = 3) 
# 进行Hough_line直线检测
lines = cv2.HoughLinesP(edges,1,np.pi/180, 80, 30, 10) 

# 遍历每一条直线
for i in range(len(lines)): 
	cv2.line(img,(lines[i, 0, 0],lines[i, 0, 1]), (lines[i, 0, 2],lines[i, 0, 3]), (0,255,0),2) 
# 保存结果
cv2.imwrite('result/detected_lines5.jpg', img) 
cv2.imshow("result", img)
cv2.waitKey(0)
