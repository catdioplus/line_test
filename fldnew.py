import cv2
import numpy as np

# 读取输入图片
new_image_path = '2.png'
img0 = cv2.imread(new_image_path)
assert img0 is not None, "Image not found"

# 将彩色图片转换为HSV颜色空间
hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)

# 定义HSV中红色的范围
# 红色通常在HSV颜色空间的两个区域
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# 创建红色的掩码
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 应用形态学操作来减少小的噪点
kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
mask = cv2.erode(mask, kernel, iterations=1)

# 在掩码图像上应用直线检测
edges = cv2.Canny(mask, 50, 150, apertureSize=3)

# 创建一个FLD对象
fld = cv2.ximgproc.createFastLineDetector()

# 执行检测结果
dlines = fld.detect(edges)

# 绘制检测结果
if dlines is not None:
    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)

# 显示并保存结果
output_path = 'result/FLDnew_2.jpg'
cv2.imwrite(output_path, img0)
cv2.imshow("Detected Red Lines", img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
