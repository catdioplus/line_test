import cv2
import numpy as np

# 读取输入图片
new_image_path = '2.png'
img0 = cv2.imread(new_image_path)
assert img0 is not None, "Image not found"

# 转换为灰度图
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊降低噪声
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(gray_blurred, 50, 150, apertureSize=3)

# 膨胀操作强化粗线条
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# 直线检测
fld = cv2.ximgproc.createFastLineDetector()
dlines = fld.detect(dilated_edges)

# 绘制检测结果
if dlines is not None:
    for dline in dlines:
        x0, y0, x1, y1 = map(int, dline[0])
        # 可以在此处添加过滤条件，例如线条长度
        line_length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if line_length > 30:  # 示例阈值，需要根据实际情况调整
            cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)

# 显示并保存结果
output_path = 'fldnew2_2.png'
cv2.imwrite(output_path, img0)
cv2.imshow("Detected Lines", img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
