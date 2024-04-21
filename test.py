import cv2
import numpy as np
from matplotlib import pyplot as plt

new_image_path = '2.png'
new_image = cv2.imread(new_image_path)


gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 5)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=20)
line_image = np.copy(new_image)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

plt.imshow(line_image_rgb)
plt.title('Detected Lines')
plt.show()

output_path = 'result/Hough2.png'
cv2.imwrite(output_path, line_image)

output_path

