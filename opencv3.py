import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('albion_cabbage.jpg', cv.IMREAD_UNCHANGED)

needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]


resut = cv.matchTemplate(haystack_img, needle_img, cv.TM_SQDIFF_NORMED)


threshold = 0.17
locations = np.where(resut <= threshold)
locations = list(zip(*locations[::-1]))
print(locations)

rectangles = []

for loc in locations:
    rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
    rectangles.append(rect)
    rectangles.append(rect)

   
rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
print(rectangles) 

if len(rectangles):
    print("found Needle")
    marker_color = (255, 0, 255)
    marker_type = cv.MARKER_CROSS

    for (x, y, w, k) in rectangles:
        '''
        top_left = (x, y)
        bottom_right = (x + w, y + k)

        cv.rectangle(haystack_img, top_left, bottom_right, line_color, line_type)
        '''
        center_x = x + int(w/2)
        center_y = y + int(k/2)
        cv.drawMarker(haystack_img, (center_x, center_y), marker_color, marker_type)

    cv.imshow("Matchers", haystack_img)
    cv.waitKey()   