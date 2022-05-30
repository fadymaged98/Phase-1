import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def display_lines(image, lines):                         # makes a black image with the lanes marked
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0, 255, 0), 10)
    return line_image        


def region_of_interest(image):                           # makes all the image black except the region we need to process
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


def Canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)       # applying grayscale filter
    blur = cv2.GaussianBlur(gray, (5,5), 0)              # applying gaussian filter to reduce noise
    canny = cv2.Canny(blur,50,150)                       # applying canny filter to detect gradients
    return canny


def main():
    image = cv2.imread("test_image.jpg")                 # importing test image
    image_copy = np.copy(image)                          # making a copy of the image to process it
    canny = Canny(image_copy)
    cropped = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    averaged_lines = average_slope_intercept(image_copy, lines)
    line_image = display_lines(image_copy, averaged_lines)
    combo_image = cv2.addWeighted(image_copy, 0.8, line_image, 1, 1)
    cv2.imshow("cropped", combo_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()