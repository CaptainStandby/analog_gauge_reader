'''
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import typing
import sys
import math
import cv2
import numpy as np
from util_types import Circle, Point, Range, Line, Vector, GaugeOption


OutputImageFunc = typing.Callable[[cv2.Mat, str], None]


def avg_circles(circles: list[Circle]) -> Circle:
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for x, y, r in circles:
        # optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x += x
        avg_y += y
        avg_r += r
    avg_x = int(avg_x/len(circles))
    avg_y = int(avg_y/len(circles))
    avg_r = int(avg_r/len(circles))
    return (avg_x, avg_y, avg_r)


def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def distance(p1: Point, p2: Point) -> float:
    x, y = pt_delta(p1, p2)
    return np.sqrt(x**2 + y**2)


def pt_delta(p1: Point, p2: Point) -> Point:
    return (p1[0] - p2[0], p1[1] - p2[1])


def vector(origin: Point, destination: Point) -> Vector:
    return (destination[0] - origin[0], destination[1] - origin[1])


def dist_line_2_point(p1: Point, p2: Point, p3: Point) -> float:
    return np.abs(np.cross(pt_delta(p2, p1), pt_delta(p1, p3))/np.linalg.norm(pt_delta(p2, p1)))


def angle_between(v1: Vector, v2: Vector) -> float:
    return np.rad2deg(np.arctan2(np.cross(v1, v2), np.dot(v1, v2))) + 180.0


def find_circle(img: cv2.Mat) -> (Circle | None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    circles: list[list[Circle]] = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    if circles is None:
        return None

    # flatten
    circles = [item for sublist in circles for item in sublist]

    if len(circles) == 0:
        return None

    return avg_circles(circles)


def calibrate_gauge(img: cv2.Mat, gauge_name: str, output_fun: OutputImageFunc):
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''
    # height, width = img.shape[:2]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.medianBlur(gray, 5)

    # for testing, output gray image
    # cv2.imwrite('%s-bw%s' %(base_name, extension),gray)

    # detect circles
    # restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    # these are pixel values which correspond to the possible radii search range.
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    # a, b, c = circles.shape
    # x,y,r = avg_circles(circles, b)
    circle = find_circle(img)
    if circle is None:
        print('Could not find circle', file=sys.stderr)
        return

    x, y, r = circle

    img = img.copy()
    # draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3,
               cv2.LINE_AA)  # draw center of circle

    # for testing, output circles on image
    #cv2.imwrite('gauge-%s-circles.%s' % (gauge_number, file_type), img)

    # for calibration, plot lines from center going out at every 10 degrees and add marker
    # for i from 0 to 36 (every 10 deg)

    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 10.0  # in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))  # set empty arrays
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p1[i][j] = x + 0.9 * r * \
                    np.cos(separation * i * 3.14 / 180)  # point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                # point for text labels, i+9 rotates the labels by 90 degrees
                p_text[i][j] = x - text_offset_x + 1.2 * r * \
                    np.cos((separation) * (i+9) * 3.14 / 180)
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                # point for text labels, i+9 rotates the labels by 90 degrees
                p_text[i][j] = y + text_offset_y + 1.2 * r * \
                    np.sin((separation) * (i+9) * 3.14 / 180)

    # add the lines and labels to the image
    for i in range(0, interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])),
                 (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
        cv2.putText(img, '%s' % (int(i*separation)), (int(p_text[i][0]), int(
            p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    output_fun(img,f'{gauge_name}-calibration')

    return x, y, r


def get_current_value(img: cv2.Mat, angle_range: Range, value_range: Range, circle: Circle, gauge_name: str) -> (float | None):

    orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    min_angle, max_angle = angle_range
    min_value, max_value = value_range

    x, y, r = circle
    center = (x, y)

    h, w = img.shape[:2]

    img = cv2.adaptiveThreshold(
        img, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 2)
    cv2.circle(img, center, math.ceil(r * 0.1), 128, cv2.FILLED)
    # cv2.imwrite(f'{output_dir}/{gauge_name}-01-threshold.jpg', img)

    # build mask for floodfill based on gauge circle
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.circle(floodfill_mask, (x+1, y+1), math.ceil(r * 0.8), 255, cv2.FILLED)

    cv2.floodFill(img, cv2.bitwise_not(floodfill_mask),
                  seedPoint=(x, y), newVal=255)
    # cv2.imwrite(f'{output_dir}/{gauge_name}-02-floodFill.jpg', img)

    _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(f'{output_dir}/{gauge_name}-03-threshold.jpg', img)
    # edge_image = cv2.inRange(edge_image, np.ones((h,w), np.uint8) * 127, np.ones((h,w), np.uint8) * 128)
    # bla = np.ones((h,w), np.uint8) * 128
    # cv2.imwrite('bla.jpeg', bla)
    # edge_image = cv2.b(bla, edge_image)
    # edge_image = cv2.threshold(edge_image,128,)

    # found Hough Lines generally performs better without Canny / blurring, though there were a couple exceptions where it would only work with Canny / blurring
    # dst2 = cv2.medianBlur(gray2, 5)
    # dst2 = cv2.Canny(gray2, 50, 150)
    # dst2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # for testing, show image after thresholding
    # cv2.imwrite(f'{gauge_name}-dial.jpg', img)

    # find lines
    # rho is set to 3 to detect more lines, easier to get more then filter them out later
    lines = cv2.HoughLinesP(image=img, rho=3, theta=np.pi / 180, threshold=100,
                            minLineLength=math.ceil(r * 0.5), maxLineGap=math.ceil(r * 0.1))

    if lines is None:
        return None

    # print( f'lines: {lines}' )
    # flatten
    lines = [item for sublist in lines for item in sublist]

    # for testing purposes, show all found lines
    # for i in range(0, len(lines)):
    #   for x1, y1, x2, y2 in lines[i]:
    #      cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # for lines in line_results:
    # print( 'line: %s' %line )

    def augment_lines(line: Line) -> tuple[Line, float, float]:
        zero_vector = [0, -1]
        p1 = line[0:2]
        p2 = line[2:4]
        dist = dist_line_2_point(p1, p2, center)
        dist_p1 = distance(center, p1)
        dist_p2 = distance(center, p2)
        p1, p2 = (p1, p2) if dist_p2 > dist_p1 else (p2, p1)
        v = vector(p1, p2)
        angle = angle_between(zero_vector, v)

        return ((p1[0], p1[1], p2[0], p2[1]), dist, angle)

    lines = [augment_lines(line) for line in lines]
    for line, _, _ in lines:
        cv2.line(orig, line[0:2], line[2:4], (255, 0, 0), 2)

    # filter lines, that are to far from the center
    lines = [(line, dist, angle)
             for line, dist, angle in lines if dist <= math.ceil(r * 0.25)]
    # filter lines, that are out of the valid range
    lines = [(line, dist, angle) for line, dist,
             angle in lines if angle >= min_angle and angle <= max_angle]

    if not lines:
        return None

    # print( lines )
    # lines = [(x1,y1,x2,y2) for x1,y1,x2,y2 in lines if dist_line_2_point((x1,y1),(x2,y2),(x,y)) <= math.ceil(r * 0.25)]
    # for line,_,_ in lines:
    #     cv2.line(orig, line[0:2], line[2:4], (0, 255, 0), 2)

    # calculate median angle from valid lines
    median_angle = np.median([angle for _, _, angle in lines])

    angle_range = (max_angle - min_angle)
    percent = (median_angle - min_angle) / angle_range
    value_range = (max_value - min_value)
    value = percent * value_range + min_value

    return value


def capture_stream(url: str) -> (cv2.Mat | None):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print('Failed to open stream', file=sys.stderr)
        return None
    try:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read from stream', file=sys.stderr)
            return None
        return frame
    finally:
        cap.release()


def prepare_image(img: cv2.Mat, gauge: GaugeOption) -> cv2.Mat:
    match gauge.rotation:
        case 'ROTATE_90_CLOCKWISE':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        case 'ROTATE_90_COUNTERCLOCKWISE':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        case 'ROTATE_180':
            img = cv2.rotate(img, cv2.ROTATE_180)

    x, y, w, h = gauge.rect
    return img[y:y+h, x:x+w]


def process(img: cv2.Mat, gauge: GaugeOption) -> (float | None):
    img = prepare_image(img, gauge)
    # cv2.imwrite(f'{output_dir}/{gauge.name}.jpg', img)
    circle = find_circle(img)
    if circle is None:
        return None
    return get_current_value(img, gauge.angles, gauge.values, circle, gauge.name)
