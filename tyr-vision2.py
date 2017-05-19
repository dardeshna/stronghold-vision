import cv2
import numpy as np
import scipy.optimize
import time
from networktables import NetworkTable
import operator
from math import *

cap = cv2.VideoCapture("http://10.32.100.210/live")

WIDTH = 640
HEIGHT = 480

f = 566.0699 #calculated for iPhone SE
#f = 252 #old value for iPhone SE

goal_width = 1.6875
goal_height = 0.9583

goal_points = np.float32([[0, 0, 0],
                          [goal_width, 0, 0],
                          [goal_width, goal_height, 0],
                          [0, goal_height, 0]])

"""
#Green HSV Values
COLOR_MIN = np.array([40, 75, 130], np.uint8) #min and max hsv thresholds
COLOR_MAX = np.array([65, 254, 254], np.uint8)
"""

threshold = 0

COLOR_MIN = np.array([225, 225, 225], np.uint8) #min and max hsv thresholds
COLOR_MAX = np.array([255, 255, 255], np.uint8)

def draw_HUD(img, x, y, fps, angle, calculated):
    cv2.rectangle(img, (0, 0), (380, 70), (255, 255, 255), 2)
    cv2.rectangle(img, (WIDTH-60, 0), (WIDTH-2, 40), (255, 255, 255), 2)
    displacement_x = x-WIDTH/2
    displacement_y = -y+HEIGHT/2
    text = "<%d, %d> A: %d    D: %.1f    S: %.1fdeg" % (displacement_x, displacement_y, np.around(angle, 1), calculated[3], calculated[4])
    text2 = "           X: %.1f    Y: %.1f    Z: %.1f" % (calculated[0], calculated[1], calculated[2])
    cv2.putText(img, "%s" % text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) #x and y displacement
    cv2.putText(img, "%s" % text2, (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) #x and y displacement
    cv2.putText(img, "FPS: %s" % fps, (582, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0)) #FPS meter

def main():
    initial_time = time.time()
    total_frames = 0
    times = [time.time()]

    while True:
        
        ret, img = cap.read()

        if ret == False:
            continue
        
        total_frames += 1 #calculations for fps
        times.append(time.time())
        if len(times) > 10:
            times.pop(0)
        fps = (int)(1.0 / (times[-1] - times[-2]))
        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert image to
        threshold = cv2.inRange(img, COLOR_MIN, COLOR_MAX) #hsv threshold 
        printable_threshold = threshold.copy()
        
        areaArray = []
        contours, heirarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #find all contours in the image
        
        if len(contours) != 0: #put contours in an array
            for i, c in enumerate(contours):
                    area = cv2.contourArea(c)
                    areaArray.append(area)

            sortedData = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse = True) #sort array for biggest contour
            largestContour = sortedData[0][1]

            epsilon = 0.012 * cv2.arcLength(largestContour, True) 
            approx = cv2.approxPolyDP(largestContour, epsilon, True) #approvimate polygon from contour
            
            corners = getCorners(approx)

            x = corners[0][0]
            y = corners[0][1]

            K = np.float32([[f, 0, 0.5*(WIDTH-1)],
                        [0, f, 0.5*(HEIGHT-1)],
                        [0.0,0.0, 1.0]])
            
            ret, rvec, tvec = cv2.solvePnP(goal_points, np.float32((corners, )), K, None, flags=2)

            calculated = (tvec[0], tvec[2], tvec[1], 0, 0)
                                           
            angle = np.rad2deg(np.arctan((x - WIDTH/2) / f))

            draw_HUD(img, x, y, fps, angle, calculated) #draw the hud on the flipped image
            cv2.drawContours(img, [approx], -1, (255, 150, 0), 2) #draw the contours on the flipped image
        
            if corners != None:
                for i in corners:
                    cv2.circle(img, (i[0], i[1]), 3, (0, 255, 0)) #green circles at each corner
            cv2.imshow('tyr-vision', img) #create a window with the complete image
            
        else:
            
            calculated = (0, 0, 0, 0, 0)
            angle = 0
            
            draw_HUD(img, 320, 240, fps, angle, calculated) #draw a hud with no contour detected 
            cv2.imshow('tyr-vision', img)

        key = cv2.waitKey(30) & 0xff #press ESC to quit
        if key == 27:
            print "Total frames: %d" %total_frames
            break
        
    cv2.destroyAllWindows()

# returns (top left, top right, bottom right, bottom left)
def getCorners(approx):

    #window constants (px)
    a = WIDTH
    b = HEIGHT

    distances = []
    for i in approx:
        i = i[0]
        distances.append(list(np.sqrt([(i[0])**2+(i[1])**2, (a-i[0])**2+(i[1])**2, (a-i[0])**2+(b-i[1])**2, (i[0])**2+(b-i[1])**2])))

    points = []
    for i in range(0,4):
        points.append(tuple(approx[min(enumerate([row[i] for row in distances]), key=operator.itemgetter(1))[0]][0]))

    return tuple(points)

def normalize(x, y, z):
    x, y, z = float(x), float(y), float(z)
    d = sqrt(x**2+y**2+z**2)
    return (x/d, y/d, z/d)


def getError(a, b):
    if len(a) != len(b): return 0

    error = 0
    for i in range(0, len(a)):
        error+=(a[i]-b[i])**2

    return error

def E(x, alpha):

    l = x[0]
    d = x[1]
    h = x[2]

    vectors_p = []
    alpha_p = []

    vectors_p.append(normalize(l, h, d))
    vectors_p.append(normalize(l+goal_width, h, d))
    vectors_p.append(normalize(l+goal_width, h-goal_height, d))
    vectors_p.append(normalize(l, h-goal_height, d))

    for i in range(0, len(vectors_p)):
        for j in range(i+1, len(vectors_p)):
            alpha_p.append(degrees(acos(max(-1.0, min(1.0, np.dot(vectors_p[i], vectors_p[j]))))))

    return getError(alpha, alpha_p)    


def calculate(corners):

    alpha = []
    vectors = []
    
    for i in corners:
        u = i[0]-WIDTH/2
        v = -(i[1]-HEIGHT/2)
        vectors.append(normalize(u, v, f))
    
    for i in range(0, len(vectors)):
        for j in range(i+1, len(vectors)):
            alpha.append(degrees(acos(max(-1.0, min(1.0, np.dot(vectors[i], vectors[j]))))))

    minimize = scipy.optimize.minimize(E, (0, 8, 0), args=(alpha, ), method='SLSQP')

    best_l, best_d, hi = minimize['x']

    distance = sqrt(best_l**2+best_d**2)
    skew = degrees(atan(best_l/best_d))

    return (best_l, best_d, hi, distance, skew)


if __name__ == '__main__':
        main()
