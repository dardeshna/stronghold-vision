import cv2
import numpy as np
import scipy.optimize
import time
from networktables import NetworkTable
import operator
from math import *

cap = cv2.VideoCapture("http://10.0.8.23/live")

WIDTH = 640
HEIGHT = 480

f = 566.0699 #calculated for iPhone SE
#f = 252 #old value for iPhone SE

goal_width = 1.6875
goal_height = 0.9583

goal_points = np.float32([[-goal_width/2, -goal_height/2, 0],
                          [goal_width/2, -goal_height/2, 0],
                          [goal_width/2, goal_height/2, 0],
                          [-goal_width/2, goal_height/2, 0]])
goal_behind_points = np.float32([[-goal_width/2, -goal_height/2, .5],
                          [goal_width/2, -goal_height/2, .5],
                          [goal_width/2, goal_height/2, .5],
                          [-goal_width/2, goal_height/2, .5]])

K = np.float32([[f, 0, 0.5*(WIDTH-1)],
                        [0, f, 0.5*(HEIGHT-1)],
                        [0.0,0.0, 1.0]])

COLOR_MIN = np.array([225, 225, 225], np.uint8) #min and max hsv thresholds
COLOR_MAX = np.array([255, 255, 255], np.uint8)

def draw_HUD(img, fps, x, y, z, distance, skew, angle):
    cv2.rectangle(img, (0, 0), (380, 70), (255, 255, 255), 2)
    cv2.rectangle(img, (WIDTH-60, 0), (WIDTH-2, 40), (255, 255, 255), 2)
    text = "  D: %.1f    A: %.1fdeg    S: %.1fdeg" % (distance, angle, skew)
    text2 = "  X: %.1f    Y: %.1f    Z: %.1f" % (x, y, z)
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
            
            ret, c_rvec, c_tvec = cv2.solvePnP(goal_points, np.float32(corners), K, None)

            c_pos = np.multiply(-1, np.dot(np.linalg.inv(cv2.Rodrigues(c_rvec)[0]), c_tvec))

            c_rot = cv2.Rodrigues(np.transpose(cv2.Rodrigues(c_rvec)[0]))[0]
            
            x = c_pos[0]
            y = c_pos[1]
            z = c_pos[2]
            distance = sqrt(c_pos[0]**2+c_pos[2]**2)
            skew = degrees(atan(c_pos[0]/c_pos[2]))
            angle = degrees(c_rot[1]) - skew
            
            draw_HUD(img, fps, x, y, z, distance, skew, angle) #draw the hud on the flipped image
            
            cv2.drawContours(img, [corners], -1, (255, 150, 0), 2) #draw the contours on the flipped image
            for i in corners:
                cv2.circle(img, (i[0][0], i[0][1]), 3, (0, 255, 0)) #green circles at each corner
                
            cv2.imshow('tyr-vision', img) #create a window with the complete image

        else:
            draw_HUD(img, fps, 0, 0, 0, 0, 0, 0) #draw the hud on the flipped image
            cv2.imshow('tyr-vision', img)

        key = cv2.waitKey(30) & 0xff #press ESC to quit
        if key == 27:
            print "Total frames: %d" %total_frames
            break
        
    cv2.destroyAllWindows()
    exit()

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
        points.append(np.array([approx[min(enumerate([row[i] for row in distances]), key=operator.itemgetter(1))[0]][0]]))

    return np.array(points)


if __name__ == '__main__':
        main()
