import cv2
import numpy as np
import scipy.optimize
import time
import operator
from math import *
import random

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

global counter

output = open('output.csv', 'w')

"""
#Green HSV Values
COLOR_MIN = np.array([40, 75, 130], np.uint8) #min and max hsv thresholds
COLOR_MAX = np.array([65, 254, 254], np.uint8)
"""

COLOR_MIN = np.array([225, 225, 225], np.uint8) #min and max hsv thresholds
COLOR_MAX = np.array([255, 255, 255], np.uint8)

def draw_HUD(img, fps, x, y, z, distance, skew, angle):
    cv2.rectangle(img, (0, 0), (380, 70), (255, 255, 255), 2)
    cv2.rectangle(img, (WIDTH-60, 0), (WIDTH-2, 40), (255, 255, 255), 2)
    text = "  D: %.1f    A: %.1fdeg    S: %.1fdeg" % (distance, skew, angle)
    text2 = "  X: %.1f    Y: %.1f    Z: %.1f" % (x, y, z)
    cv2.putText(img, "%s" % text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) #x and y displacement
    cv2.putText(img, "%s" % text2, (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) #x and y displacement
    cv2.putText(img, "FPS: %s" % fps, (582, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0)) #FPS meter

def main():
    initial_time = time.time()
    total_frames = 0
    times = [time.time()]

    counter = 0

    while True:
        
        img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
        img[:,:] = (32,32,32)
        
        total_frames += 1 #calculations for fps
        times.append(time.time())
        if len(times) > 10:
            times.pop(0)
        fps = (int)(1.0 / (times[-1] - times[-2]))

        pos = np.float32([-10, 0, -10])
        rot = np.array([0, pi/4, 0])
        
        a_skew = degrees(atan(pos[0]/pos[2]))
        
        rvec = np.transpose(cv2.Rodrigues(rot)[0])
        tvec = np.float32(np.multiply(-1, np.dot(rvec, pos)))
        
        goal = cv2.projectPoints(goal_points, rvec, tvec, K, None)
        goal_behind = cv2.projectPoints(goal_behind_points, rvec, tvec, K, None)

        for i in goal_behind[0]:
            cv2.circle(img, (i[0][0], i[0][1]), 3, (0, 0, 255)) #red circles at each corner
            
        for i in goal[0]:
            cv2.circle(img, (i[0][0], i[0][1]), 3, (0, 255, 0)) #green circles at each corner
        """
        for i in goal[0]:
            i[0][0] += random.randint(-1, 1)
            i[0][1] += random.randint(-1, 1)
        """
        
        corners = goal[0]

        ret, c_rvec, c_tvec = cv2.solvePnP(goal_points, np.float32(corners), K, None)

        #print c_rvec[1]
        
        c_pos = np.multiply(-1, np.dot(np.linalg.inv(cv2.Rodrigues(c_rvec)[0]), c_tvec))

        c_rot = cv2.Rodrigues(np.transpose(cv2.Rodrigues(c_rvec)[0]))[0]

        print c_rot[1]
        
        x = c_pos[0]
        y = c_pos[1]
        z = c_pos[2]
        distance = sqrt(c_pos[0]**2+c_pos[2]**2)
        skew = degrees(atan(c_pos[0]/c_pos[2]))
        angle = degrees(c_rot[1]) - skew

        v_obj = normalize(c_pos[0]+goal_width/2, c_pos[1]+goal_height/2, c_pos[2])
        v_img = normalize(0.5*(WIDTH-1)-corners[0][0][0], 0.5*(HEIGHT-1)-corners[0][0][1], -f)

        cross = np.cross(v_obj, v_img)
        angle2 = acos(max(min(np.dot(v_obj, v_img), 1.0), -1))
        
        angle2 = cross[1]*degrees(angle2) - skew
        
        #output.write(str(float((c_pos[0]-pos[0]))) + ", " + str(float((c_pos[1]-pos[1]))) + ", " + str(float((c_pos[2]-pos[2]))) + ", " + str(float(sqrt(c_pos[0]**2+c_pos[2]**2)-sqrt(pos[0]**2+pos[2]**2))) + ", " + str(float(sqrt(c_pos[0]**2+c_pos[2]**2))) + "\n") 
        output.write(str(float(degrees(rot[1]))) + ", " + str(float(degrees(rot[1]) - a_skew)) + ", " + str(float(degrees(c_rot[1]) - skew)) + ", " + str(float((angle2))) + "\n")
        
        draw_HUD(img, fps, x, y, z, distance, skew, angle) #draw the hud on the flipped image
        cv2.imshow('tyr-vision', img) #create a window with the complete image

        counter = counter + 1
        
        key = cv2.waitKey(30) & 0xff #press ESC to quit
        if key == 27:
            print "Total frames: %d" %total_frames
            break
        
    cv2.destroyAllWindows()
    output.close()
    
def normalize(x, y, z):
    x, y, z = float(x), float(y), float(z)
    d = sqrt(x**2+y**2+z**2)
    return (x/d, y/d, z/d)

if __name__ == '__main__':
        main()
