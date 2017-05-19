import cv2
import urllib2
import numpy as np

stream = urllib2.urlopen("http://10.0.8.23/live")
cap = cv2.VideoCapture("http://10.0.8.23/live")

def test():
    bytes = ' ' #bytestream from the camera

    while True:
        
        bytes += stream.read(1024) #read the bytes
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')

        
        if a!=-1 and b!=-1:
            
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),1) #turn into image

            cv2.imshow('image', img) #create a window with the complete image
            key = cv2.waitKey(30) & 0xff #press ESC to quit
            if key == 27:
                break

    cv2.destroyAllWindows()

def test2():
    while True:
        ret, img = cap.read()
        if ret == True:
            cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff #press ESC to quit
        if key == 27:
            break
    cv2.destroyAllWindows()
    
test2()
