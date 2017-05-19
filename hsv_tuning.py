import cv2
import numpy as np 
import time
import urllib2

stream = urllib2.urlopen("http://10.0.8.23/live")
COLOR_MAX = np.array([65, 254, 230], np.uint8)

def main():
	bytes = ''
	h = 40
	s = 75
	v = 130
	while True:
		bytes += stream.read(1024)
		a = bytes.find('\xff\xd8')
		b = bytes.find('\xff\xd9')

		if a!=-1 and b!=-1:
			jpg = bytes[a:b+2]
			bytes= bytes[b+2:]
			img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),1)
		
			COLOR_MIN = np.array([h, s, v], np.uint8)
			hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
			cv2.imshow('tyr-vision', frame_threshed)

			key = cv2.waitKey(30) & 0xff #quit on pressing 'esc'
			if key == 27:
				break
			elif key == ord('w'):
				h += 1
			elif key == ord('s'):
				h -= 1
			elif key == ord('a'):
				s += 1
			elif key == ord('d'):
				s -= 1
			elif key == ord('1'):
				v -= 1
			elif key == ord('2'):
				v += 1
			
			print "H value: %d" % h 
			print "S value: %d" % s
			print "V value: %d" % v
		else:
			bytes += stream.read(1024)
			a = bytes.find('\xff\xd8')
			b = bytes.find('\xff\xd9')
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
