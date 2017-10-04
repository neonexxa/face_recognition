# from goprocam import GoProCamera
# from goprocam import constants
# gopro = GoProCamera.GoPro()
# gopro.stream("rtp://10.5.5.9:8554")

# from goprohero import GoProHero
# camera = GoProHero(password='azrielcbq8676')
# camera.command('record', 'on')
# status = camera.status()

import cv2
import numpy as np

from goprohero import GoProHero
camera = GoProHero(password='azrielcbq8676')
camera.command('record', 'on')
# status = camera.status()
# from goprocam import GoProCamera
# from goprocam import constants
# cascPath="./haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# gpCam = GoProCamera.GoPro()
cap = cv2.VideoCapture(camera)
while True:
    ret, frame = cap.read()
    cv2.imshow("GoPro OpenCV", camera)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# ffplay -an -fflags nobuffer -f:v mpegts -probesize 8192 rtp://10.5.5.9:8554
# http://10.5.5.9/gp/gpControl/execute?p1=gpStream&c1=restart