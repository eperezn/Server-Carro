
import base64
import cv2
import time
import requests


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        buffer = cv2.imencode(".jpg", img)
        txt = base64.b64encode(buffer)
        cv2.imshow('my webcam', img)
         r = requests.post("http://192.168.0.105:5000/img", json={"img":str(text)})
        print(r.json())

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    show_webcam()