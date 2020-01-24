import cv2
import keyboard
import time
vid = cv2.VideoCapture(0)
count = 0
# $ xrandr --output eDP-1 --brightness 0.75 to adjust the screen brightness. toggle between 0.1 and 1
while True:
    count += 1
    if count % 10 == 0:
        print("caught")
        return_value, frame = vid.read()
        if keyboard.is_pressed('z'):
            # this use of ticme library ensures unique file names
            cv2.imwrite("./dataset/looking/"+str(int(time.time()))+".jpg", frame)
        else:
            cv2.imwrite("./dataset/notlooking/"+str(int(time.time()))+".jpg", frame)