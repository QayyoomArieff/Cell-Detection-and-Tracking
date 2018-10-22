import cv2
import numpy as np

frame_dots = 0
frame_copy = 0

#Mouse callback
def annotate(event, x, y, flags, params):

    if event == 1:
        frame_dots[y][x] = 255

        global frame_copy

        frame_copy = cv2.drawMarker(frame_copy, position=(x, y), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                                  markerSize=4, thickness=1)
        cv2.imshow('image', frame_copy)

# Writes the labelled images to data folder (Change if necessary)
def Video(path):

    vid = cv2.VideoCapture(path)

    next, frame = vid.read()
    count = 1

    global frame_dots
    frame_dots = np.zeros(np.shape(frame))

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 800)

    while(next):
        global frame_copy
        frame_copy = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("data/"+str(count).zfill(3)+"cell.tif", frame)  #Change Write folder

        cv2.setMouseCallback('image', annotate)
        cv2.imshow('image', frame_copy)

        cv2.waitKey(0)

        cv2.imwrite("data/"+str(count).zfill(3)+"dots.png", frame_dots) #Change Write folder

        frame_dots = np.zeros(np.shape(frame))

        next, frame = vid.read()
        count += 1

    print (count)
    return


if __name__ == '__main__':
    # Uncomment and change the video path
    # Video('phase_contrast_smegmatis.mov')
