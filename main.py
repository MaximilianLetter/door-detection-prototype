import cv2
import sys
import numpy as np


def detect(img):
    img = resize(img, 500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Auto thresholds for now
    sigma = 0.33
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(gray, lower, upper)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, 50, 20)
    img = showLines(img, lines)

    return img

def showLines(img, lines):
    for i in range(0, len(lines)):
        # HoughLinesP or HoughLines
        if len(lines[i][0]) == 4:
            x1, y1, x2, y2 = lines[i][0]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        else:
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

    return img

def resize(img, width):
    h, w = img.shape[:2]
    height = int(h * (width / w))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized

def stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret_cam, frame = cap.read()

        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q') or ch == 27:
            break

        frame = detect(frame)

        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()

def single(path):
    img = cv2.imread('images/' + path + '.jpg')

    img = detect(img)

    cv2.imshow('frame', img)
    ch = cv2.waitKey(0)


# USAGE
# python main.py door_X --> single image
# python main.py --> stream
def main():
    if len(sys.argv) > 1:
        single(sys.argv[1])
    else:
        stream()

main()
