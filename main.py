import cv2
import sys


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = resize(gray, 400)

    return gray

def resize(img, width):
    h, w = img.shape
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
