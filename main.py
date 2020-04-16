import cv2
import sys
import numpy as np
import math

# Detection method that gets used in all cases
def detect(img):
    roiOffset = 5
    img = resize(img, 120)
    width, height = img.shape[:2]
    # img = img[roiOffset:width-roiOffset, roiOffset:height-roiOffset]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3,3), 0.5)

    # Auto thresholds for now
    sigma = 0.33
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(blurred, lower, upper)

    # OPTION 1: Hough Transform for extracting lines
    # img = houghOperations(img, edges)

    # OPTION 2: LSD LineSegmentDetector
    # NOTE: Was removed in OpenCV 4.1.0+
    # img = lsdOperations(gray)

    # Option 3: FLD FastLineDetector
    # img = fldOperations(gray)

    # Option 4: Saliency [multiple saliency objects exist]
    # sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    # sal = cv2.saliency.StaticSaliencyFineGrained_create()
    # success, img = sal.computeSaliency(img)

    # Option 5: Corners
    img = useCorners(img, edges, gray)

    # ctest1 = [1, 0]
    # ctest2 = [0, 600]
    #
    # print(np.rad2deg(getOrientation(ctest1, ctest2)))

    return img

# START---Detection of doors with the use of corners and edges
def useCorners(img, edges, gray):
    # corners = cv2.goodFeaturesToTrack(edges, 50, 0.1, 10)

    off = 15
    roi = [off, edges.shape[0] - off, off, edges.shape[1] - off]
    mask = np.zeros_like(gray)
    mask[roi[0]:roi[1], roi[2]:roi[3]] = 255

    corners = cv2.goodFeaturesToTrack(gray, 50, 0.05, 5, mask=mask)

    for c in corners:
        x, y = c.ravel()
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    print(len(corners))

    # Group corners
    c_groups = groupCorners(corners, edges)

    # DRAW DOOR POSTS
    # for g in c_groups:
    #     c1 = tuple(g[0])
    #     c2 = tuple(g[1])
    #     cv2.line(img, c1, c2, (0, 255, 0))

    # DRAW ALL CANDIDATES
    # for g in c_groups:
    #     pts = np.array([g], np.int32)
    #     pts = pts.reshape((-1,1,2))
    #     cv2.polylines(img, [pts], True, (0,255,255), 1, cv2.LINE_AA)

    # Evaluate found groups and do further processing
    # edges = cv2.dilate(edges, (7,7), iterations=3)
    cv2.imshow('edges', edges)

    doors = []
    doorsRanking = []
    for g in c_groups:
        percentage = testCandidate(g, edges)

        if percentage > 0.7:
            doors.append(g)
            doorsRanking.append(percentage)

    # for door in doors:
    #     pts = np.array([door], np.int32)
    #     pts = pts.reshape((-1,1,2))
    #     cv2.polylines(img, [pts], True, (0,255,255), 1, cv2.LINE_AA)

    if len(doors):
        door = doors[np.argmax(doorsRanking)]
        pts = np.array([door], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,255), 1, cv2.LINE_AA)

    return img

def groupCorners(corners, img):
    height, width = img.shape[:2]

    THRESH_DIST_MAX = height * 0.8
    THRESH_DIST_MIN = height * 0.4

    # Goal is as high as possible somehow
    THRESH_ORI_MAX = 180
    THRESH_ORI_MIN = 54

    doorPosts = []

    done = np.zeros(len(corners))

    # Assuming that door posts are almost vertical
    for i, c1 in enumerate(corners):
        c1 = c1.ravel()
        for j, c2 in enumerate(corners):

            # Filter out duplicates
            if done[j] == True:
                continue

            c2 = c2.ravel()

            distance = getDistance(c1, c2)
            if distance < THRESH_DIST_MIN or distance > THRESH_DIST_MAX:
                continue

            orientation = np.degrees(getOrientation(c1, c2))
            if orientation < THRESH_ORI_MIN or orientation > THRESH_ORI_MAX:
                continue

            # sort so that the high point is always the first
            group = sorted([c1, c2], key=lambda k: [k[1], k[0]])

            doorPosts.append(group)

        # after all possibilities with c1 are done delete it
        done[i] = True

    print('DOORPOSTS', len(doorPosts))

    THRESH_DIST_MAX = THRESH_DIST_MAX * 0.5
    THRESH_DIST_MIN = THRESH_DIST_MIN * 0.5

    THRESH_ORI_MAX = 20

    cornerGroups = []
    done = np.zeros(len(doorPosts))

    # Possible door posts are collected, try to join them together
    for i, line1 in enumerate(doorPosts):
        c11, c12 = line1
        length1 = getDistance(c11, c12)
        for j, line2 in enumerate(doorPosts):

            # Filter out duplicates
            if done[j] == True:
                continue

            c21, c22 = line2
            length2 = getDistance(c21, c22)

            # if one of the points is the same -> continue
            if np.array_equal(c11, c21) or np.array_equal(c12, c22):
                continue

            # if the length of door posts is too different -> continue
            lengthDiff = abs(length1 - length2)
            if lengthDiff > length1 * 0.25 or lengthDiff > length2 * 0.25:
                continue

            distance = getDistance(c11, c21)
            if distance < THRESH_DIST_MIN or distance > THRESH_DIST_MAX:
                continue

            distance = getDistance(c12, c22)
            if distance < THRESH_DIST_MIN or distance > THRESH_DIST_MAX:
                continue

            orientation = np.degrees(getOrientation(c11, c21))
            if orientation > THRESH_ORI_MAX:
                continue

            orientation = np.degrees(getOrientation(c12, c22))
            if orientation > THRESH_ORI_MAX:
                continue

            # sort to draw door candidate
            group = [c11, c21, c22, c12]
            cornerGroups.append(group)

        # doorpost i does not need further testing
        done[i] = True

    print('CORNERGROUPS', len(cornerGroups))

    return cornerGroups

def testCandidate(corners, edges):
    # lineImg = np.zeros(edges.shape)
    p1, p2, p3, p4 = corners

    # NOTE: bottom line is not checked
    lines = [
        [p4, p1],
        [p1, p2],
        [p2, p3]
    ]

    percentages = []

    for line in lines:
        p1, p2 = line
        maskImg = np.zeros(edges.shape)
        cv2.line(maskImg, tuple(p1), tuple(p2), 1, 2)

        roi = edges[maskImg == 1]

        # percentage = np.count_nonzero(roi) / len(roi)
        percentage = np.count_nonzero(roi) / getDistance(p1, p2)
        percentage = min(percentage, 1.0)

        # print('LINE', percentage)
        # cv2.imshow('test', maskImg)
        # cv2.waitKey(0)

        if percentage < 0.4:
            return 0

        percentages.append(percentage)
    # pts = np.array([corners], np.int32)
    # pts = pts.reshape((-1,1,2))
    # cv2.polylines(lineImg, [pts], True, 1, 1, cv2.LINE_AA)

    # extract the part of the drawn lines
    # roi = edges[lineImg == 1]
    #
    # percentage = np.count_nonzero(roi) / len(roi)
    # print('::::')
    # print(len(roi))
    # print(np.count_nonzero(roi))
    # print(percentage)

    # cv2.imshow('edges', edges)



    return np.mean(percentages)
# END---Detection of doors with the use of corners and edges

# START---Detection of doors with lines from FastLineDetector
def fldOperations(img):
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(img)

    # cv2.imshow('PRE MERGE', fld.drawSegments(img, lines))

    hor_lines, vert_lines = processLines(lines)

    img = fld.drawSegments(img, lines)
    img = fld.drawSegments(img, np.concatenate((vert_lines, hor_lines), axis=0))

    # Line Selection naive algorithm
    h, w = img.shape[:2]

    candidates = findRectFromLines(hor_lines, vert_lines, w, h)
    print('CANDIDATES LEN', len(candidates))

    for c in candidates:
        pts = np.array([c], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,255), 1, cv2.LINE_AA)

    return img

def processLines(lines):
    lines_hor = []
    lines_vert = []

    # Seperate in horizontal and vertical lines
    for line in lines:
        line = line[0]
        ori = getOrientation(line)

        if 45 < ori < 135:
            lines_hor.append(line)
        else:
            lines_vert.append(line)

    # this changes things --> not all algorithms are working correctly
    # lines_vert = sorted(lines_vert, key=lambda line: line[1])
    # lines_hor = sorted(lines_hor, key=lambda line: line[0])

    # The higher the more room for merging
    dist_thresh = 5
    ori_thresh = 5

    lines_vert = groupLines(lines_vert, dist_thresh, ori_thresh)
    lines_hor = groupLines(lines_hor, dist_thresh, ori_thresh)

    # Reform for drawing [[x1, y1, x2, y2]]
    for i in range(len(lines_hor)):
        lines_hor[i] = np.array([lines_hor[i]])
    for i in range(len(lines_vert)):
        lines_vert[i] = np.array([lines_vert[i]])

    return np.array(lines_hor), np.array(lines_vert)

def groupLines(lines, dist_thresh, ori_thresh):
    seen = [lines[0]] # Start with first group containing first line
    for line in lines[1:]: # Check all other lines starting from second
        merged = False
        for index, line_ in enumerate(seen):
            dist = getDistanceLines(line, line_)
            ori = getOrientationDifferences(line, line_)

            if dist < dist_thresh and ori < ori_thresh:
                seen[index] = getDistanceLines(line, line_, True)
                merged = True
                break

        # only append if no line in seen fits
        if merged == False:
            seen.append(line)

    print('MERGED: ', len(lines) - len(seen))
    return seen

def findRectFromLines(hor_lines, vert_lines, w, h):
    """
    Naive algorithm for detecting a rectangle.
    Selection when multiple candidates are found is missing.
    Due not merged lines most results do not work out.
    Also only working on frontal images of doors.
    """
    MIN_DIST = w / 4
    HOR_THRESH = 0.1
    VERT_THRESH = 2.0
    POINT_THRESH = 5.0

    candidates = []

    for line in hor_lines:
        door_corners = []

        x1, y1, x2, y2 = line.ravel()
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        m = abs((y2-y1) / (x2-x1))

        # Horizontal line found
        if dist > MIN_DIST and m < HOR_THRESH:
            door_corners.append([x1, y1])
            door_corners.append([x2, y2])

            height = (y1+y2) / 2

            for line_vert in vert_lines:
                x1_v, y1_v, x2_v, y2_v = line_vert.ravel()

                dist = np.sqrt((x1_v-x2_v)**2 + (y1_v-y2_v)**2)
                if x1_v != x2_v:
                    m = abs((y2_v-y1_v) / (x2_v-x1_v))
                else:
                    m = float("inf")

                # NOTE: m could be aborted here maybe
                if dist > (MIN_DIST * 1.5) and m > VERT_THRESH:
                    # Check if points are possible connection points
                    dist_x11 = abs(x1-x1_v)
                    dist_y11 = abs(y1-y1_v)

                    dist_x12 = abs(x1-x2_v)
                    dist_y12 = abs(y1-y2_v)

                    dist_x21 = abs(x2-x1_v)
                    dist_y21 = abs(y2-y1_v)

                    dist_x22 = abs(x2-x2_v)
                    dist_y22 = abs(y2-y2_v)

                    if (dist_x11 < POINT_THRESH and dist_y11 < POINT_THRESH) or (dist_x21 < POINT_THRESH and dist_y21 < POINT_THRESH):
                        if y2_v > height:
                            door_corners.append([x2_v, y2_v])

                    if (dist_x12 < POINT_THRESH and dist_y12 < POINT_THRESH) or (dist_x22 < POINT_THRESH and dist_y22 < POINT_THRESH):
                        if y1_v > height:
                            door_corners.append([x1_v, y1_v])

                if len(door_corners) == 4:
                    candidates.append(door_corners)
                    door_corners = [[x1, y1], [x2, y2]]

    return candidates
# END---Deteciton of doors with lines from FastLineDetector

# Helper functions
def getOrientationDifferences(line1, line2):
    ori1 = getOrientation(line1)
    ori2 = getOrientation(line2)

    return abs(ori1-ori2)

def getDistanceLines(line1, line2, merge=False):
    # x1, y1, x2, y2 = line1.ravel()
    # x1_, y1_, x2_, y2_ = line2.ravel()
    x1, y1, x2, y2 = line1
    x1_, y1_, x2_, y2_ = line2

    dist1 = getDistance((x1, y1), (x1_, y1_))
    dist2 = getDistance((x2, y2), (x2_, y2_))
    dist3 = getDistance((x1, y1), (x2_, y2_))
    dist4 = getDistance((x2, y2), (x1_, y1_))

    if not merge:
        dist = min(dist1, dist2, dist3, dist4)
        return dist
    else:
        dist = max(dist1, dist2, dist3, dist4)
        if dist1 == dist:
            new_line = [x1, y1, x1_, y1_]
        elif dist2 == dist:
            new_line = [x2, y2, x2_, y2_]
        elif dist3 == dist:
            new_line = [x1, y1, x2_, y2_]
        else:
            new_line = [x2, y2, x1_, y1_]

        return new_line

def getDistance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def getOrientation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dir = 179
    if x1 != x2:
        dir = (2 / np.pi) * np.arctan(abs(y2-y1) / abs(x2-x1))
    return dir

def getOrientationLine(line):
    orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
    return math.degrees(orientation)

def resize(img, width):
    h, w = img.shape[:2]
    height = int(h * (width / w))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized

# No longer used methods
def lsdOperations(img):
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    img = lsd.drawSegments(img, lines)

    return img

def houghOperations(img, edges):
    # Experiment extracting vertical and horizontal lines, standard Hough Transform
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    # vert_lines = [line for line in lines if (abs(line[0][1]) < 0.1 or abs(line[0][1]) > np.pi * 2 - 0.1)]
    # hor_lines = [line for line in lines if (abs(line[0][1]) > (np.pi / 2)-0.1 and abs(line[0][1]) < (np.pi / 2) + 0.1)]
    #
    # img = showLines(img, vert_lines)
    # img = showLines(img, hor_lines)

    # Experiment extracting vertical and horizontal lines, probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 5, np.pi/180, 10, 50, 5)

    # straight_lines = [l for l in lines if direction(l[0][0], l[0][1], l[0][2], l[0][3]) == 0]
    # img = showLines(img, straight_lines)

    img = showLines(img, lines)

    return img

def direction(x1, y1, x2, y2):
    dir = 0
    if x1 != x2:
        dir = (2 / np.pi) * np.arctan(abs(y2-y1) / abs(x2-x1))
    return dir

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

# Decision on input
def stream(input = 0):
    fullPath = 0
    webCam = True

    if input != 0:
        fullPath = 'videos/' + input + '.mp4'
        webCam = False

    cap = cv2.VideoCapture(fullPath)

    if webCam:
        resultSize = (int(cap.get(3)), int(cap.get(4)))
    else:
        resultSize = (450, 600)

    out = cv2.VideoWriter('results/video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, resultSize)

    while True:
        ret_cam, frame = cap.read()

        if not webCam:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q') or ch == 27:
            break

        frame = detect(frame)

        frame = cv2.resize(frame, resultSize, interpolation = cv2.INTER_AREA)

        cv2.imshow('frame', frame)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def single(path):
    img = cv2.imread('images/' + path + '.jpg')

    img = detect(img)

    dim = (640, 480)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('frame', resized)
    cv2.imwrite('results/' + path + '.jpg', img)

    ch = cv2.waitKey(0)

# USAGE
# python main.py door_X --> single image
# python main.py --> stream
def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'vid':
            stream(sys.argv[2])
        else:
            single(sys.argv[1])
    else:
        stream()

main()
