import cv2
import sys
import numpy as np
import math
import time

def detect(img):
    """
    Main detection function.
    Set up the image, gray image and edge image.
    """

    print('###################')

    overallTime = time.time()
    startTime = time.time()

    # Sample the image down to 120 width image
    img = resize(img, 120)
    shape = img.shape[:2]
    height, width = shape

    # Incrase contrast, this a kind of workaround
    img = cv2.addWeighted(img, 1.5, img, 0, 0)

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blurred = cv2.GaussianBlur(gray, (3,3), 2.5)

    # Generate edge image with Canny
    sigma = 0.33
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v) / 2)
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(blurred, lower, upper)

    # Show edges
    cv2.imshow('edges', edges)

    # Create region of interest so corners at the image borders can be omitted
    off = 10
    roi = [off, height - off, off, width - off]
    mask = np.zeros_like(gray)
    mask[roi[0]:roi[1], roi[2]:roi[3]] = 255

    # Generate corners to track and use for corner grouping
    corners = cv2.goodFeaturesToTrack(gray, 40, 0.05, 10, mask=mask)
    # corners = cv2.goodFeaturesToTrack(gray, 50, 0.001, 7, mask=mask, useHarrisDetector=True)

    # Show corners for development
    for c in corners:
        x, y = c.ravel()
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    print('_preperations_:', time.time() - startTime)
    startTime = time.time()

    # Group corners to vertical lines that represent a door post
    doorPosts = cornersToDoorposts(corners, height)
    # print('DOORPOSTS:', len(doorPosts))

    print('_doorposts_:', time.time() - startTime)
    startTime = time.time()

    # Show doorPosts for development
    # for g in doorPosts:
    #     c1 = tuple(g[0])
    #     c2 = tuple(g[1])
    #     cv2.line(img, c1, c2, (0, 255, 0))
    #
    # cv2.imshow('doorposts', img)
    # cv2.waitKey(0)

    # Build candidates out of the door posts
    rectangles = buildRectangles(doorPosts)

    print('_rectangles_:', time.time() - startTime)
    startTime = time.time()

    doors = []
    doorsRanking = []
    RECT_THRESH = 0.85
    for rect in rectangles:
        percentage = testCandidate(rect, edges)

        if percentage > RECT_THRESH:
            doorsRanking.append(percentage)
            doors.append(rect)

    print('_candidates_:', time.time() - startTime)
    startTime = time.time()

    # Show door candidates for development
    for door in doors:
        pts = np.array([door], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (255,0,0), 1, cv2.LINE_AA)

    # Compare the candidates to get the best one and draw it
    if len(doors):
        door = chooseBestCandidate(doors, doorsRanking, gray)
        # pts = np.array([door], np.int32)
        # pts = pts.reshape((-1,1,2))
        # cv2.polylines(img, [pts], True, (0,255,255), 1, cv2.LINE_AA)
    else:
        return False, [], img

    print('_choosebest_:', time.time() - startTime)

    print('_overall_:', time.time() - overallTime)

    print('STATS---------')
    print('CORNERS:', len(corners))
    print('DOORPOSTS:', len(doorPosts))
    print('RECTANGLES:', len(rectangles))
    print('CANDIDATES:', len(doors))
    print('--------------')

    return True, door, img

def cornersToDoorposts(corners, height):
    """
    Group corners by their relation to each other.
    Vertical lines as representives of door posts are selected.
    """

    # Min and max length of a door post
    LENGTH_MAX = height * 0.85
    LENGTH_MIN = height * 0.3

    # Goal is as high as possible somehow
    ANGLE_MAX = 180
    ANGLE_MIN = 50

    doorPosts = []
    done = np.zeros(len(corners))

    # Check every corner in the image for potential door posts
    for i, c1 in enumerate(corners):
        c1 = c1.ravel()
        for j, c2 in enumerate(corners):

            # Filter out duplicates
            if done[j] == True:
                continue

            c2 = c2.ravel()

            distance = getDistance(c1, c2)
            if distance < LENGTH_MIN or distance > LENGTH_MAX:
                continue

            orientation = np.degrees(getOrientation(c1, c2))
            if orientation < ANGLE_MIN or orientation > ANGLE_MAX:
                continue

            # Sort so that the high point is always the first
            group = sorted([c1, c2], key=lambda k: [k[1], k[0]])

            doorPosts.append(group)

        # After all possibilities with c1 are done delete it
        done[i] = True

    return doorPosts

def buildRectangles(doorPosts):
    """
    Connect the found door posts to a door candidate, if angle and distance
    between fits.
    """

    # Maximum angle of the horizontal lines
    ANGLE_MAX = 10

    LENGTH_DIFF_MAX = 0.15

    ASPECT_RATIO_MIN = 0.35
    ASPECT_RATIO_MAX = 0.7

    MAX_BOTTOM_LENGTH = 1.1
    MIN_BOTTOM_LENGTH = 0.7

    cornerGroups = []
    done = np.zeros(len(doorPosts))

    # Check every line with every other line
    for i, line1 in enumerate(doorPosts):
        c11, c12 = line1
        length1 = getDistance(c11, c12)
        for j, line2 in enumerate(doorPosts):

            # Filter out duplicates
            if done[j] == True:
                continue

            c21, c22 = line2

            # if one of the points is the same -> continue
            if np.array_equal(c11, c21) or np.array_equal(c12, c22):
                continue

            length2 = getDistance(c21, c22)

            # if the length of door posts is too different -> continue
            lengthDiff = abs(length1 - length2)
            if lengthDiff > length1 * LENGTH_DIFF_MAX or lengthDiff > length2 * LENGTH_DIFF_MAX:
                continue

            # TODO look up real door aspect ratios
            lengthAVG = (length1 + length2) / 2
            minLength = lengthAVG * ASPECT_RATIO_MIN
            maxLength = lengthAVG * ASPECT_RATIO_MAX

            distanceTop = getDistance(c11, c21)
            if distanceTop < minLength or distanceTop > maxLength:
                continue

            distanceBot = getDistance(c12, c22)
            if distanceBot > distanceTop * MAX_BOTTOM_LENGTH or distanceBot < distanceTop * MIN_BOTTOM_LENGTH:
                continue

            orientation = np.degrees(getOrientation(c11, c21))
            if orientation > ANGLE_MAX:
                continue

            orientation = np.degrees(getOrientation(c12, c22))
            if orientation > ANGLE_MAX:
                continue

            # Sort to draw the door candidate
            group = [c11, c21, c22, c12]
            cornerGroups.append(group)

        # doorpost i does not need further testing
        done[i] = True

    return cornerGroups

def testCandidate(corners, edges):
    """
    Test if the lines of the corner group overlap with pixels in the edge image
    """
    p1, p2, p3, p4 = corners

    # NOTE: Bottom line is sometimes not present in the edge image and therefor
    # treated specially
    lines = [
        [p4, p1],
        [p1, p2],
        [p2, p3],
        [p3, p4] # Bottom line
    ]

    percentages = []
    bonus = 0

    LINE_WIDTH = 2
    LINE_THRESH = 0.5
    BOT_LINE_BONUS = 0.25

    for i, line in enumerate(lines):
        p1, p2 = line
        maskImg = np.zeros(edges.shape)
        cv2.line(maskImg, tuple(p1), tuple(p2), 1, LINE_WIDTH)

        roi = edges[maskImg == 1]

        # Potential for improvement -> is there really a line?
        percentage = np.count_nonzero(roi) / getDistance(p1, p2)
        percentage = min(percentage, 1.0)

        # Bottom line
        if i == 3:
            bonus = percentage * BOT_LINE_BONUS
            break

        if percentage < LINE_THRESH:
            return 0

        percentages.append(percentage)

    score = np.average(percentages) + bonus

    return score

def chooseBestCandidate(doors, scores, img):
    """
    Select the best candidate by comparing their scores. The scores get
    increased if special requirements are met.
    """
    UPVOTE_FACTOR = 1.2
    DOOR_IN_DOOR_DIFF_THRESH = 10 # pixels
    COLOR_DIFF_THRESH = 50
    ANGLE_DEVIATION_THRESH = 10

    for i, corners in enumerate(doors):
        # Unpack corners
        topLeft, topRight, botRight, botLeft = corners

        # Angle stability
        angle1 = getCornerAngles(botLeft, topLeft, topRight)
        angle2 = getCornerAngles(botRight, topRight, topLeft)
        angle3 = getCornerAngles(topLeft, botLeft, botRight)
        angle4 = getCornerAngles(botLeft, botRight, topRight)

        # Get overall similar angles
        mean = np.mean([angle1, angle2, angle3, angle4])
        angleDeviation = max([abs(mean - angle1), abs(mean - angle2), abs(mean - angle3), abs(mean - angle4)])
        if angleDeviation < ANGLE_DEVIATION_THRESH:
            scores[i] = scores[i] * UPVOTE_FACTOR

        # Alternative check if opposing angles are similar
        # angleOpposite1 = abs(angle1 - angle4)
        # angleOpposite2 = abs(angle2 - angle3)
        # angleStability.append(angleOpposite1 + angleOpposite2)

        # Color difference inside and outside
        left = int(min(botLeft[0], topLeft[0]))
        right = int(max(botRight[0], topRight[0]))
        top = int(max(topRight[1], topLeft[1]))
        bottom = int(min(botLeft[1], botRight[1]))

        mask = np.zeros(img.shape, np.uint8)
        mask[top:bottom, left:right] = 255 #top, bottom switched
        maskInv = cv2.bitwise_not(mask)

        # inner = np.median(cv2.bitwise_and(img, img, mask=mask))
        # outer = np.median(cv2.bitwise_and(img, img, mask=maskInv))

        inner = np.average(img[mask == 255])
        outer = np.average(img[maskInv == 255])
        colorDiff = abs(inner - outer)
        if colorDiff > COLOR_DIFF_THRESH:
            scores[i] = scores[i] * UPVOTE_FACTOR

        # cv2.imshow('a', cv2.bitwise_and(img, img, mask=mask))
        # cv2.imshow('b', cv2.bitwise_and(img, img, mask=maskInv))

        # Check if another door is part of this door
        for corners2 in doors:
            topLeft2, topRight2, botRight2, botLeft2 = corners2

            if np.array_equal(corners, corners2):
                continue

            if np.array_equal(topLeft, topLeft2) and np.array_equal(topRight, topRight2):
                top2 = int(max(topRight2[1], topLeft2[1]))
                bottom2 = int(min(botLeft2[1], botRight2[1]))

                height = abs(top - bottom)
                height2 = abs(top2 - bottom2)
                diff = abs(height - height2)

                if height > height2 and diff > DOOR_IN_DOOR_DIFF_THRESH:
                    scores[i] = scores[i] * UPVOTE_FACTOR
                    break

    result = doors[np.array(scores).argmax()]
    # print('WINNING SCORE: ', max(scores))

    return result

# Smoothing video
def checkDifferences(door, prev):
    DIFF_THRESH_SMALL = 5
    DIFF_THRESH_BIG = 10
    diffCounter = 0

    if prev == []:
        return False, door

    door = np.array(door)
    prev = np.array(prev)

    diffs = door - prev

    for diff in diffs:
        x, y = diff
        x = abs(x)
        y = abs(y)
        if x > DIFF_THRESH_SMALL or y > DIFF_THRESH_SMALL:
            diffCounter += 1
            if x > DIFF_THRESH_BIG or y > DIFF_THRESH_BIG:
                return False, []

    if diffCounter > 2:
        return False, []

    return True, door

# Helper functions
def getCornerAngles(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def getOrientationDifferences(line1, line2):
    ori1 = getOrientationLine(line1)
    ori2 = getOrientationLine(line2)

    return abs(ori1-ori2)

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

def resize(img, width):
    h, w = img.shape[:2]
    height = int(h * (width / w))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized

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

    previousDoor = []
    prevCounter = 0

    while True:
        ret_cam, frame = cap.read()

        if not webCam:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q') or ch == 27:
            break

        found, door, frame = detect(frame)

        if found:
            show, door = checkDifferences(door, previousDoor)
            if show:
                pts = np.array([door], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame, [pts], True, (0,255,255), 1, cv2.LINE_AA)
                previousDoor = door
                prevCounter = 0
            else:
                pts = np.array([previousDoor], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame, [pts], True, (0,255,255), 1, cv2.LINE_AA)

                prevCounter += 1
                if prevCounter > 2:
                    prevCounter = 0
                    previousDoor = door


        else:
            if previousDoor != []:
                pts = np.array([previousDoor], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame, [pts], True, (0,255,255), 1, cv2.LINE_AA)

                prevCounter += 1
                if prevCounter > 2:
                    previousDoor = []
                    prevCounter = 0

        frame = cv2.resize(frame, resultSize, interpolation = cv2.INTER_AREA)

        cv2.imshow('frame', frame)
        out.write(frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()

def single(path):
    img = cv2.imread('images/' + path + '.jpg')

    bool, door, img = detect(img)

    dim = (450, 600)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('frame', resized)
    cv2.imwrite('results/' + path + '.jpg', img)

    ch = cv2.waitKey(0)

# USAGE
# python main.py door_X --> single image
# python main.py vid door_X --> video
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
