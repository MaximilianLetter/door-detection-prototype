"""
List of all parameters used in the door detection process.
"""

# CONTRAST
# cv2.addWeighted(img, alpha, img, beta, gamma)
alpha = 1.5
beta = 0
gamma = 0

# BLUR
# cv2.GaussianBlur(gray, ksize, sigmaX)
ksize = (3,3)
sigmaX = 2.5

# EDGES
# cv2.Canny(blurred, lower, upper)
sigma = 0.33
v = np.median(blurred)
lower = int(max(0, (1.0 - sigma) * v) / 2)
upper = int(min(255, (1.0 + sigma) * v))

# CORNERS
# corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, mask=mask)
maxCorners = 40
qualityLevel = 0.05
minDistance = 10

off = 10
~mask = [off, height - off, off, width - off]

# DOOR POST GROUPING
THRESH_DIST_MAX = height * 0.85
THRESH_DIST_MIN = height * 0.3
ANGLE_MAX = 180
ANGLE_MIN = 50 # if > 60 nothing is found => check numbers

# DOOR CANDIDATE GROUPING
ANGLE_MAX = 10 # pixel
LENGTH_DIFF_MAX = 0.15
ASPECT_RATIO_MIN = 0.35
ASPECT_RATIO_MAX = 0.7
MAX_BOTTOM_LENGTH = 1.1

# TEST CANDIDATE
RECT_THRESH = 0.85
LINE_THRESH = 0.4
LINE_WIDTH = 2
BOT_LINE_BONUS = percentage * 0.25

# CHOOSE BEST CANDIDATE
UPVOTE_FACTOR = 1.2
