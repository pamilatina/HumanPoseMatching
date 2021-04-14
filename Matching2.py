import cv2
import time
import numpy as np
from math import sqrt
import argparse

#image_file = cv2.imread("img.jpeg")
device = "cpu"
#parser = argparse.ArgumentParser(description='Run keypoint detection')
#parser.add_argument("--device", default="cpu", help="Device to inference on")
#parser.add_argument("--image_file", default="single.jpeg", help="Input image")

#args = parser.parse_args()

def euclidean_distance(u, v):
    """
    Returns the euclidean distance between vectors u and v. This is equivalent
    to the length of the vector (u - v).
    """
    diff = u - v
    return sqrt(np.dot(diff, diff))

def cosine_distance(u, v):
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is equal to
    1 - (u.v / |u||v|).
    """
    return 1 - (np.dot(u, v) / (sqrt(np.dot(u, u)) * sqrt(np.dot(v, v))))

def similarity_score(pose1, pose2):
    p1 = []
    p2 = []
    pose_1 = np.array(pose1, dtype=float)
    pose_2 = np.array(pose2, dtype=float)

    # Normalize coordinates
    pose_1[:,0] = pose_1[:,0] / max(pose_1[:,0])
    pose_1[:,1] = pose_1[:,1] / max(pose_1[:,1])
    pose_2[:,0] = pose_2[:,0] / max(pose_2[:,0])
    pose_2[:,1] = pose_2[:,1] / max(pose_2[:,1])

    for joint in range(pose_1.shape[0]):
        x1 = pose_1[joint][0]
        y1 = pose_1[joint][1]
        x2 = pose_2[joint][0]
        y2 = pose_2[joint][1]

        p1.append(x1)
        p1.append(y1)
        p2.append(x2)
        p2.append(y2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    # Looking to minimize the distance if there is a match
    # Computing two different distance metrics
    scoreA = cosine_distance(p1, p2)
    scoreB = euclidean_distance(p1, p2)

    print("Cosine Distance:", scoreA)
    print("Euclidean Distance:", scoreB)

MODE = "COCO"

if MODE == "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
    POSE_PAIRS2 = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODE == "MPI":
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]
    POSE_PAIRS2 = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

frame = cv2.imread("img.jpeg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

t = time.time()
# input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H


    if prob > threshold:
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

cv2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Output-Skeleton', frame)

cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))

frame2 = cv2.imread("image1.jpeg")
frameCopy2 = np.copy(frame2)
frameWidth2 = frame2.shape[1]
frameHeight2 = frame2.shape[0]
threshold2 = 0.1

net2 = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if device == "cpu":
    net2.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif device == "gpu":
    net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

t2 = time.time()
# input image dimensions for the network
inWidth2 = 368
inHeight2 = 368
inpBlob2 = cv2.dnn.blobFromImage(frame2, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

net2.setInput(inpBlob2)

output2 = net2.forward()
print("time taken by network : {:.3f}".format(time.time() - t2))

H2 = output2.shape[2]
W2 = output2.shape[3]

# Empty list to store the detected keypoints
points2 = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap2 = output2[0, i, :, :]

    # Find global maxima of the probMap.
    minVal2, prob2, minLoc2, point2 = cv2.minMaxLoc(probMap2)

    # Scale the point to fit on the original image
    x2 = (frameWidth2 * point2[0]) / W2
    y2 = (frameHeight2 * point2[1]) / H2


    if prob2 > threshold2:
        cv2.circle(frameCopy2, (int(x2), int(y2)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy2, "{}".format(i), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points2.append((int(x2), int(y2)))
    else:
        points2.append(None)

# Draw Skeleton
for pair in POSE_PAIRS:
    partA2 = pair[0]
    partB2 = pair[1]

    if points2[partA2] and points2[partB2]:
        cv2.line(frame2, points2[partA2], points2[partB2], (0, 255, 255), 2)
        cv2.circle(frame2, points2[partA2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

cv2.imshow('Output-Keypoints2', frameCopy2)
cv2.imshow('Output-Skeleton2', frame2)

cv2.imwrite('Output-Keypoints2.jpg', frameCopy2)
cv2.imwrite('Output-Skeleton2.jpg', frame2)

print("Total time taken : {:.3f}".format(time.time() - t2))
similarity_score(points,points2)

canvas = np.ones(frame.shape)
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(canvas, points[partA], points[partB], (255, 0, 255), 2)
        cv2.circle(canvas, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

for pair in POSE_PAIRS2:
    partA2 = pair[0]
    partB2 = pair[1]

    if points2[partA2] and points2[partB2]:
        cv2.line(canvas, points2[partA2], points2[partB2], (255, 0, 255), 2)
        cv2.circle(canvas, points2[partA2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

cv2.imshow('Output-Vis', canvas)


cv2.waitKey(0)