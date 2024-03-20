from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import os
import sys

IMAGE_NAME = 'snap1.jpg'
VIDEO_NAME = "vid1.mp4"
MODEL_FILENAME = "frozen_east_text_detection.pb"

IMAGE_PATH = os.path.join(os.curdir, "media", IMAGE_NAME)
VIDEO_PATH = os.path.join(os.curdir, "media", VIDEO_NAME)
MODEL_PATH = os.path.join(os.curdir, MODEL_FILENAME)

MIN_CONFIDENCE = 0.5
WIDTH, HEIGHT = 1280, 1280
LAYERS = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

isVideo = False

if len(sys.argv) > 1:
    if sys.argv[1] == 'video':
        isVideo = True


def getVideoCaptureDevice():
    if not os.path.isfile(VIDEO_PATH):
        return None

    cap = cv2.VideoCapture(VIDEO_PATH)

    return cap


def showProcessedFrame(image, window_name, model, scale_factor=None):
    start = time.perf_counter()

    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (WIDTH, HEIGHT)
    ratioWidth = W / float(newW)
    ratioHeight = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    model_start = time.perf_counter()

    model.setInput(blob)
    (scores, geometry) = model.forward(LAYERS)

    detection_time = f"Detection Took - {time.perf_counter() - model_start:.2f} seconds"

    print(detection_time)

    (numRows, numCols) = scores.shape[2:4]

    boxes = getBoundingBoxes(numRows, numCols, scores, geometry)

    final_image = drawBoundingBoxes(ratioHeight, ratioWidth, boxes, orig)

    if isVideo:

        fps = f"{1 / (time.perf_counter() - start):.2f} FPS"

        cv2.putText(orig, fps, (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 3)

    if scale_factor:
        cv2.imshow(window_name, cv2.resize(final_image, (0, 0),
                                           fx=scale_factor[0], fy=scale_factor[1]))

    else:
        cv2.imshow(window_name, final_image)


def loadModel():
    if not os.path.isfile(MODEL_PATH):
        print("Model not found :(")
        sys.exit(1)

    return cv2.dnn.readNet(MODEL_PATH)


def getBoundingBoxes(numRows, numCols, scores, geometry):
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < MIN_CONFIDENCE:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return non_max_suppression(np.array(rects), probs=confidences)


def drawBoundingBoxes(ratioHeight, ratioWidth, boundingBoxes, original_image):
    final_image = None
    for (startX, startY, endX, endY) in boundingBoxes:
        startX = int(startX * ratioWidth)
        startY = int(startY * ratioHeight)

        endX = int(endX * ratioWidth)
        endY = int(endY * ratioHeight)

        final_image = cv2.rectangle(
            original_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return final_image


def main():
    cap = getVideoCaptureDevice()

    if not cap:
        print("Video File Not Found :(")
        sys.exit(1)

    model = loadModel()

    print(f"Is video - {isVideo}")

    if isVideo:
        while cap.isOpened():
            _, frame = cap.read()
            showProcessedFrame(frame, "Base Video", model=model,
                               scale_factor=(0.5, 0.5))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    else:
        img = cv2.imread(IMAGE_PATH)
        showProcessedFrame(img, "Image", model=model, scale_factor=(0.5, 0.5))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
