import numpy as np
import cv2 as cv
import math
import argparse

# Argument parser
parser = argparse.ArgumentParser(
    description="Use this script to run text detection and recognition on an image.")
parser.add_argument('--input', default='img.png', help='Path to input image')
parser.add_argument('--model', default='frozen_east_text_detection.pb', help='Path to the detector model.')
parser.add_argument('--ocr', default="CRNN_VGG_BiLSTM_CTC.onnx", help="Path to the OCR model.")
parser.add_argument('--width', type=int, default=320, help='Resize width for preprocessing.')
parser.add_argument('--height', type=int, default=320, help='Resize height for preprocessing.')
parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold.')
parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold.')
args = parser.parse_args()

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv.getPerspectiveTransform(vertices, targetVertices)
    result = cv.warpPerspective(frame, rotationMatrix, outputSize)
    return result

def decodeText(scores):
    text = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)

def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]
            if (score < scoreThresh):
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    return [detections, confidences]

def merge_boxes(boxes, confidences, threshold=0.5):
    merged_boxes = []
    merged_confidences = []
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, threshold, 0.4)
    indices = indices.flatten()

    for i in indices:
        merged_boxes.append(boxes[i])
        merged_confidences.append(confidences[i])

    return merged_boxes, merged_confidences

if __name__ == "__main__":
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    modelDetector = args.model
    modelRecognition = args.ocr

    # Load networks
    detector = cv.dnn.readNet(modelDetector)
    recognizer = cv.dnn.readNet(modelRecognition)

    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")

    frame = cv.imread(args.input)

    if frame is None:
        print(f"Error: Unable to load image at path '{args.input}'")
        exit()

    tickmeter = cv.TickMeter()

    height_ = frame.shape[0]
    width_ = frame.shape[1]
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    detector.setInput(blob)

    tickmeter.start()
    outs = detector.forward(outNames)
    tickmeter.stop()

    scores = outs[0]
    geometry = outs[1]
    [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

    print("Detected boxes:")
    for box in boxes:
        print(box)

    print("Confidences:")
    for confidence in confidences:
        print(confidence)

    boxes, confidences = merge_boxes(boxes, confidences, confThreshold)

    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
    indices = indices.flatten()
    lines = []
    for i in indices:
        vertices = cv.boxPoints(boxes[i])
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH

        # Debug: Draw the detected boxes
        cv.drawContours(frame, [np.int0(vertices)], 0, (0, 255, 0), 2)

        if modelRecognition:
            cropped = fourPointsTransform(frame, vertices)
            cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

            blob = cv.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
            recognizer.setInput(blob)

            tickmeter.start()
            result = recognizer.forward()
            tickmeter.stop()

            wordRecognized = decodeText(result)
            lines.append((vertices, wordRecognized))
            print(f"Recognized text: {wordRecognized}")

            cv.putText(frame, wordRecognized, (int(vertices[1][0]), int(vertices[1][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)

        # Debug: Draw box points
        for j in range(4):
            p1 = (int(vertices[j][0]), int(vertices[j][1]))
            p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
            cv.line(frame, p1, p2, (0, 0, 255), 1)

    # Debug print: Inspect the structure of lines
    print("Lines before sorting:")
    for line in lines:
        print(line)

    # Sort lines based on their y-coordinate
    lines = sorted(lines, key=lambda x: min([v[1] for v in x[0]]))

    print("Lines after sorting:")
    for line in lines:
        print(line)

    cv.imshow("Text Detection", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
