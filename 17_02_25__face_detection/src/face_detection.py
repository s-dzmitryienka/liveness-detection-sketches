import cv2
import os


def highlight_face(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame_opencv_dnn, face_boxes


dirname = os.path.dirname(__file__)
face_proto = os.path.join(dirname, "opencv_face_detector.pbtxt")
face_model = os.path.join(dirname, "opencv_face_detector_uint8.pb")

face_net = cv2.dnn.readNet(face_model, face_proto)

video = cv2.VideoCapture(0)
while cv2.waitKey(1) < 0:
    has_frame, frame = video.read()
    if not has_frame:
        cv2.waitKey()
        break

    result_image, face = highlight_face(face_net, frame)
    if not face:
        print('Face not found')
    cv2.imshow("Face detection", result_image)
