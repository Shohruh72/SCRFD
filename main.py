import argparse
import cv2

from nets.nn import FaceDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', default=-1, help='image file path')
    parser.add_argument('--model', default='weights/model_1.onnx', help='model file path')
    args = parser.parse_args()

    detector = FaceDetector(onnx_file=args.model)
    stream = cv2.VideoCapture(args.cam_id)

    if not stream.isOpened():
        print("Error opening video stream or file")

    while True:
        success, frame = stream.read()

        if success:
            boxes, _ = detector.detect(frame, input_size=(640, 640))
            boxes = boxes.astype('int32')
            for box in boxes:
                x_min, y_min, x_max, y_max, _ = box
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 1)

                cv2.line(frame, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), (255, 0, 255), 3)
                cv2.line(frame, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), (255, 0, 255), 3)

                cv2.line(frame, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), (255, 0, 255), 3)
                cv2.line(frame, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), (255, 0, 255), 3)

                cv2.line(frame, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), (255, 0, 255), 3)
                cv2.line(frame, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), (255, 0, 255), 3)

                cv2.line(frame, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), (255, 0, 255), 3)
                cv2.line(frame, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), (255, 0, 255), 3)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


