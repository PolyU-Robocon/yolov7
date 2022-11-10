import cv2
import time

if __name__ == '__main__':
    cv2.namedWindow("live", cv2.WINDOW_NORMAL) 
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print("STARTED CAM")
    
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        cv2.imshow('live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} fps: {round(1 / (time.time() - start), 5)}")
    print("END")
    cap.release()
    cv2.destroyAllWindows()