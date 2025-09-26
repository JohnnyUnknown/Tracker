import cv2
import time


def create_tracker(tracker_type):
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT.create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF.create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE.create()
    else:
        raise ValueError("Неизвестный тип трекера")

def reinitialize_tracker(frame):
    cv2.destroyAllWindows()
    new_bbox = cv2.selectROI("New Frame", frame, False)
    cv2.destroyWindow("New Frame")
    new_tracker = create_tracker(tracker_name)
    new_tracker.init(frame, new_bbox)
    return new_tracker, new_bbox


cap = cv2.VideoCapture("C:\\My\\Projects\\images\\123.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (1024, 576))



# Выбираем ROI вручную
bbox = cv2.selectROI("Frame", frame, False)
cv2.destroyWindow("Frame")



tracker_name = "CSRT"
tracker = create_tracker(tracker_name)
tracker.init(frame, bbox)

frame_count, elapsed_time, fps_process = 0, 0, 0
frames_for_averaging = 10

cv2.namedWindow(tracker_name, cv2.WINDOW_NORMAL)

while True:
    start_time = time.time()
    frame_count += 1
    
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1024, 576))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        tracker, bbox = reinitialize_tracker(frame)
    if key == ord('q'):
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

    # Вычисление FPS
    time_diff = time.time() - start_time
    if time_diff > 0:
        elapsed_time += 1.0 / time_diff
    if frame_count == frames_for_averaging:
        fps_process = elapsed_time / frames_for_averaging
        frame_count, elapsed_time = 0, 0

    # Обновление заголовка окна
    window_title = f"{tracker_name} - FPS: {fps_process:.2f}"
    cv2.setWindowTitle(tracker_name, window_title)
    
    cv2.imshow(tracker_name, frame)


cap.release()
cv2.destroyAllWindows()
