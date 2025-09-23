import cv2
import time

# Выбери тип трекера
def create_tracker(tracker_type):
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT.create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF.create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE.create()
    else:
        raise ValueError("Неизвестный тип трекера")


cap = cv2.VideoCapture("C:\\My\\Projects\\images\\move6.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (1024, 576))

# Выбираем ROI вручную
bbox = cv2.selectROI("Frame", frame, False)
cv2.destroyWindow("Frame")


tracker_name = "CSRT"
tracker = create_tracker(tracker_name)
tracker.init(frame, bbox)

frame_count = 0
start_time = time.time()

cv2.namedWindow(tracker_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1024, 576))
    frame_count += 1

    # Обновляем трекер
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

    # Вычисляем FPS
    elapsed_time = time.time() - start_time
    fps_process = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Обновляем заголовок окна без создания нового
    window_title = f"{tracker_name} - FPS: {fps_process:.2f}"
    cv2.setWindowTitle(tracker_name, window_title)

    cv2.imshow(tracker_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
