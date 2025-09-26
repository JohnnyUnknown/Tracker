import cv2
import time
import numpy as np

# Глобальные переменные для обработки мыши
clicked_point = None
mouse_clicked = False

# Обработчик клика мыши
def mouse_callback(event, x, y, flags, param):
    global clicked_point, mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        mouse_clicked = True


def create_tracker(tracker_type):
    if tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT.create()
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF.create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE.create()
    else:
        raise ValueError("Неизвестный тип трекера")

# Поиск БЛИЖАЙШЕГО контура к точке (не только внутри!)
def find_closest_contour(contours, point, frame_shape):
    if not contours:
        return None
    H, W = frame_shape[:2]
    frame_area = W * H
    max_allowed_area = frame_area * 0.8

    min_dist = float('inf')
    closest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_allowed_area:
            continue  # пропускаем огромные контуры

        dist = abs(cv2.pointPolygonTest(contour, point, True))
        if dist < min_dist:
            min_dist = dist
            closest_contour = contour

    if min_dist > 50:
        return None
    return closest_contour

# Получение bbox из контура
def get_bbox_from_contour(contour, frame_shape):
    if contour is None or len(contour) == 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    H, W = frame_shape[:2]

    if w < 20 or h < 20:
        return None
    if w * h > 0.8 * W * H:
        return None
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return None

    return (x, y, w, h)

# Инициализация трекера по клику мыши
def initialize_tracker_by_click(frame, tracker_type):
    global clicked_point, mouse_clicked, tracker

    if frame is None:
        return None, None, frame
    if frame.dtype != np.uint8:
        return None, None, frame

    frame_copy = frame.copy()
    H, W = frame.shape[:2]
    frame_area = W * H

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # ФИЛЬТР: только контуры от 100 до 80% кадра
    min_contour_area = 100
    max_contour_area = frame_area * 0.8
    contours = [
        c for c in contours
        if min_contour_area < cv2.contourArea(c) < max_contour_area
    ]

    cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 1)

    if clicked_point:
        # 🔍 Ищем ближайший контур, исключая гигантские
        target_contour = find_closest_contour(contours, clicked_point, frame.shape)
        if target_contour is not None:
            bbox = get_bbox_from_contour(target_contour, frame.shape)
            if bbox is None:
                print("❌ Контур некорректен или слишком большой/мал.")
                return None, None, frame_copy

            x, y, w, h = bbox

            # Визуализация
            cv2.drawContours(frame_copy, [target_contour], -1, (0, 0, 255), 3)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame_copy, clicked_point, 5, (255, 255, 0), -1)

            try:
                tracker = create_tracker(tracker_type)
                print(f"▶️ Инициализация с bbox={bbox} (размер: {w}x{h})")
                ok = tracker.init(frame, bbox)

                if not ok:
                    print("❌ tracker.init() вернул False. Попробуйте другой объект или трекер.")
                    return None, None, frame_copy
                return tracker, bbox, frame_copy
            except Exception as e:
                print(f"❌ Исключение: {e}")
                return None, None, frame_copy
        else:
            print("❌ Не найден подходящий контур рядом с кликом.")

    return None, None, frame_copy



# Основная программа
cap = cv2.VideoCapture("C:\\My\\Projects\\images\\djelga_video2.mp4")
if not cap.isOpened():
    exit()
ret, frame = cap.read()
if not ret:
    exit()

frame = cv2.resize(frame, (1024, 576))

tracker_name = "CSRT"
tracker, bbox = None, None

frame_count, elapsed_time, fps_process = 0, 0, 0
frames_for_averaging = 10

cv2.namedWindow(tracker_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(tracker_name, mouse_callback)

tracking_initialized = False

while True:
    start_time = time.time()
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1024, 576))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        tracking_initialized = False
        tracker, bbox = None, None
        print("Перезапуск: кликните по новому объекту")


    # Если трекер не инициализирован — ждём клика
    if not tracking_initialized:

        if mouse_clicked:
            tracker, bbox, contour_frame = initialize_tracker_by_click(frame, tracker_name)
            if tracker and bbox:
                tracking_initialized = True
                cv2.imshow(tracker_name, contour_frame)
                cv2.waitKey(1000)
            mouse_clicked = False  

    else:
        success, bbox = tracker.update(frame)

        if success and bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            if w > 0 and h > 0 and x >= 0 and y >= 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                success = False

        if not success:
            cv2.putText(frame, "Tracking failure", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


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

    if not tracking_initialized: time.sleep(0.1)
    cv2.imshow(tracker_name, frame)

cap.release()
cv2.destroyAllWindows()