import cv2 as cv
import numpy as np


class tracker_maker:
    def __init__(self, tracker_type, frame, point):
        self.tracker_type = tracker_type
        self.frame = frame
        self.clicked_point = point
    
    def create_tracker(self, tracker_type):
        if tracker_type == 'CSRT':
            return cv.legacy.TrackerCSRT.create()
        elif tracker_type == 'KCF':
            return cv.legacy.TrackerKCF.create()
        elif tracker_type == 'MOSSE':
            return cv.legacy.TrackerMOSSE.create()
        else:
            raise ValueError("Неизвестный тип трекера")
        
    # Поиск БЛИЖАЙШЕГО контура к точке (не только внутри!)
    def find_closest_contour(self, contours, point, frame_shape):
        if not contours:
            return None
        H, W = frame_shape[:2]
        frame_area = W * H
        max_allowed_area = frame_area * 0.8

        min_dist = float('inf')
        closest_contour = None

        for contour in contours:
            area = cv.contourArea(contour)
            if area > max_allowed_area:
                continue  # пропускаем огромные контуры
            dist = abs(cv.pointPolygonTest(contour, point, True))
            if dist < min_dist:
                min_dist = dist
                closest_contour = contour
        if min_dist > 50:
            return None
        return closest_contour

    # Получение bbox из контура
    def get_bbox_from_contour(self, contour, frame_shape):
        if contour is None or len(contour) == 0:
            return None
        x, y, w, h = cv.boundingRect(contour)
        H, W = frame_shape[:2]

        if w < 20 or h < 20:
            return None
        if w * h > 0.8 * W * H:
            return None
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return None

        return (x-5, y-5, w+10, h+10)

    # Инициализация трекера по клику мыши
    def initialize_tracker_by_click(self):
        if self.frame is None:
            return None, None, self.frame
        if self.frame.dtype != np.uint8:
            return None, None, self.frame

        frame_copy = self.frame.copy()
        H, W = self.frame.shape[:2]
        frame_area = W * H

        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (9, 9), 0)
        edges = cv.Canny(blurred, 50, 150)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # ФИЛЬТР: только контуры от 100 до 80% кадра
        min_contour_area = 100
        max_contour_area = frame_area * 0.8
        contours = [
            c for c in contours
            if min_contour_area < cv.contourArea(c) < max_contour_area
        ]

        cv.drawContours(frame_copy, contours, -1, (0, 255, 0), 1)

        if self.clicked_point:
            # 🔍 Ищем ближайший контур, исключая гигантские
            target_contour = self.find_closest_contour(contours, self.clicked_point, self.frame.shape)
            if target_contour is not None:
                bbox = self.get_bbox_from_contour(target_contour, self.frame.shape)
                if bbox is None:
                    print("❌ Контур некорректен или слишком большой/мал.")
                    return None, None, frame_copy

                x, y, w, h = bbox

                # Визуализация
                cv.drawContours(frame_copy, [target_contour], -1, (0, 0, 255), 3)
                cv.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.circle(frame_copy, self.clicked_point, 5, (255, 255, 0), -1)

                try:
                    tracker = self.create_tracker(self.tracker_type)
                    print(f"▶️ Инициализация с bbox={bbox} (размер: {w}x{h})")
                    ok = tracker.init(self.frame, bbox)

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
    
