import cv2
import time
import numpy as np
from camera_object import camera_source
from tracker_object import tracker_maker


class main_process:
    camera_sources = ["picam", "siyi", "video"]
    detection = ["auto", "manual"]
    detect_methods = []
    tracker_names = ["CSRT", "KCF", "MOSSE"]

    def __init__(self, contour_detection="manual", tracker_name="CSRT", camera_name=None, video_path=None, detector=None):
        self.contour_detect = contour_detection if contour_detection in self.detection else "manual"
        self.tracker_name = tracker_name if tracker_name in self.tracker_names else "CSRT"
        self.camera_source = camera_name if camera_name in self.camera_sources else None
        self.video_path = video_path
        # Инициализация камеры
        self.camera = camera_source(camera_name=self.camera_source, video_path=self.video_path)

        if self.contour_detect == "manual":
            self.clicked_point = None
            self.mouse_clicked = False
        # if self.contour_detect == "auto":
        #     self.detector = detector


    # Обработчик клика мыши
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            self.mouse_clicked = True
        

    # Главный процесс
    def video_flow(self):
        frame = self.camera.get_frame()
        frame = cv2.resize(frame, (1024, 576))

        frame_count, elapsed_time, fps_process = 0, 0, 0
        frames_for_averaging = 10
        tracking_initialized = False

        cv2.namedWindow(self.tracker_name, cv2.WINDOW_NORMAL)
        if self.contour_detect == "manual":
            cv2.setMouseCallback(self.tracker_name, self.mouse_callback)

        while True:
            start_time = time.time()
            frame_count += 1

            frame = self.camera.get_frame()
            if type(frame) != np.ndarray:
                break
            frame = cv2.resize(frame, (1024, 576))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                tracking_initialized = False
                tracker, bbox = None, None
                print("Перезапуск: кликните по новому объекту")

            #---------------------------------------------------------------------------------------

            # Если трекер не инициализирован — ждём клика
            if not tracking_initialized:
                if self.contour_detect == "manual" and self.mouse_clicked:
                    tracker_obj = tracker_maker(self.tracker_name, frame, self.clicked_point)
                    tracker, bbox, contour_frame = tracker_obj.initialize_tracker_by_click()
                    if tracker and bbox:
                        tracking_initialized = True
                        # Отображение найденных контуров
                        cv2.imshow(self.tracker_name, contour_frame)
                        cv2.waitKey(1000)
                    self.mouse_clicked = False  
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

            #---------------------------------------------------------------------------------------

            # Вычисление FPS
            time_diff = time.time() - start_time
            if time_diff > 0:
                elapsed_time += 1.0 / time_diff
            if frame_count == frames_for_averaging:
                fps_process = elapsed_time / frames_for_averaging
                frame_count, elapsed_time = 0, 0

            # Обновление заголовка окна
            window_title = f"{self.tracker_name} - FPS: {fps_process:.2f}"
            cv2.setWindowTitle(self.tracker_name, window_title)

            if not tracking_initialized: time.sleep(0.01)   # Замедление отображения видео
            cv2.imshow(self.tracker_name, frame)

        cv2.destroyAllWindows()



path = "C:\\My\\Projects\\images\\djelga_video2.mp4"
process = main_process(contour_detection="manual", tracker_name="CSRT", camera_name="video", video_path=path)
frame = process.video_flow()
