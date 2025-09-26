import cv2 as cv
from time import sleep

class camera_source:
    camera_sources = ["picam", "siyi", "video"]

    def __init__(self, camera_name=None, video_path=None):
        self.camera_source = camera_name if camera_name in self.camera_sources else None
        self.video_path = video_path
        self.camera_object = self.camera_init()
    

    def camera_init(self):
        VIDEO_SIZE = (1920, 1080)
        camera = None

        if not self.camera_source or self.camera_source == "video":
            if self.video_path:
                camera = cv.VideoCapture(self.video_path)
            else:
                print("Укажите путь к видеофайлу.")

        elif self.camera_source == "picam":
            try:
                from libcamera import controls
                from picamera import Picamera2
                camera = Picamera2()
                video_config = camera.create_video_configuration(
                    main={"size": VIDEO_SIZE, "format": 'RGB888'},
                    controls={
                    "FrameRate": 30, 
                    "AfMode": controls.AfModeEnum.Manual, 
                    "LensPosition": 0.0, 
                    # "ExposureTime": 10000,
                    # "AeEnable": False, 
                    # "ScalerCrop": (0, 0, 128, 128),
                    }
                )
                camera.configure(video_config)
                camera.start()
            except ModuleNotFoundError:
                print("Установить соединение с камерой Raspbery не удалось.")

        elif self.camera_source == "siyi":
            pipeline = (
                'rtspsrc location="rtsp://192.168.144.25:8554/main.264" latency=1 '
                '! rtpjitterbuffer mode=1 do-lost=true latency=50 drop-on-latency=false '
                '! rtph264depay '
                '! h264parse '
                '! avdec_h264 '
                '! queue max-size-buffers=0 max-size-bytes=0 max-size-time=100000000 '
                '! videoconvert n-threads=2 '
                '! video/x-raw,format=BGR '  
                '! appsink sync=True'                  
                )
            camera = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
            count_fails = 0
            while not camera.isOpened():
                print(f"Fail {count_fails + 1}. Try open stream again.")
                camera = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
                count_fails += 1
                if count_fails >= 1:
                    print("Установить соединение с камерой SIYI не удалось.")
                    exit()
                sleep(10)

        return camera
    

    def get_frame(self):
        try:
            if self.camera_source == "picam":
                return self.camera_object.capture_array()
            else:
                _, frame = self.camera_object.read()
                return frame
        except TypeError:
            print("Выберите тип камеры ('picam', 'siyi') или 'video' и путь к видеофайлу.")
        except AttributeError:
            print("Выберите тип камеры ('picam', 'siyi') или 'video' и путь к видеофайлу.")
        