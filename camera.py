import cv2

class VideoCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.is_streaming = True

    def get_frame(self):
        if self.is_streaming:
            ret, frame = self.camera.read()
            if not ret:
                return None
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return None

    def get_raw_frame(self):
        if self.is_streaming:
            ret, frame = self.camera.read()
            if not ret:
                return None
            return frame
        return None

    def stop(self):
        self.is_streaming = False
        self.camera.release()

    def start(self):
        self.is_streaming = True
        self.camera = cv2.VideoCapture(0)  # Reinitialize the camera
