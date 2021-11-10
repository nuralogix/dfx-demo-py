import multiprocessing as mp
import os
import queue
from statistics import median

import dlib
import numpy as np


class DlibTracker():
    def __init__(self, face_detect_strategy=None):
        try:
            self._detect_proc = None
            model_path = os.path.join(os.getcwd(), "res", "shape_predictor_68_face_landmarks.dat")

            self._face_detector = dlib.get_frontal_face_detector()
            self._pose_estimator = dlib.shape_predictor(model_path)
            self._smoothed = {pt: ([], []) for pt in DlibTracker._dlib2mpeg4}
            if face_detect_strategy is None:
                face_detect_strategy = "smart"
            self._fd_fast = face_detect_strategy != "brute"
            self._fd_smart = face_detect_strategy == "smart"
            if self._fd_fast:
                self._last_detected_faces = []
                self._work_queue = mp.Queue(1)
                self._results_queue = mp.Queue(1)
                self._detect_proc = mp.Process(target=self._detectFacesThreaded, name="dlib_tracker")
                self._detect_proc.start()
        except RuntimeError:
            print("Please download and unzip "
                  "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 into 'res' folder")
            raise

    @staticmethod
    def __version__():
        return dlib.__version__

    def __del__(self):
        self.stop()

    def _detectFacesThreaded(self):
        while True:
            image = self._work_queue.get()
            if image is None:
                break
            detected_faces = self._face_detector(image, 0)
            try:
                self._results_queue.put_nowait(detected_faces)
            except queue.Full:
                pass

    def trackFaces(self, image, frameNumber, timeStamp_ms, searchRect=None, desiredAttributes=None):
        x, y, w, h = self._sanitizeRoi(image.shape, searchRect)
        searchImage = image[y:y + h, x:x + w]

        if self._fd_fast:
            try:
                self._work_queue.put_nowait(np.copy(searchImage))
            except queue.Full:
                pass

            try:
                self._last_detected_faces = self._results_queue.get_nowait()
            except queue.Empty:
                pass

            # Force face detection if the result was empty if using the "smart" strategy
            if self._fd_smart and not self._last_detected_faces:
                self._last_detected_faces = self._face_detector(searchImage, 0)
        else:
            self._last_detected_faces = self._face_detector(searchImage, 0)

        face_rects = self._last_detected_faces
        faces = {}
        for i, rect in enumerate(face_rects):
            face_points = self._pose_estimator(searchImage, rect)
            points = {}
            for j in range(face_points.num_parts):
                # The 0 check is here is because Dlib will happily give you negative coordinates
                # which the SDK obviously cannot handle
                newx = face_points.part(j).x + x if face_points.part(j).x + x > 0 else 0
                newy = face_points.part(j).y + y if face_points.part(j).y + y > 0 else 0
                pointname = DlibTracker._dlib2mpeg4[j]
                smoothedx, smoothedy = self._smoothPoints(newx, newy, pointname)
                points[pointname] = {"x": smoothedx, "y": smoothedy, "valid": True, "estimated": True, "quality": 1.0}
            faces[str(i)] = {
                "id": str(i + 1),
                "rect.x": rect.left() + x if rect.left() + x > 0 else 0,
                "rect.y": rect.top() + y if rect.top() + y > 0 else 0,
                "rect.w": rect.width(),
                "rect.h": rect.height(),
                "detected": True,
                "poseValid": True if len(points) > 0 else False,
                "points": points
            }

        return faces

    def stop(self):
        if self._detect_proc and self._detect_proc.is_alive():
            self._work_queue.put(None)
            self._detect_proc.terminate()
            self._work_queue.cancel_join_thread()
            self._results_queue.cancel_join_thread()

    def _smoothPoints(self, newx, newy, pointname, framesToSmooth=10):
        xs, ys = self._smoothed[pointname]
        xs.append(newx)
        ys.append(newy)
        if len(xs) > framesToSmooth:
            self._smoothed[pointname] = xs[1:], ys[1:]
        smoothedx = median(xs)
        smoothedy = median(ys)

        return smoothedx, smoothedy

    # TODO check this
    def _sanitizeRoi(self, shape, rect=None):
        if rect is None:
            return (0, 0, shape[1] - 1, shape[0] - 1)
        else:
            x, y, w, h = rect
            left = x if x > 0 else 0
            top = y if y > 0 else 0
            width = w if w > 0 and w < shape[1] else shape[1]
            height = h if h > 0 and h < shape[0] else shape[0]
            return (left, top, width, height)

    # This is a class attribute
    _dlib2mpeg4 = [
        "13.2",  # DLIB: 0
        "13.4",  # DLIB: 1
        "13.6",  # DLIB: 2
        "13.8",  # DLIB: 3
        "13.10",  # DLIB: 4
        "13.12",  # DLIB: 5
        "13.14",  # DLIB: 6
        "13.16",  # DLIB: 7
        "13.17",  # DLIB: 8
        "13.15",  # DLIB: 9
        "13.13",  # DLIB: 10
        "13.11",  # DLIB: 11
        "13.9",  # DLIB: 12
        "13.7",  # DLIB: 13
        "13.5",  # DLIB: 14
        "13.3",  # DLIB: 15
        "13.1",  # DLIB: 16
        # left eye brow
        "4.6",  # DLIB: 17
        "14.4",  # DLIB: 18
        "4.4",  # DLIB: 19
        "14.2",  # DLIB: 20
        "4.2",  # DLIB: 21
        # right eye brow
        "4.1",  # DLIB: 22
        "14.1",  # DLIB: 23
        "4.3",  # DLIB: 24
        "14.3",  # DLIB: 25
        "4.5",  # DLIB: 26
        # nose bridge
        "12.1",  # DLIB: 27
        "9.12",  # DLIB: 28    -- This is a point that does not exist in Visage
        "9.12",  # DLIB: 29
        "9.3",  # DLIB: 30
        # lower nose
        "9.2",  # DLIB: 31
        "9.4",  # DLIB: 32
        "9.15",  # DLIB: 33
        "9.5",  # DLIB: 34
        "9.1",  # DLIB: 35
        # right eye relative to the user
        "3.12",  # DLIB: 36
        "12.10",  # DLIB: 37
        "12.6",  # DLIB: 38
        "3.8",  # DLIB: 39
        "12.8",  # DLIB: 40
        "12.12",  # DLIB: 41
        # left eye relative to the user
        "3.11",  # DLIB: 42
        "12.9",  # DLIB: 43
        "12.5",  # DLIB: 44
        "3.7",  # DLIB: 45
        "12.7",  # DLIB: 46
        "12.11",  # DLIB: 47
        # mouth
        "8.4",  # DLIB: 48
        "8.6",  # DLIB: 49
        "8.9",  # DLIB: 50
        "8.1",  # DLIB: 51
        "8.10",  # DLIB: 52
        "8.5",  # DLIB: 53
        "8.3",  # DLIB: 54
        "8.7",  # DLIB: 55   -- This is a point that does not exist in Visage, consider 8.8
        "8.7",  # DLIB: 56
        "8.2",  # DLIB: 57
        "8.8",  # DLIB: 58   -- This is a point that does not exist in Visage, consider 8.7
        "8.8",  # DLIB: 59
        # mouth region
        "2.5",  # DLIB: 60
        "2.7",  # DLIB: 61
        "2.2",  # DLIB: 62
        "2.6",  # DLIB: 63
        "2.4",  # DLIB: 64
        "2.8",  # DLIB: 65
        "2.3",  # DLIB: 66
        "2.9"  # DLIB: 67
    ]
