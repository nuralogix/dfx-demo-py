#
#              Copyright (c) 2022, Nuralogix Corp.
#                      All Rights reserved
#
#      THIS SOFTWARE IS LICENSED BY AND IS THE CONFIDENTIAL AND
#      PROPRIETARY PROPERTY OF NURALOGIX CORP. IT IS
#      PROTECTED UNDER THE COPYRIGHT LAWS OF THE USA, CANADA
#      AND OTHER FOREIGN COUNTRIES. THIS SOFTWARE OR ANY
#      PART THEREOF, SHALL NOT, WITHOUT THE PRIOR WRITTEN CONSENT
#      OF NURALOGIX CORP, BE USED, COPIED, DISCLOSED,
#      DECOMPILED, DISASSEMBLED, MODIFIED OR OTHERWISE TRANSFERRED
#      EXCEPT IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF A
#      NURALOGIX CORP SOFTWARE LICENSE AGREEMENT.
#

import multiprocessing as mp
import queue

import cv2
import mediapipe
import numpy as np


class MediaPipeTracker():

    def __init__(self, max_faces, track_in_background=False) -> None:
        self._mediapipe_initialized = False
        self._track_in_background = track_in_background
        self._init_params = max_faces, True, 0.5, 0.5
        self._last_tracked_faces = {}
        if self._track_in_background:
            self._work_queue = mp.Queue(1)
            self._results_queue = mp.Queue(1)
            self._track_proc = mp.Process(target=self._trackFacesThreaded, name="mediapipe_tracker")
            self._track_proc.start()
        else:
            max_faces, refine_landmarks, min_det_conf, min_track_conf = self._init_params
            self._initializeMediaPipe(max_faces, refine_landmarks, min_det_conf, min_track_conf)

    def _initializeMediaPipe(self, max_faces, refine_landmarks, min_det_conf, min_track_conf):
        self._face_mesh = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=max_faces,
                                                                 refine_landmarks=refine_landmarks,
                                                                 min_detection_confidence=min_det_conf,
                                                                 min_tracking_confidence=min_track_conf)
        self._mediapipe_initialized = True

    @staticmethod
    def __version__():
        return mediapipe.__version__

    @property
    def pointsPerFace(self):
        return 130

    def _trackFacesThreaded(self):
        maxFaces, refine_landmarks, min_det_conf, min_track_conf = self._init_params
        self._initializeMediaPipe(maxFaces, refine_landmarks, min_det_conf, min_track_conf)

        while True:
            image, frameNumber, timeStamp_ms = self._work_queue.get()
            if image is None:
                break
            tracked_faces = self._trackFaces(image, frameNumber, timeStamp_ms)
            try:
                self._results_queue.put_nowait(tracked_faces)
            except queue.Full:
                pass

    def trackFaces(self, image, frameNumber, timeStamp_ms):
        if self._track_in_background:
            try:
                self._work_queue.put_nowait((np.copy(image), frameNumber, timeStamp_ms))
            except queue.Full:
                pass

            try:
                self._last_tracked_faces = self._results_queue.get_nowait()
            except queue.Empty:
                pass

            # Force face detection if the result was empty if using the "smart" strategy
            if not self._last_tracked_faces:
                self._last_tracked_faces = self._trackFaces(image, frameNumber, timeStamp_ms)
        else:
            self._last_tracked_faces = self._trackFaces(image, frameNumber, timeStamp_ms)

        return self._last_tracked_faces

    def _trackFaces(self, bgrImage, _frameNumber, _timeStamp_ms):
        faces = {}

        if not self._mediapipe_initialized:
            return faces

        # Convert to rgb
        image = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self._face_mesh.process(image)

        if not results.multi_face_landmarks:
            return faces

        for faceNumber, face in enumerate(results.multi_face_landmarks):
            tlx, tly, brx, bry = self._findBoundingBox(image.shape, face.landmark, 0.05)
            faces[str(faceNumber + 1)] = {
                "id": str(faceNumber + 1),
                "detected": True,
                "poseValid": True,
                "rect.x": tlx,
                "rect.y": tly,
                "rect.w": brx - tlx,
                "rect.h": bry - tly,
            }
            faces[str(faceNumber + 1)].update({
                "points": {
                    # Outline
                    "13.1": self._pipe2mpeg4(image.shape, face.landmark, 356, 368),
                    "13.3": self._pipe2mpeg4(image.shape, face.landmark, 454, 447),
                    "13.5": self._pipe2mpeg4(image.shape, face.landmark, 366, 352),
                    "13.7": self._pipe2mpeg4(image.shape, face.landmark, 435, 288),
                    "13.9": self._pipe2mpeg4(image.shape, face.landmark, 397, 367),
                    "13.11": self._pipe2mpeg4(image.shape, face.landmark, 394, 365),
                    "13.13": self._pipe2mpeg4(image.shape, face.landmark, 378),
                    "13.15": self._pipe2mpeg4(image.shape, face.landmark, 377),
                    "13.17": self._pipe2mpeg4(image.shape, face.landmark, 152),
                    "13.16": self._pipe2mpeg4(image.shape, face.landmark, 148),
                    "13.14": self._pipe2mpeg4(image.shape, face.landmark, 149),
                    "13.12": self._pipe2mpeg4(image.shape, face.landmark, 136, 150),
                    "13.10": self._pipe2mpeg4(image.shape, face.landmark, 172),
                    "13.8": self._pipe2mpeg4(image.shape, face.landmark, 215, 58),
                    "13.6": self._pipe2mpeg4(image.shape, face.landmark, 137, 93, 132),
                    "13.4": self._pipe2mpeg4(image.shape, face.landmark, 234),
                    "13.2": self._pipe2mpeg4(image.shape, face.landmark, 162, 127),

                    # Eyebrow
                    "4.6": self._pipe2mpeg4(image.shape, face.landmark, 70),
                    "14.4": self._pipe2mpeg4(image.shape, face.landmark, 63),
                    "4.4": self._pipe2mpeg4(image.shape, face.landmark, 105),
                    "14.2": self._pipe2mpeg4(image.shape, face.landmark, 66),
                    "4.2": self._pipe2mpeg4(image.shape, face.landmark, 107),

                    # Eyebrow
                    "4.1": self._pipe2mpeg4(image.shape, face.landmark, 336),
                    "14.1": self._pipe2mpeg4(image.shape, face.landmark, 296),
                    "4.3": self._pipe2mpeg4(image.shape, face.landmark, 334),
                    "14.3": self._pipe2mpeg4(image.shape, face.landmark, 293),
                    "4.5": self._pipe2mpeg4(image.shape, face.landmark, 300),

                    # Nose
                    "12.1": self._pipe2mpeg4(image.shape, face.landmark, 196, 197),
                    "9.4": self._pipe2mpeg4(image.shape, face.landmark, 240, 75),
                    "9.2": self._pipe2mpeg4(image.shape, face.landmark, 64, 219),
                    "9.3": self._pipe2mpeg4(image.shape, face.landmark, 4),
                    "9.12": self._pipe2mpeg4(image.shape, face.landmark, 195, 197),
                    "9.1": self._pipe2mpeg4(image.shape, face.landmark, 294),
                    "9.5": self._pipe2mpeg4(image.shape, face.landmark, 290),
                    "9.15": self._pipe2mpeg4(image.shape, face.landmark, 2, 370),

                    # Eye
                    "3.12": self._pipe2mpeg4(image.shape, face.landmark, 130),
                    "12.10": self._pipe2mpeg4(image.shape, face.landmark, 160, 161),
                    "3.2": self._pipe2mpeg4(image.shape, face.landmark, 159),
                    "12.6": self._pipe2mpeg4(image.shape, face.landmark, 469),
                    "3.8": self._pipe2mpeg4(image.shape, face.landmark, 173),
                    "12.8": self._pipe2mpeg4(image.shape, face.landmark, 153),
                    "3.4": self._pipe2mpeg4(image.shape, face.landmark, 472, 144),
                    "12.12": self._pipe2mpeg4(image.shape, face.landmark, 163),

                    # Eye
                    "3.11": self._pipe2mpeg4(image.shape, face.landmark, 398),
                    "12.9": self._pipe2mpeg4(image.shape, face.landmark, 476, 384),
                    "3.1": self._pipe2mpeg4(image.shape, face.landmark, 386),
                    "12.5": self._pipe2mpeg4(image.shape, face.landmark, 387),
                    "3.7": self._pipe2mpeg4(image.shape, face.landmark, 263),
                    "12.7": self._pipe2mpeg4(image.shape, face.landmark, 390),
                    "3.3": self._pipe2mpeg4(image.shape, face.landmark, 374),
                    "12.11": self._pipe2mpeg4(image.shape, face.landmark, 380),

                    # Lips outer
                    "8.4": self._pipe2mpeg4(image.shape, face.landmark, 202, 61),
                    "8.6": self._pipe2mpeg4(image.shape, face.landmark, 73, 39),
                    "8.9": self._pipe2mpeg4(image.shape, face.landmark, 72, 37),
                    "8.1": self._pipe2mpeg4(image.shape, face.landmark, 11),
                    "8.10": self._pipe2mpeg4(image.shape, face.landmark, 302, 267),
                    "8.5": self._pipe2mpeg4(image.shape, face.landmark, 269, 303),
                    "8.3": self._pipe2mpeg4(image.shape, face.landmark, 273, 422, 291),
                    "8.7": self._pipe2mpeg4(image.shape, face.landmark, 405, 418),
                    "8.2": self._pipe2mpeg4(image.shape, face.landmark, 17, 200),
                    "8.8": self._pipe2mpeg4(image.shape, face.landmark, 194, 181),

                    # Lips inner
                    "2.5": self._pipe2mpeg4(image.shape, face.landmark, 61, 202),
                    "2.7": self._pipe2mpeg4(image.shape, face.landmark, 179),
                    "2.2": self._pipe2mpeg4(image.shape, face.landmark, 15, 13, 14),
                    "2.6": self._pipe2mpeg4(image.shape, face.landmark, 403, 402),
                    "2.4": self._pipe2mpeg4(image.shape, face.landmark, 291, 273),
                    "2.8": self._pipe2mpeg4(image.shape, face.landmark, 403, 311),
                    "2.3": self._pipe2mpeg4(image.shape, face.landmark, 15, 13, 14),
                    "2.9": self._pipe2mpeg4(image.shape, face.landmark, 179, 81, 178),

                    # Forehead
                    "11.2": self._pipe2mpeg4(image.shape, face.landmark, 67),
                    "11.1": self._pipe2mpeg4(image.shape, face.landmark, 10),
                    "11.3": self._pipe2mpeg4(image.shape, face.landmark, 297),

                    # Other
                    "2.1": self._pipe2mpeg4(image.shape, face.landmark, 152),
                    "2.10": self._pipe2mpeg4(image.shape, face.landmark, 152, 124, 117),
                    "2.11": self._pipe2mpeg4(image.shape, face.landmark, 378),
                    "2.12": self._pipe2mpeg4(image.shape, face.landmark, 149),
                    "2.13": self._pipe2mpeg4(image.shape, face.landmark, 397),
                    "2.14": self._pipe2mpeg4(image.shape, face.landmark, 172),
                    "3.5": self._pipe2mpeg4(image.shape, face.landmark, 473),
                    "3.6": self._pipe2mpeg4(image.shape, face.landmark, 468),
                    "3.9": self._pipe2mpeg4(image.shape, face.landmark, 253),
                    "3.10": self._pipe2mpeg4(image.shape, face.landmark, 23, 24),
                    "3.13": self._pipe2mpeg4(image.shape, face.landmark, 257, 475),
                    "3.14": self._pipe2mpeg4(image.shape, face.landmark, 26, 470),
                    "5.1": self._pipe2mpeg4(image.shape, face.landmark, 185, 156),
                    "5.2": self._pipe2mpeg4(image.shape, face.landmark, 213, 187, 192),
                    "5.3": self._pipe2mpeg4(image.shape, face.landmark, 280, 266, 330),
                    "5.4": self._pipe2mpeg4(image.shape, face.landmark, 50, 205, 101),
                    "9.6": self._pipe2mpeg4(image.shape, face.landmark, 188),
                    "9.7": self._pipe2mpeg4(image.shape, face.landmark, 317),
                    "9.13": self._pipe2mpeg4(image.shape, face.landmark, 437, 466),
                    "9.14": self._pipe2mpeg4(image.shape, face.landmark, 217, 236),
                    "14.5": self._pipe2mpeg4(image.shape, face.landmark, 409, 270),
                    "14.6": self._pipe2mpeg4(image.shape, face.landmark, 185),
                    "14.7": self._pipe2mpeg4(image.shape, face.landmark, 321),
                    "14.8": self._pipe2mpeg4(image.shape, face.landmark, 91),
                    "14.9": self._pipe2mpeg4(image.shape, face.landmark, 73),
                    "14.10": self._pipe2mpeg4(image.shape, face.landmark, 303, 302),
                    "14.11": self._pipe2mpeg4(image.shape, face.landmark, 314, 421),
                    "14.12": self._pipe2mpeg4(image.shape, face.landmark, 201, 84),
                    "14.13": self._pipe2mpeg4(image.shape, face.landmark, 89),
                    "14.14": self._pipe2mpeg4(image.shape, face.landmark, 319),
                    "14.15": self._pipe2mpeg4(image.shape, face.landmark, 89, 88),
                    "14.16": self._pipe2mpeg4(image.shape, face.landmark, 319, 310),
                    "14.17": self._pipe2mpeg4(image.shape, face.landmark, 86),
                    "14.18": self._pipe2mpeg4(image.shape, face.landmark, 316, 312),
                    "14.19": self._pipe2mpeg4(image.shape, face.landmark, 86),
                    "14.20": self._pipe2mpeg4(image.shape, face.landmark, 316),
                    "14.21": self._pipe2mpeg4(image.shape, face.landmark, 4),
                    "14.22": self._pipe2mpeg4(image.shape, face.landmark, 5),
                    "14.23": self._pipe2mpeg4(image.shape, face.landmark, 195),
                    "14.24": self._pipe2mpeg4(image.shape, face.landmark, 357),
                    "14.25": self._pipe2mpeg4(image.shape, face.landmark, 375),
                    "15.1": self._pipe2mpeg4(image.shape, face.landmark, 356, 368),
                    "15.2": self._pipe2mpeg4(image.shape, face.landmark, 296),
                    "15.3": self._pipe2mpeg4(image.shape, face.landmark, 454),
                    "15.4": self._pipe2mpeg4(image.shape, face.landmark, 234),
                    "15.5": self._pipe2mpeg4(image.shape, face.landmark, 352, 366),
                    "15.6": self._pipe2mpeg4(image.shape, face.landmark, 137),
                    "15.7": self._pipe2mpeg4(image.shape, face.landmark, 288),
                    "15.8": self._pipe2mpeg4(image.shape, face.landmark, 215),
                    "15.9": self._pipe2mpeg4(image.shape, face.landmark, 397),
                    "15.10": self._pipe2mpeg4(image.shape, face.landmark, 172),
                    "15.11": self._pipe2mpeg4(image.shape, face.landmark, 394, 365),
                    "15.12": self._pipe2mpeg4(image.shape, face.landmark, 136),
                    "15.13": self._pipe2mpeg4(image.shape, face.landmark, 378),
                    "15.14": self._pipe2mpeg4(image.shape, face.landmark, 149),
                    "15.15": self._pipe2mpeg4(image.shape, face.landmark, 377),
                    "15.16": self._pipe2mpeg4(image.shape, face.landmark, 148),
                    "15.17": self._pipe2mpeg4(image.shape, face.landmark, 152),
                }
            })

        return faces

    def stop(self):
        if self._track_in_background and self._track_proc and self._track_proc.is_alive():
            self._work_queue.put((None, None, None))
            self._track_proc.terminate()
            self._work_queue.cancel_join_thread()
            self._results_queue.cancel_join_thread()

    def _findBoundingBox(self, imageShape, landmarks, expand):
        tlx, tly, brx, bry = (imageShape[1], imageShape[0], 0, 0)
        for landmark in landmarks:
            scaledX, scaledY = int(landmark.x * imageShape[1]), int(landmark.y * imageShape[0])
            tlx = scaledX if scaledX < tlx else tlx  # Create the bounding rect
            tly = scaledY if scaledY < tly else tly
            brx = scaledX if scaledX > brx else brx
            bry = scaledY if scaledY > bry else bry

        # Expand bounding box
        dw = (brx - tlx) * expand
        dh = (bry - tly) * expand
        tlx, tly, brx, bry = tlx - dw, tly - 2 * dh, brx + dw, bry

        # Ensure bounding box within image
        tlx, tly, brx, bry = max(tlx, 0), max(tly, 0), min(brx, imageShape[1]), min(bry, imageShape[0])

        return tlx, tly, brx, bry

    def _pipe2mpeg4(self, imageShape, landmarks, *indices):
        x, y, z = 0, 0, 0
        for i in indices:
            x = x + landmarks[i].x * imageShape[1]
            y = y + landmarks[i].y * imageShape[0]
            z = z + landmarks[i].z
        L = len(indices)

        return {
            "x": x / L,
            "y": y / L,
            "z": z / L,
            "valid": True,
            "estimated": True,
            "quality": 1.0,
        }
