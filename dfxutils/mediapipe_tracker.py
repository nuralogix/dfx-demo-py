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
                    "13.1": self._pipe2mpeg4(image.shape, face.landmark, 356),
                    "13.3": self._pipe2mpeg4(image.shape, face.landmark, 447),
                    "13.5": self._pipe2mpeg4(image.shape, face.landmark, 361),
                    "13.7": self._pipe2mpeg4(image.shape, face.landmark, 288),
                    "13.9": self._pipe2mpeg4(image.shape, face.landmark, 397),
                    "13.11": self._pipe2mpeg4(image.shape, face.landmark, 394),
                    "13.13": self._pipe2mpeg4(image.shape, face.landmark, 378),
                    "13.15": self._pipe2mpeg4(image.shape, face.landmark, 377),
                    "13.17": self._pipe2mpeg4(image.shape, face.landmark, 152),
                    "13.16": self._pipe2mpeg4(image.shape, face.landmark, 148),
                    "13.14": self._pipe2mpeg4(image.shape, face.landmark, 176),
                    "13.12": self._pipe2mpeg4(image.shape, face.landmark, 150),
                    "13.10": self._pipe2mpeg4(image.shape, face.landmark, 136),
                    "13.8": self._pipe2mpeg4(image.shape, face.landmark, 215),
                    "13.6": self._pipe2mpeg4(image.shape, face.landmark, 132),
                    "13.4": self._pipe2mpeg4(image.shape, face.landmark, 93),
                    "13.2": self._pipe2mpeg4(image.shape, face.landmark, 127),

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
                    "12.1": self._pipe2mpeg4(image.shape, face.landmark, 195),
                    "9.4": self._pipe2mpeg4(image.shape, face.landmark, 99),
                    "9.2": self._pipe2mpeg4(image.shape, face.landmark, 166),
                    "9.3": self._pipe2mpeg4(image.shape, face.landmark, 275),
                    "9.12": self._pipe2mpeg4(image.shape, face.landmark, 248),
                    "9.1": self._pipe2mpeg4(image.shape, face.landmark, 294),
                    "9.5": self._pipe2mpeg4(image.shape, face.landmark, 460),
                    "9.15": self._pipe2mpeg4(image.shape, face.landmark, 370),

                    # Eye
                    "3.12": self._pipe2mpeg4(image.shape, face.landmark, 130),
                    "12.10": self._pipe2mpeg4(image.shape, face.landmark, 161),
                    "3.2": self._pipe2mpeg4(image.shape, face.landmark, 159),
                    "12.6": self._pipe2mpeg4(image.shape, face.landmark, 469),
                    "3.8": self._pipe2mpeg4(image.shape, face.landmark, 133),
                    "12.8": self._pipe2mpeg4(image.shape, face.landmark, 153),
                    "3.4": self._pipe2mpeg4(image.shape, face.landmark, 472),
                    "12.12": self._pipe2mpeg4(image.shape, face.landmark, 163),

                    # Eye
                    "3.11": self._pipe2mpeg4(image.shape, face.landmark, 398),
                    "12.9": self._pipe2mpeg4(image.shape, face.landmark, 476),
                    "3.1": self._pipe2mpeg4(image.shape, face.landmark, 386),
                    "12.5": self._pipe2mpeg4(image.shape, face.landmark, 387),
                    "3.7": self._pipe2mpeg4(image.shape, face.landmark, 359),
                    "12.7": self._pipe2mpeg4(image.shape, face.landmark, 390),
                    "3.3": self._pipe2mpeg4(image.shape, face.landmark, 374),
                    "12.11": self._pipe2mpeg4(image.shape, face.landmark, 380),

                    # Lips outer
                    "8.4": self._pipe2mpeg4(image.shape, face.landmark, 185),
                    "8.6": self._pipe2mpeg4(image.shape, face.landmark, 39),
                    "8.9": self._pipe2mpeg4(image.shape, face.landmark, 37),
                    "8.1": self._pipe2mpeg4(image.shape, face.landmark, 11),
                    "8.10": self._pipe2mpeg4(image.shape, face.landmark, 267),
                    "8.5": self._pipe2mpeg4(image.shape, face.landmark, 269),
                    "8.3": self._pipe2mpeg4(image.shape, face.landmark, 287),
                    "8.7": self._pipe2mpeg4(image.shape, face.landmark, 400),
                    "8.2": self._pipe2mpeg4(image.shape, face.landmark, 152),
                    "8.8": self._pipe2mpeg4(image.shape, face.landmark, 176),

                    # Lips inner
                    "2.5": self._pipe2mpeg4(image.shape, face.landmark, 185),
                    "2.7": self._pipe2mpeg4(image.shape, face.landmark, 37),
                    "2.2": self._pipe2mpeg4(image.shape, face.landmark, 0),
                    "2.6": self._pipe2mpeg4(image.shape, face.landmark, 269),
                    "2.4": self._pipe2mpeg4(image.shape, face.landmark, 273),
                    "2.8": self._pipe2mpeg4(image.shape, face.landmark, 428),
                    "2.3": self._pipe2mpeg4(image.shape, face.landmark, 175),
                    "2.9": self._pipe2mpeg4(image.shape, face.landmark, 208),

                    # Forehead
                    "11.2": self._pipe2mpeg4(image.shape, face.landmark, 67),
                    "11.1": self._pipe2mpeg4(image.shape, face.landmark, 10),
                    "11.3": self._pipe2mpeg4(image.shape, face.landmark, 297),
                }
            })

        return faces

    def stop(self):
        if self._track_in_background and self._track_proc and self._track_proc.is_alive():
            self._work_queue.put(None)
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

        dw = (brx - tlx) * expand
        dh = (bry - tly) * expand

        return (int(tlx - dw), int(tly - 2 * dh), int(brx + dw), int(bry))

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
