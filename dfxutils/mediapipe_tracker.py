#
#              Copyright (c) 2016-2021, Nuralogix Corp.
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
import os
import queue
import pathlib

import mediapipe


class MediaPipeTracker():

    def __init__(self, maxFaces, track_in_background=False) -> None:
        self._mediapipe_initialized = False
        self._track_in_background = track_in_background
        self._last_tracked_faces = {}
        self._face_mesh = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=maxFaces,
                                                                 refine_landmarks=True,
                                                                 min_detection_confidence=0.5,
                                                                 min_tracking_confidence=0.5)
        self._mediapipe_initialized = True

    def __del__(self):
        del self._face_mesh

    def trackFaces(self, image, frameNumber, timeStamp_ms):
        self._last_tracked_faces = self._trackFaces(image, frameNumber, timeStamp_ms)

        return self._last_tracked_faces

    def _trackFaces(self, image, _frameNumber, _timeStamp_ms):
        faces = {}

        if not self._mediapipe_initialized:
            return faces

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
        pass

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
