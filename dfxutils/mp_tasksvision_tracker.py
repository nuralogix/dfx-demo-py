#
#              Copyright (c) 2025, Nuralogix Corp.
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

import cv2
import mediapipe
from mediapipe.tasks import python as mppython

import os


class TaskvisionTracker():

    def __init__(self, max_faces, track_in_background=False) -> None:
        self._mediapipe_initialized = False
        self._track_in_background = track_in_background
        self._init_params = max_faces, 0.5, 0.5
        self._last_tracked_faces = {}
        self._max_faces = max_faces
        max_faces, min_det_conf, min_track_conf = self._init_params
        self._initializeMediaPipe(max_faces, min_det_conf, min_track_conf)

    def _initializeMediaPipe(self, max_faces, min_det_conf, min_track_conf):
        model_path = os.path.join(os.getcwd(), "res", "face_landmarker.task")
        if not os.path.exists(model_path):
            print(
                "Please download "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task into 'res' folder"
            )
            raise FileNotFoundError(f"FaceLandmarker model file not found at {model_path}.")

        base_options = mppython.BaseOptions(model_asset_path=model_path)
        options = mppython.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=min_det_conf,
            min_face_presence_confidence=min_track_conf,
            min_tracking_confidence=min_track_conf,
            running_mode=mppython.vision.RunningMode.LIVE_STREAM
            if self._track_in_background else mppython.vision.RunningMode.VIDEO,
            result_callback=self._on_async_result if self._track_in_background else None,
        )
        self._face_landmarker = mppython.vision.FaceLandmarker.create_from_options(options)
        self._mediapipe_initialized = True

    @staticmethod
    def __version__():
        return mediapipe.__version__

    @property
    def pointsPerFace(self):
        return 468  # FaceLandmarker returns 468 landmarks per face

    def _on_async_result(self, result, output_image, timestamp_ms):
        # Callback for detect_async
        if result is not None and result.face_landmarks:
            shape = (output_image.height, output_image.width, output_image.channels)
            faces = self._parse_faces_from_result(result, shape)
            self._last_tracked_faces = faces
        else:
            self._last_tracked_faces = {}

    def trackFaces(self, image, _frameNumber, timeStamp_ms):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=image_rgb)
        if self._track_in_background:
            self._face_landmarker.detect_async(mp_image, int(timeStamp_ms))
            return self._last_tracked_faces
        else:
            result = self._face_landmarker.detect_for_video(mp_image, int(timeStamp_ms))
            if result is not None and result.face_landmarks:
                faces = self._parse_faces_from_result(result, image_rgb.shape)
                return faces
            else:
                return {}

    def _parse_faces_from_result(self, result, shape):
        faces = {}
        for faceNumber, face_landmarks in enumerate(result.face_landmarks):
            tlx, tly, brx, bry = self._findBoundingBox(shape, face_landmarks, 0.05)
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
                    "13.1": self._pipe2mpeg4(shape, face_landmarks, 356, 368),
                    "13.3": self._pipe2mpeg4(shape, face_landmarks, 454, 447),
                    "13.5": self._pipe2mpeg4(shape, face_landmarks, 366, 323, 361),
                    "13.7": self._pipe2mpeg4(shape, face_landmarks, 435, 288),
                    "13.9": self._pipe2mpeg4(shape, face_landmarks, 397, 367),
                    "13.11": self._pipe2mpeg4(shape, face_landmarks, 394, 365),
                    "13.13": self._pipe2mpeg4(shape, face_landmarks, 378),
                    "13.15": self._pipe2mpeg4(shape, face_landmarks, 377),
                    "13.17": self._pipe2mpeg4(shape, face_landmarks, 152),
                    "13.16": self._pipe2mpeg4(shape, face_landmarks, 148),
                    "13.14": self._pipe2mpeg4(shape, face_landmarks, 149),
                    "13.12": self._pipe2mpeg4(shape, face_landmarks, 136, 150),
                    "13.10": self._pipe2mpeg4(shape, face_landmarks, 172),
                    "13.8": self._pipe2mpeg4(shape, face_landmarks, 215, 58),
                    "13.6": self._pipe2mpeg4(shape, face_landmarks, 137, 93, 132),
                    "13.4": self._pipe2mpeg4(shape, face_landmarks, 234),
                    "13.2": self._pipe2mpeg4(shape, face_landmarks, 162, 127),

                    # Eyebrow
                    "4.6": self._pipe2mpeg4(shape, face_landmarks, 70),
                    "14.4": self._pipe2mpeg4(shape, face_landmarks, 63),
                    "4.4": self._pipe2mpeg4(shape, face_landmarks, 105),
                    "14.2": self._pipe2mpeg4(shape, face_landmarks, 66),
                    "4.2": self._pipe2mpeg4(shape, face_landmarks, 107),

                    # Eyebrow
                    "4.1": self._pipe2mpeg4(shape, face_landmarks, 336),
                    "14.1": self._pipe2mpeg4(shape, face_landmarks, 296),
                    "4.3": self._pipe2mpeg4(shape, face_landmarks, 334),
                    "14.3": self._pipe2mpeg4(shape, face_landmarks, 293),
                    "4.5": self._pipe2mpeg4(shape, face_landmarks, 300),

                    # Nose
                    "12.1": self._pipe2mpeg4(shape, face_landmarks, 196, 197),
                    "9.4": self._pipe2mpeg4(shape, face_landmarks, 240, 75),
                    "9.2": self._pipe2mpeg4(shape, face_landmarks, 64, 219),
                    "9.3": self._pipe2mpeg4(shape, face_landmarks, 4),
                    "9.12": self._pipe2mpeg4(shape, face_landmarks, 195, 197),
                    "9.1": self._pipe2mpeg4(shape, face_landmarks, 294),
                    "9.5": self._pipe2mpeg4(shape, face_landmarks, 290),
                    "9.15": self._pipe2mpeg4(shape, face_landmarks, 2, 370),

                    # Eye
                    "3.12": self._pipe2mpeg4(shape, face_landmarks, 130),
                    "12.10": self._pipe2mpeg4(shape, face_landmarks, 160, 161),
                    "3.2": self._pipe2mpeg4(shape, face_landmarks, 159),
                    "12.6": self._pipe2mpeg4(shape, face_landmarks, 469),
                    "3.8": self._pipe2mpeg4(shape, face_landmarks, 173),
                    "12.8": self._pipe2mpeg4(shape, face_landmarks, 153),
                    "3.4": self._pipe2mpeg4(shape, face_landmarks, 472, 144),
                    "12.12": self._pipe2mpeg4(shape, face_landmarks, 163),

                    # Eye
                    "3.11": self._pipe2mpeg4(shape, face_landmarks, 398),
                    "12.9": self._pipe2mpeg4(shape, face_landmarks, 476, 384),
                    "3.1": self._pipe2mpeg4(shape, face_landmarks, 386),
                    "12.5": self._pipe2mpeg4(shape, face_landmarks, 387),
                    "3.7": self._pipe2mpeg4(shape, face_landmarks, 263),
                    "12.7": self._pipe2mpeg4(shape, face_landmarks, 390),
                    "3.3": self._pipe2mpeg4(shape, face_landmarks, 374),
                    "12.11": self._pipe2mpeg4(shape, face_landmarks, 380),

                    # Lips outer
                    "8.4": self._pipe2mpeg4(shape, face_landmarks, 202, 61),
                    "8.6": self._pipe2mpeg4(shape, face_landmarks, 73, 39),
                    "8.9": self._pipe2mpeg4(shape, face_landmarks, 72, 37),
                    "8.1": self._pipe2mpeg4(shape, face_landmarks, 11),
                    "8.10": self._pipe2mpeg4(shape, face_landmarks, 302, 267),
                    "8.5": self._pipe2mpeg4(shape, face_landmarks, 269, 303),
                    "8.3": self._pipe2mpeg4(shape, face_landmarks, 273, 422, 291),
                    "8.7": self._pipe2mpeg4(shape, face_landmarks, 405, 418),
                    "8.2": self._pipe2mpeg4(shape, face_landmarks, 17, 200),
                    "8.8": self._pipe2mpeg4(shape, face_landmarks, 194, 181),

                    # Lips inner
                    "2.5": self._pipe2mpeg4(shape, face_landmarks, 61, 202),
                    "2.7": self._pipe2mpeg4(shape, face_landmarks, 179),
                    "2.2": self._pipe2mpeg4(shape, face_landmarks, 15, 13, 14),
                    "2.6": self._pipe2mpeg4(shape, face_landmarks, 403, 402),
                    "2.4": self._pipe2mpeg4(shape, face_landmarks, 291, 273),
                    "2.8": self._pipe2mpeg4(shape, face_landmarks, 403, 311),
                    "2.3": self._pipe2mpeg4(shape, face_landmarks, 15, 13, 14),
                    "2.9": self._pipe2mpeg4(shape, face_landmarks, 179, 81, 178),

                    # Forehead
                    "11.2": self._pipe2mpeg4(shape, face_landmarks, 67),
                    "11.1": self._pipe2mpeg4(shape, face_landmarks, 10),
                    "11.3": self._pipe2mpeg4(shape, face_landmarks, 297),

                    # Other
                    "2.1": self._pipe2mpeg4(shape, face_landmarks, 152),
                    "2.10": self._pipe2mpeg4(shape, face_landmarks, 152, 124, 117),
                    "2.11": self._pipe2mpeg4(shape, face_landmarks, 378),
                    "2.12": self._pipe2mpeg4(shape, face_landmarks, 149),
                    "2.13": self._pipe2mpeg4(shape, face_landmarks, 397),
                    "2.14": self._pipe2mpeg4(shape, face_landmarks, 172),
                    "3.5": self._pipe2mpeg4(shape, face_landmarks, 473),
                    "3.6": self._pipe2mpeg4(shape, face_landmarks, 468),
                    "3.9": self._pipe2mpeg4(shape, face_landmarks, 253),
                    "3.10": self._pipe2mpeg4(shape, face_landmarks, 23, 24),
                    "3.13": self._pipe2mpeg4(shape, face_landmarks, 257, 475),
                    "3.14": self._pipe2mpeg4(shape, face_landmarks, 26, 470),
                    "5.1": self._pipe2mpeg4(shape, face_landmarks, 185, 156),
                    "5.2": self._pipe2mpeg4(shape, face_landmarks, 213, 187, 192),
                    "5.3": self._pipe2mpeg4(shape, face_landmarks, 280, 266, 330),
                    "5.4": self._pipe2mpeg4(shape, face_landmarks, 50, 205, 101),
                    "9.6": self._pipe2mpeg4(shape, face_landmarks, 188),
                    "9.7": self._pipe2mpeg4(shape, face_landmarks, 317),
                    "9.13": self._pipe2mpeg4(shape, face_landmarks, 437, 466),
                    "9.14": self._pipe2mpeg4(shape, face_landmarks, 217, 236),
                    "14.5": self._pipe2mpeg4(shape, face_landmarks, 409, 270),
                    "14.6": self._pipe2mpeg4(shape, face_landmarks, 185),
                    "14.7": self._pipe2mpeg4(shape, face_landmarks, 321),
                    "14.8": self._pipe2mpeg4(shape, face_landmarks, 91),
                    "14.9": self._pipe2mpeg4(shape, face_landmarks, 73),
                    "14.10": self._pipe2mpeg4(shape, face_landmarks, 303, 302),
                    "14.11": self._pipe2mpeg4(shape, face_landmarks, 314, 421),
                    "14.12": self._pipe2mpeg4(shape, face_landmarks, 201, 84),
                    "14.13": self._pipe2mpeg4(shape, face_landmarks, 89),
                    "14.14": self._pipe2mpeg4(shape, face_landmarks, 319),
                    "14.15": self._pipe2mpeg4(shape, face_landmarks, 89, 88),
                    "14.16": self._pipe2mpeg4(shape, face_landmarks, 319, 310),
                    "14.17": self._pipe2mpeg4(shape, face_landmarks, 86),
                    "14.18": self._pipe2mpeg4(shape, face_landmarks, 316, 312),
                    "14.19": self._pipe2mpeg4(shape, face_landmarks, 86),
                    "14.20": self._pipe2mpeg4(shape, face_landmarks, 316),
                    "14.21": self._pipe2mpeg4(shape, face_landmarks, 4),
                    "14.22": self._pipe2mpeg4(shape, face_landmarks, 5),
                    "14.23": self._pipe2mpeg4(shape, face_landmarks, 195),
                    "14.24": self._pipe2mpeg4(shape, face_landmarks, 357),
                    "14.25": self._pipe2mpeg4(shape, face_landmarks, 375),
                    "15.1": self._pipe2mpeg4(shape, face_landmarks, 356, 368),
                    "15.2": self._pipe2mpeg4(shape, face_landmarks, 296),
                    "15.3": self._pipe2mpeg4(shape, face_landmarks, 454),
                    "15.4": self._pipe2mpeg4(shape, face_landmarks, 234),
                    "15.5": self._pipe2mpeg4(shape, face_landmarks, 352, 366),
                    "15.6": self._pipe2mpeg4(shape, face_landmarks, 137),
                    "15.7": self._pipe2mpeg4(shape, face_landmarks, 288),
                    "15.8": self._pipe2mpeg4(shape, face_landmarks, 215),
                    "15.9": self._pipe2mpeg4(shape, face_landmarks, 397),
                    "15.10": self._pipe2mpeg4(shape, face_landmarks, 172),
                    "15.11": self._pipe2mpeg4(shape, face_landmarks, 394, 365),
                    "15.12": self._pipe2mpeg4(shape, face_landmarks, 136),
                    "15.13": self._pipe2mpeg4(shape, face_landmarks, 378),
                    "15.14": self._pipe2mpeg4(shape, face_landmarks, 149),
                    "15.15": self._pipe2mpeg4(shape, face_landmarks, 377),
                    "15.16": self._pipe2mpeg4(shape, face_landmarks, 148),
                    "15.17": self._pipe2mpeg4(shape, face_landmarks, 152),
                }
            })

        return faces

    def stop(self):
        # No longer needed, but kept for API compatibility
        pass

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
