#
#              Copyright (c) 2016, Nuralogix Corp.
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

import libvisage as visage
import numpy as np


class VisageTracker():
    def __init__(self, visageLicense, maxFaces, frameWidth, frameHeight, use_analyser, track_in_background=False):
        self._visage_initialized = False
        self._track_in_background = track_in_background
        self._visage_init_params = visageLicense, use_analyser, maxFaces, frameWidth, frameHeight
        self._last_tracked_faces = {}
        if self._track_in_background:
            self._work_queue = mp.Queue(1)
            self._results_queue = mp.Queue(1)
            self._track_proc = mp.Process(target=self._trackFacesThreaded, name="visage_tracker")
            self._track_proc.start()
        else:
            self._initializeVisage(visageLicense, use_analyser, maxFaces, frameWidth, frameHeight)

    def _initializeVisage(self, visageLicense, use_analyser, maxFaces, frameWidth, frameHeight):
        # Create a Visage Factory object
        self._visageFactory = visage.Factory()

        visageCfg = os.path.join(os.getcwd(), "res", "visage", "Facial Features Tracker - Ultra.cfg")

        # License
        if visageLicense is None:
            self._visageFactory.initializeLicenseManager("")
        else:
            self._visageFactory.initializeLicenseManager(str(os.path.realpath(visageLicense)))

        # Create tracker
        trackerPath = os.path.realpath(visageCfg)
        self._tracker = self._visageFactory.createTracker(str(trackerPath))

        # Create face analyser
        self._analyser = None
        if use_analyser:
            self._analyser = self._visageFactory.createFaceAnalyser()
            analyserResPath = os.path.join(pathlib.Path(trackerPath).parent, "bdtsdata", "LBF", "vfadata")
            self._analyser.init(str(analyserResPath))

        # Create a faceData object
        self._faceData = self._visageFactory.createFaceData(maxFaces, frameWidth, frameHeight)

        self._visage_initialized = True

    @staticmethod
    def __version__():
        return visage.__version__

    @property
    def pointsPerFace(self):
        if "8.3" in self.__version__():
            return 191
        elif "8.6" in self.__version__():
            return 188
        elif "8.7" in self.__version__():
            return 205
        elif "8.8" in self.__version__():
            return 205
        else:
            return 191

    def _trackFacesThreaded(self):
        visageLicense, use_analyser, maxFaces, frameWidth, frameHeight = self._visage_init_params
        self._initializeVisage(visageLicense, use_analyser, maxFaces, frameWidth, frameHeight)

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

    def _trackFaces(self, image, frameNumber, timeStamp_ms):
        faces = {}

        if not self._visage_initialized:
            return faces

        # x, y, w, h = self._sanitizeRoi(image.shape, None)
        # searchImage = image[y:y + h, x:x + w]
        visageFrame = visage.VisageFrame.fromNumpy(image)
        statuses = self._tracker.track(visageFrame, timeStamp_ms, self._faceData)

        for faceNumber, status in enumerate(statuses):
            if status == visage.TrackerStatus.OK:
                visageface = self._faceData[faceNumber]
                visagepoints = visageface.points
                if self._analyser is not None:
                    visageage = self._analyser.estimateAge(visageFrame, self._faceData)
                    visagegender = self._analyser.estimateGender(visageFrame, self._faceData)
                    visageemotions = self._analyser.estimateEmotion(visageFrame, self._faceData)

                points = {}
                tlx, tly, brx, bry = (image.shape[1], image.shape[0], 0, 0)
                for p in visagepoints:
                    points[p.id] = {
                        "x": p.scaledX,
                        "y": p.scaledY,
                        "z": p.z,
                        "valid": p.valid,
                        "estimated": p.estimated,
                        "quality": p.quality
                    }
                    if p.valid:
                        tlx = p.scaledX if p.scaledX < tlx else tlx
                        tly = p.scaledY if p.scaledY < tly else tly
                        brx = p.scaledX if p.scaledX > brx else brx
                        bry = p.scaledY if p.scaledY > bry else bry
                # Ensure face rect within image
                tlx, tly, brx, bry = max(tlx, 0), max(tly, 0), min(brx, image.shape[1]), min(bry, image.shape[0])

                faces[str(faceNumber + 1)] = {
                    "id": str(faceNumber + 1),
                    "rect.x": tlx,
                    "rect.y": tly,
                    "rect.w": brx - tlx,
                    "rect.h": bry - tly,
                    "detected": True,
                    "poseValid": True if len(points) > 0 else False,
                    "points": points,
                    "translation:x": visageface.face_translation[0],
                    "translation:y": visageface.face_translation[1],
                    "translation:z": visageface.face_translation[2],
                    "rotation:x": visageface.face_rotation[0],
                    "rotation:y": visageface.face_rotation[1],
                    "rotation:z": visageface.face_rotation[2],
                    "gaze_direction:x": visageface.gaze_direction[0],
                    "gaze_direction:y": visageface.gaze_direction[1],
                    "gaze_direction_global:x": visageface.gaze_direction_global[0],
                    "gaze_direction_global:y": visageface.gaze_direction_global[1],
                    "gaze_direction_global:z": visageface.gaze_direction_global[2],
                    "eye_closure:left": visageface.eye_closure[0],
                    "eye_closure:right": visageface.eye_closure[1],
                    "face_scale": visageface.face_scale,
                    "camera_focus": visageface.camera_focus,
                    "gaze_quality": visageface.gaze_quality,
                    "tracking_quality": visageface.tracking_quality
                }
                if self._analyser is not None:
                    faces[str(faceNumber + 1)].update({
                        "age": visageage,
                        "gender": visagegender,
                        "emotion:result": True if len(visageemotions) > 0 else False,
                        "emotion:anger": visageemotions[0],
                        "emotion:disgust": visageemotions[1],
                        "emotion:fear": visageemotions[2],
                        "emotion:happiness": visageemotions[3],
                        "emotion:sadness": visageemotions[4],
                        "emotion:surprise": visageemotions[5],
                        "emotion:neutral": visageemotions[6],
                    })
        return faces

    def stop(self):
        if self._track_in_background and self._track_proc and self._track_proc.is_alive():
            self._work_queue.put((None, None, None))
            self._track_proc.terminate()
            self._work_queue.cancel_join_thread()
            self._results_queue.cancel_join_thread()
