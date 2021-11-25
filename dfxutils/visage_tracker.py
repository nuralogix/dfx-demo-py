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

import os
import pathlib

import libvisage as visage


class VisageTracker():
    def __init__(self, visageLicense, maxFaces, frameWidth, frameHeight, use_analyser):
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

    @staticmethod
    def __version__():
        return visage.__version__

    @property
    def pointsPerFace(self):
        if "8.3" in self.__version__():
            return 191
        elif "8.6" in self.__version__() or "8.7" in self.__version__():
            return 188
        else:
            return 191

    def trackFaces(self, image, frameNumber, timeStamp_ms):
        # x, y, w, h = self._sanitizeRoi(image.shape, None)
        # searchImage = image[y:y + h, x:x + w]
        visageFrame = visage.VisageFrame.fromNumpy(image)
        statuses = self._tracker.track(visageFrame, timeStamp_ms, self._faceData)

        faces = {}
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
        pass
