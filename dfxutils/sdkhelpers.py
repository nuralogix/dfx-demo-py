import json
import os
from typing import Dict

import libdfx as dfxsdk


class DfxSdkHelpers:
    @staticmethod
    def save_chunk(chunk: dfxsdk.ChunkData, output_folder: str) -> None:
        props = {
            "valid": chunk.valid,
            "start_frame": chunk.start_frame,
            "end_frame": chunk.end_frame,
            "chunk_number": chunk.chunk_number,
            "number_chunks": chunk.number_chunks,
            "first_chunk_start_time_s": chunk.first_chunk_start_time_s,
            "start_time_s": chunk.start_time_s,
            "end_time_s": chunk.end_time_s,
            "duration_s": chunk.duration_s,
        }
        os.makedirs(output_folder, exist_ok=True)
        prop_path = os.path.join(output_folder, f"properties{chunk.chunk_number:04}.json")
        payload_path = os.path.join(output_folder, f"payload{chunk.chunk_number:04}.bin")
        meta_path = os.path.join(output_folder, f"metadata{chunk.chunk_number:04}.bin")
        with open(prop_path, "w") as f_props, open(payload_path, "wb") as f_pay, open(meta_path, "wb") as f_meta:
            json.dump(props, f_props)
            f_pay.write(chunk.payload_data)
            f_meta.write(chunk.metadata)

    @staticmethod
    def dfx_face_from_json(collector: dfxsdk.Collector, json_face: dict) -> dfxsdk.Face:
        face = collector.createFace(json_face["id"])
        # Face rect needs integers
        face.setRect(int(json_face['rect.x']), int(json_face['rect.y']), int(json_face['rect.w']),
                     int(json_face['rect.h']))
        face.setPoseValid(json_face['poseValid'])
        face.setDetected(json_face['detected'])
        points = json_face['points']
        for pointId, point in points.items():
            face.addPosePoint(pointId,
                              point['x'],
                              point['y'],
                              valid=point['valid'],
                              estimated=point['estimated'],
                              quality=point['quality'])
        return face

    @staticmethod
    def sdk_result_to_dict(sdk_result: dfxsdk.MeasurementResult) -> dict:
        dict_result = {}

        if not sdk_result.isValid():
            return dict_result

        dict_result["chunk_number"] = sdk_result.getMeasurementProperty('MeasurementDataID').split(':')[-1]
        status = sdk_result.getErrorCode()
        if status != "OK":
            dict_result["Status"] = status

        for k in sdk_result.getMeasurementDataKeys():
            data_result = sdk_result.getMeasurementData(k)
            value = data_result.getData()
            dict_result[k] = str(sum(value) / len(value)) if len(value) > 0 else None

        return dict_result

    _constraints_messages = {
        "FaceNone": ("no face detected", "Move face into target region"),
        "FaceOffTarget": ("not in target region", "Move face into target region"),
        "FaceDirection": ("not looking at camera", "Look straight at the camera"),
        "FaceFar": ("too far from camera", "Move closer to the camera"),
        "FaceMovement": ("moving too much", "Hold still"),
        "ImageBright": ("image too bright", "Image too bright - try a darker room"),
        "ImageDark": ("image too dark", "Image too dark - try a brighter room"),
        "ImageQuality": ("bad image quality", "Improve image quality - try alternate webcam"),
        "ImageBackLit": ("backlit face", "Remove backlight behind face"),
        "LowFps": ("framerate too low", "Framerate is too low - try alternate webcam or a brighter room"),
    }

    @staticmethod
    def user_feedback_from_constraints(constraint_violation_details: Dict[str, dfxsdk.ConstraintResult]):
        for k, v in constraint_violation_details.items():
            if v == dfxsdk.ConstraintResult.WARN or v == dfxsdk.ConstraintResult.ERROR:
                if k in DfxSdkHelpers._constraints_messages:
                    return DfxSdkHelpers._constraints_messages[k][1]
                else:
                    return k

        return ""

    @staticmethod
    def failure_causes_from_constraints(constraint_violation_details: Dict[str, dfxsdk.ConstraintResult]):
        causes = []
        for k, v in constraint_violation_details.items():
            if v == dfxsdk.ConstraintResult.ERROR:
                if k in DfxSdkHelpers._constraints_messages:
                    causes.append(DfxSdkHelpers._constraints_messages[k][0])
                else:
                    causes.append(k)
        if len(causes) > 0:
            return "Failed because " + ", ".join(causes)
        else:
            return "Failed because of unknown reasons!"

    class ConstraintsConfig():
        def __init__(self, cfg_str: str) -> None:
            self.__dict__.update(json.loads(cfg_str))

        def __str__(self) -> str:
            return json.dumps(self.__dict__)

    @staticmethod
    def json_result_to_dict(json_result: dict) -> dict:
        result = {
            "chunk_number":
            int(json_result['MeasurementDataID'].split(':')[-1]) if 'MeasurementDataID' in json_result else None,
            "Status": json_result["Error"]["Code"],
            "Errors": json_result["Error"]["Errors"] if "Errors" in json_result["Error"] else None,
        }

        if "Channels" not in json_result:
            return result

        notes = {}
        multiplier = float(json_result["Multiplier"])
        for k, v in sorted(json_result["Channels"].items()):
            data_values = v["Data"]
            if (len(data_values) > 0):
                result.update({k: (sum(data_values) / len(data_values)) / multiplier})
            else:
                result.update({k: None})
            if "Notes" in v:
                notes[k] = v["Notes"]
        if len(notes) > 0:
            result["Notes"] = notes

        return result

    @staticmethod
    def set_user_demographics(collector, demographics):
        if (gender := demographics.get("gender")) is not None:
            sex_at_birth = dfxsdk.FaceValue.SEX_NOT_PROVIDED
            if gender == "male":
                sex_at_birth = dfxsdk.FaceValue.SEX_ASSIGNED_MALE_AT_BIRTH
            elif gender == "female":
                sex_at_birth = dfxsdk.FaceValue.SEX_ASSIGNED_FEMALE_AT_BIRTH
            collector.setFaceAttribute("1", dfxsdk.FaceAttribute.SEX_ASSIGNED_AT_BIRTH, sex_at_birth)
            print("       SEX_ASSIGNED_AT_BIRTH:", sex_at_birth)
        if (age := demographics.get("age")) is not None:
            success = collector.setFaceAttribute("1", dfxsdk.FaceAttribute.AGE_YEARS, age)
            if success:
                print(f"       AGE_YEARS: {age}")
            else:
                return False
        if (height := demographics.get("height")) is not None:
            success = collector.setFaceAttribute("1", dfxsdk.FaceAttribute.HEIGHT_CM, height)
            if success:
                print(f"       HEIGHT_CM: {height}")
            else:
                return
        if (weight := demographics.get("weight")) is not None:
            success = collector.setFaceAttribute("1", dfxsdk.FaceAttribute.WEIGHT_KG, weight)
            if success:
                print(f"       WEIGHT_KG: {weight}")
            else:
                return
        if (smoking := demographics.get("smoking")) is not None:
            collector.setFaceAttribute("1", dfxsdk.FaceAttribute.SMOKER, smoking)
            print(f"       SMOKER: {smoking}")
        if (diabetes_text := demographics.get("diabetes")) in ["0", "type1", "type2"]:
            diabetes = dfxsdk.FaceValue.DIABETES_NONE
            if diabetes_text == "type1":
                diabetes = dfxsdk.FaceValue.DIABETES_TYPE1
            elif diabetes_text == "type2":
                diabetes = dfxsdk.FaceValue.DIABETES_TYPE2
            collector.setFaceAttribute("1", dfxsdk.FaceAttribute.DIABETES, diabetes)
            print("       DIABETES:", diabetes)
        if (bloodpressuremedication := demographics.get("bloodpressuremedication")) is not None:
            collector.setFaceAttribute("1", dfxsdk.FaceAttribute.BLOOD_PRESSURE_MEDICATION, bloodpressuremedication)
            print(f"       BLOOD_PRESSURE_MEDICATION: {bloodpressuremedication}")
        if (hypertensive := demographics.get("hypertensive")) is not None:
            collector.setFaceAttribute("1", dfxsdk.FaceAttribute.HYPERTENSIVE, hypertensive)
            print(f"       HYPERTENSIVE: {hypertensive}")

        return True
