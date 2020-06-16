import json
import os

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
        face.setRect(json_face['rect.x'], json_face['rect.y'], json_face['rect.w'], json_face['rect.h'])
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
