import asyncio
import datetime
import json
import os

import cv2
import numpy as np

TIMESTAMP_KEYS = ["Created", "Updated"]


async def find_video_rotation(video_path):
    """Find video rotation using ffprobe."""
    angle = 0
    ffprobe_cmd = "ffprobe -v quiet -select_streams v:0 -show_entries " \
                  "stream_tags=rotate -of default=nw=1:nk=1".split(' ')
    ffprobe_cmd.append(video_path)
    try:
        proc = await asyncio.create_subprocess_exec(*ffprobe_cmd, stdout=asyncio.subprocess.PIPE)
        op, err = await proc.communicate()
        op = op.decode()
        for line in op.split('\n'):
            if "90" in line:
                angle = 90
            elif "180" in line:
                angle = 180
            elif "270" in line:
                angle = 270
    except OSError:
        # Likely couldn't find ffprobe
        pass

    if angle < 0:
        angle = angle + 360

    return angle


async def read_next_frame(video_cap, target_fps, rotation, mirror):
    if target_fps > 0:
        await asyncio.sleep(1.0 / target_fps)

    read, frame = video_cap.read()

    if read and frame is not None and frame.size != 0:
        # Mirror frame if necessary
        if mirror:
            frame = cv2.flip(frame, 1)

        # Rotate frame if necessary
        if rotation == 90:
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
        elif rotation == 180:
            frame = cv2.flip(frame, -1)
        elif rotation == 270:
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 0)
        else:  # 0 or some other weird result
            pass
    else:
        # This is needed because OpenCV doesn't do a good job of ensuring this.
        read = False
        frame = None

    return read, frame


def save_chunk(chunk, output_folder):
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


def dfx_face_from_json(collector, json_face):
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


def draw_on_image(dfxframe,
                  render_image,
                  image_src_name,
                  frame_number,
                  video_duration_frames,
                  fps,
                  measurement_active,
                  results,
                  message=None):
    # Render the face polygons
    for faceID in dfxframe.getFaceIdentifiers():
        for regionID in dfxframe.getRegionNames(faceID):
            if (dfxframe.getRegionIntProperty(faceID, regionID, "draw") != 0):
                polygon = dfxframe.getRegionPolygon(faceID, regionID)
                cv2.polylines(render_image, [np.array(polygon)],
                              isClosed=True,
                              color=(255, 255, 0),
                              thickness=1,
                              lineType=cv2.LINE_AA)
    # Render the "Extracting " message
    current_row = 30
    if measurement_active:
        msg = f"Extracting from {image_src_name} - {video_duration_frames - frame_number} frames left ({fps:.2f} fps)"
    else:
        msg = f"Reading from {image_src_name} - ({fps:.2f} fps)"
    cv2.putText(render_image,
                msg,
                org=(10, current_row),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)

    # Render the message
    if message:
        current_row += 30
        cv2.putText(render_image,
                    message,
                    org=(10, current_row),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # Render the results
    if results:
        for k, v in results.items():
            current_row += 12
            cv2.putText(render_image,
                        f"{k}: {v}",
                        org=(20, current_row),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=0.8,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

    # Render the current time (so user knows things arent frozen)
    now = datetime.datetime.now()
    cv2.putText(render_image,
                f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}",
                org=(render_image.shape[1] - 90, render_image.shape[0] - 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 0) if now.second % 2 == 0 else (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)

    return current_row


def print_pretty(x, csv=False, indent=0) -> None:
    if type(x) == list:
        print_list(x, csv, indent)
    elif type(x) == dict:
        print_dict(x, csv, indent)
    else:
        print(x)


def print_sdk_result(result):
    if not result.isValid():
        print("Received invalid result from DFX SDK collector decode!!")
        return

    print(f"Received chunk {result.getMeasurementProperty('MeasurementDataID').split(':')[-1]}")

    dict_result = {}

    status = result.getErrorCode()
    if status != "OK":
        dict_result["Status"] = status

    for k in result.getMeasurementDataKeys():
        data_result = result.getMeasurementData(k)
        value = data_result.getData()
        dict_result[k] = str(sum(value) / len(value)) if len(value) > 0 else None

    print_pretty(dict_result, indent=2)


def print_meas(measurement_results, csv=False):
    if measurement_results["Results"]:
        grid_results = []
        for signal_id, signal_name in measurement_results["SignalNames"].items():
            result_data = measurement_results["Results"][signal_id][0]
            units = measurement_results["SignalUnits"][signal_id]
            description = measurement_results["SignalDescriptions"][signal_id]
            if not csv and len(description) > 100:
                description = description[:100] + "..."
            result_value = sum(result_data["Data"]) / len(result_data["Data"]) / result_data["Multiplier"] if len(
                result_data["Data"]) > 0 and result_data["Multiplier"] > 0 else None
            grid_result = {
                "ID": signal_id,
                "Name": signal_name,
                "Value": result_value,
                "Unit": units if units is not None else "",
                "Category": measurement_results["SignalConfig"][signal_id]["category"],
                "Description": description
            }
            grid_results.append(grid_result)
        measurement_results["Results"] = grid_results
        del measurement_results["SignalNames"]
        del measurement_results["SignalUnits"]
        del measurement_results["SignalDescriptions"]
        del measurement_results["SignalConfig"]
    print_pretty(measurement_results, csv)


def print_dict(dict_, csv, indent) -> None:
    sep = "," if csv else ": "
    for k, v in dict_.items():
        if type(v) == list:
            print(indent * " " + f"{k}{sep}")
            print_list(v, csv, indent + 2)
        elif type(v) == dict:
            print(indent * " " + f"{k}{sep}")
            print_dict(v, csv, indent + 2)
        else:
            if v is None:
                vv = ""
            elif k in TIMESTAMP_KEYS:
                vv = datetime.datetime.fromtimestamp(v)
            else:
                vv = v
            print(indent * " " + f"{k}{sep}{vv}")


def print_list(list_, csv, indent):
    if len(list_) > 0 and type(list_[0]) == dict:
        print_grid(list_, csv, indent)
        return

    for item in list_:
        if type(item) == list:
            print_list(item, csv, indent + 2)
        elif type(item) == dict:
            print_dict(item, csv, indent + 2)
        else:
            print(indent * " " + item)


def print_grid(list_of_dicts, csv, indent):
    if len(list_of_dicts) <= 0:
        return

    for dict_ in list_of_dicts:
        for k, v in dict_.items():
            if v is None:
                dict_[k] = ""
            elif k in TIMESTAMP_KEYS:
                ts = datetime.datetime.fromtimestamp(v)
                if csv:
                    dict_[k] = str(ts)
                else:
                    dict_[k] = ts.strftime("%Y-%m-%d")

    if csv:
        print(indent * " " + ",".join([f"{key}" for key in list_of_dicts[0].keys()]))
        for dict_ in list_of_dicts:
            print(indent * " " + ",".join([f"{value}" for value in dict_.values()]))
        return

    col_widths = [len(str(k)) for k in list_of_dicts[0].keys()]
    for dict_ in list_of_dicts:
        for i, v in enumerate(dict_.values()):
            col_widths[i] = max(col_widths[i], len(str(v)))
    print(indent * " " + "".join([f"{str(key):{cw}} " for (cw, key) in zip(col_widths, list_of_dicts[0].keys())]))
    for dict_ in list_of_dicts:
        print(indent * " " + "".join([f"{str(value):{cw}} " for (cw, value) in zip(col_widths, dict_.values())]))
