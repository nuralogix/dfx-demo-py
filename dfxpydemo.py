import argparse
import asyncio
import base64
import json
import math
import os.path
import random
import string

import aiohttp
import cv2
import numpy as np

import libdfx as dfxsdk
import dfx_apiv2_client as dfxapi

from dfxpydemoutils import (DlibTracker, dfx_face_from_json, draw_on_image, find_video_rotation, print_meas,
                            print_pretty, read_next_frame, save_chunk)


async def main(args):
    # Load config
    config = load_config(args.config_file)

    # Check API status
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        api_status = await dfxapi.General.api_status(session)
        if not api_status["StatusID"] == "ACTIVE":
            print(f"DFX API Status: {api_status['StatusID']} ({dfxapi.Settings.rest_url})")

            return

    # Register or unregister
    if args.command == "orgs":
        if args.subcommand == "unregister":
            success = await unregister(config, args.config_file)
        else:
            success = await register(config, args.config_file, args.license_key)

        if success:
            save_config(config, args.config_file)
        return

    # Login or logout
    if args.command == "users":
        if args.subcommand == "logout":
            success = logout(config, args.config_file)
        else:
            success = await login(config, args.config_file, args.email, args.password)

        if success:
            save_config(config, args.config_file)
        return

    # The commands below need a token, so make sure we are registered and/or logged in
    if not dfxapi.Settings.device_token and not dfxapi.Settings.user_token:
        print("Please register and/or login first to obtain a token")
        return

    # Use the token to create the headers
    token = dfxapi.Settings.user_token if dfxapi.Settings.user_token else dfxapi.Settings.device_token
    headers = {"Authorization": f"Bearer {token}"}

    # Retrieve or list studies
    if args.command == "studies":
        async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
            if args.subcommand == "get":
                study_id = config["selected_study"] if args.study_id is None else args.study_id
                if not study_id or study_id.isspace():
                    print("Please select a study or pass a study id")
                    return
                study = await dfxapi.Studies.retrieve(session, study_id)
                print(json.dumps(study)) if args.json else print_pretty(study, args.csv)
            elif args.subcommand == "list":
                studies = await dfxapi.Studies.list(session)
                print(json.dumps(studies)) if args.json else print_pretty(studies, args.csv)
            elif args.subcommand == "select":
                config["selected_study"] = args.study_id
                save_config(config, args.config_file)
        return

    # Retrieve or list measurements
    if args.command == "meas" and args.subcommand != "make":
        async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
            if args.subcommand == "get":
                measurement_id = config["last_measurement"] if args.measurement_id is None else args.measurement_id
                if not measurement_id or measurement_id.isspace():
                    print("Please complete a measurement first or pass a measurement id")
                    return
                results = await dfxapi.Measurements.retrieve(session, measurement_id)
                print(json.dumps(results)) if args.json else print_meas(results, args.csv)
            elif args.subcommand == "list":
                measurements = await dfxapi.Measurements.list(session, limit=args.limit)
                print(json.dumps(measurements)) if args.json else print_pretty(measurements, args.csv)
        return

    # Make a measurement
    assert args.command == "meas" and args.subcommand == "make"

    # Verify preconditions
    if not config["selected_study"]:
        print("Please select a study first using 'study select'")
        return

    # Open video or camera (or FAIL)
    videocap = cv2.VideoCapture(args.video_path)
    if not videocap.isOpened():
        print(f"Could not open {args.video_path}")
        return
    fps = videocap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Video framerate {fps} is invalid. Please override using '--fps' parameter.")
        return
    frames_to_process = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames_to_process / fps > 120:
        print(f"Video duration {frames_to_process / fps:.1f}s is longer than 120s, processing first 120s only.")
        frames_to_process = int(fps * 120)
    rotation = await find_video_rotation(args.video_path)
    frame_duration_ns = 1000000000.0 / fps

    # Create a DlibTracker (or FAIL)
    tracker = None
    try:
        tracker = DlibTracker()
    except RuntimeError as e:
        print(e)
        print("Please download and unzip "
              "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 into 'res' folder")
        return

    # Create DFX SDK factory
    factory = dfxsdk.Factory()
    print("Created DFX Factory:", factory.getVersion())
    sdk_id = factory.getSdkId()

    # Get study config data from API required to initialize DFX SDK collector (or FAIL)
    # TODO: Handle 404 properly here...
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        response = await dfxapi.Studies.retrieve_sdk_config_hash(session, config["selected_study"], sdk_id)
        if response["MD5Hash"] != config["study_cfg_hash"]:
            response = await dfxapi.Studies.retrieve_sdk_config_data(session, config["selected_study"], sdk_id)
            config["study_cfg_hash"] = response["MD5Hash"]
            config["study_cfg_data"] = response["ConfigFile"]
            print(f"Retrieved new study config data with md5: {config['study_cfg_hash']}")
            save_config(config, args.config_file)
    study_cfg_bytes = base64.standard_b64decode(config["study_cfg_data"])

    # Create DFX SDK collector (or FAIL)
    if not factory.initializeStudy(study_cfg_bytes):
        print(f"DFX factory creation failed: {factory.getLastErrorMessage()}")
        return
    factory.setMode("discrete")
    collector = factory.createCollector()
    if collector.getCollectorState() == dfxsdk.CollectorState.ERROR:
        print(f"DFX collector creation failed: {collector.getLastErrorMessage()}")
        return

    print("Created DFX Collector:")
    chunk_duration_s = float(args.chunk_duration_s)
    frames_per_chunk = math.ceil(chunk_duration_s * fps)
    number_chunks = math.ceil(frames_to_process / frames_per_chunk)

    # Set collector config
    collector.setTargetFPS(fps)
    collector.setChunkDurationSeconds(chunk_duration_s)
    collector.setNumberChunks(number_chunks)
    print(f"    mode: {factory.getMode()}")
    print(f"    number chunks: {collector.getNumberChunks()}")
    print(f"    chunk duration: {collector.getChunkDurationSeconds()}s")
    for constraint in collector.getEnabledConstraints():
        print(f"    enabled constraint: {constraint}")

    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        # Create a measurement
        measurement_id = await dfxapi.Measurements.create(session, config["selected_study"])
        print(f"Created measurement {measurement_id}")

        # Start the DFX SDK collection
        collector.startCollection()

        # Use the session to connect to the WebSocket
        async with session.ws_connect(dfxapi.Settings.ws_url) as ws:
            # Subscribe to results
            results_request_id = generate_reqid()
            await dfxapi.Measurements.ws_subscribe_to_results(ws, results_request_id, measurement_id)

            # Use this to close WebSocket in the receive loop
            chunk_queue = asyncio.Queue(number_chunks)
            results_expected = number_chunks

            # Produce chunks
            async def extract_video():
                frame_number = 0
                while True:
                    read, image = await read_next_frame(videocap, fps, rotation, False)
                    if not read or image is None or frame_number >= frames_to_process:
                        # Video ended, so grab what should be the last, possibly truncated chunk
                        collector.forceComplete()
                        chunk_data = collector.getChunkData()
                        if chunk_data is not None:
                            chunk = chunk_data.getChunkPayload()
                            await chunk_queue.put(chunk)
                            break

                    # Track faces
                    tracked_faces = tracker.trackFaces(image)

                    # Create a DFX VideoFrame, then a DFX Frame from the DFX VideoFrame and add DFX faces to it
                    dfx_video_frame = dfxsdk.VideoFrame(image, frame_number, frame_number * frame_duration_ns,
                                                        dfxsdk.ChannelOrder.CHANNEL_ORDER_BGR)
                    dfx_frame = collector.createFrame(dfx_video_frame)
                    if len(tracked_faces) > 0:
                        tracked_face = next(iter(tracked_faces.values()))  # We only care about the first face
                        dfx_face = dfx_face_from_json(collector, tracked_face)
                        dfx_frame.addFace(dfx_face)

                    # Do the extraction
                    collector.defineRegions(dfx_frame)
                    result = collector.extractChannels(dfx_frame)

                    # Grab a chunk and check if we are finished
                    if result == dfxsdk.CollectorState.CHUNKREADY or result == dfxsdk.CollectorState.COMPLETED:
                        chunk_data = collector.getChunkData()
                        if chunk_data is not None:
                            chunk = chunk_data.getChunkPayload()
                            await chunk_queue.put(chunk)
                        if result == dfxsdk.CollectorState.COMPLETED:
                            break

                    # We should really be getting this from OpenCV but it's unreliable...
                    frame_number += 1

                    # Rendering
                    if not args.no_render:
                        render_image = np.copy(image)
                        draw_on_image(dfx_frame, render_image, os.path.basename(args.video_path), frame_number,
                                      frames_to_process, fps, True, None, None)

                        cv2.imshow("d", render_image)
                        cv2.waitKey(1)

                # Signal to send_chunks that we are done
                await chunk_queue.put(None)

            async def send_chunks():
                # Coroutine to iterate through the payload files and send chunks using WebSocket
                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        chunk_queue.task_done()
                        break

                    # Determine action and request id
                    action = determine_action(chunk.chunk_number, chunk.number_chunks)

                    # Add data
                    await dfxapi.Measurements.ws_add_data(ws, generate_reqid(), measurement_id, chunk.chunk_number,
                                                          action, chunk.start_time_s, chunk.end_time_s,
                                                          chunk.duration_s, chunk.metadata, chunk.payload_data)
                    print(f"Sent chunk {chunk.chunk_number}")

                    # Save chunk (for debugging purposes)
                    if args.debug_save_chunks_path:
                        save_chunk(chunk, args.debug_save_chunks_path)
                        print(f"Saved chunk {chunk.chunk_number} in '{args.debug_save_chunks_path}'")

                    chunk_queue.task_done()

            async def receive_results():
                # Coroutine to receive results
                num_results_received = 0
                async for msg in ws:
                    status, request_id, payload = dfxapi.Measurements.ws_decode(msg)
                    if request_id == results_request_id and len(payload) > 0:
                        result = collector.decodeMeasurementResult(payload)
                        print_result(result)
                        num_results_received += 1
                    if num_results_received == results_expected:
                        await ws.close()
                        break

            # Start the three coroutines and await till they finish
            try:
                await asyncio.gather(extract_video(), send_chunks(), receive_results())
            except Exception as e:
                print(e)
                print(f"Measurement {measurement_id} failed")
                return
            finally:
                tracker.stop()
                print("Stopping face tracker...")

        print(f"Measurement {measurement_id} complete")
        config["last_measurement"] = measurement_id
        save_config(config, args.config_file)
        print("Use 'python dfxpydemo.py meas get' to get comprehensive results")


def load_config(config_file):
    config = {
        "device_id": "",
        "device_token": "",
        "role_id": "",
        "user_id": "",
        "user_token": "",
        "selected_study": "",
        "last_measurement": "",
        "study_cfg_hash": "",
        "study_cfg_data": "",
    }
    if os.path.isfile(config_file):
        with open(config_file, "r") as c:
            read_config = json.loads(c.read())
            config = {**config, **read_config}

    dfxapi.Settings.device_id = config["device_id"]
    dfxapi.Settings.device_token = config["device_token"]
    dfxapi.Settings.role_id = config["role_id"]
    dfxapi.Settings.role_id = config["role_id"]
    dfxapi.Settings.user_token = config["user_token"]

    return config


def save_config(config, config_file):
    with open(config_file, "w") as c:
        c.write(json.dumps(config, indent=4))
        print(f"Config updated in {config_file}")


def print_result(result: dfxsdk.MeasurementResult):
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
        dict_result[k] = str(sum(value) / len(value))

    print_pretty(dict_result, indent=2)


def generate_reqid():
    return "".join(random.choices(string.ascii_letters, k=10))


def determine_action(chunk_number, number_chunks):
    action = 'CHUNK::PROCESS'
    if chunk_number == 0 and number_chunks > 1:
        action = 'FIRST::PROCESS'
    elif chunk_number == number_chunks - 1:
        action = 'LAST::PROCESS'
    return action


async def register(config, config_file, license_key):
    if dfxapi.Settings.device_token:
        print("Already registered")
        return False

    # TODO: Handle 404 properly here...
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        await dfxapi.Organizations.register_license(session, license_key, "LINUX", "DFX Example", "DFXCLIENT", "0.0.1")
        config["device_id"] = dfxapi.Settings.device_id
        config["device_token"] = dfxapi.Settings.device_token
        config["role_id"] = dfxapi.Settings.role_id
        config["user_token"] = dfxapi.Settings.user_token
        print(f"Register successful with new device id {config['device_id']}")
    return True


async def unregister(config, config_file):
    if not dfxapi.Settings.device_token:
        print("Not registered")
        return False

    headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        await dfxapi.Organizations.unregister_license(session)
        print(f"Unregister successful for device id {config['device_id']}")
        config["device_id"] = ""
        config["device_token"] = ""
        config["role_id"] = ""
    return True


async def login(config, config_file, email, password):
    if dfxapi.Settings.user_token:
        print("Already logged in")
        return False

    if not dfxapi.Settings.device_token:
        print("Please register first to obtain a device_token")
        return False

    headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        await dfxapi.Users.login(session, email, password)
        config["user_token"] = dfxapi.Settings.user_token
        print("Login successful")
    return True


def logout(config, config_file):
    config["user_token"] = ""
    config["user_id"] = ""
    print("Logout successful")
    return True


async def measure_websocket(session, measurement_id, measurement_files, number_chunks):
    # Use the session to connect to the WebSocket
    async with session.ws_connect(dfxapi.Settings.ws_url) as ws:
        # Subscribe to results
        results_request_id = generate_reqid()
        await dfxapi.Measurements.ws_subscribe_to_results(ws, results_request_id, measurement_id)

        # Use this to close WebSocket in the receive loop
        results_expected = number_chunks + 1

        async def send_chunks():
            # Coroutine to iterate through the payload files and send chunks using WebSocket
            for payload_file, meta_file, prop_file in measurement_files:
                with open(payload_file, 'rb') as p, open(meta_file, 'rb') as m, open(prop_file, 'r') as pr:
                    payload_bytes = p.read()
                    meta_bytes = m.read()
                    props = json.load(pr)

                    # Determine action and request id
                    action = determine_action(props["chunk_number"], props["number_chunks"])
                    request_id = generate_reqid()

                    # Add data
                    await dfxapi.Measurements.ws_add_data(ws, generate_reqid(), measurement_id, props["chunk_number"],
                                                          action, props["start_time_s"], props["end_time_s"],
                                                          props["duration_s"], meta_bytes, payload_bytes)
                    sleep_time = props["duration_s"]
                    print(
                        f"Sent chunk req#:{request_id} - {action} ...waiting {sleep_time:.0f} seconds instead of {props['duration_s']:.0f}..."
                    )

                    # Sleep to simulate a live measurement and not hit the rate limit
                    await asyncio.sleep(sleep_time)

        async def receive_results():
            # Coroutine to receive results
            num_results_received = 0
            async for msg in ws:
                status, request_id, payload = dfxapi.Measurements.ws_decode(msg)
                if request_id == results_request_id:
                    print(f"  Received result - {len(payload)} bytes {payload[:80]}")
                    num_results_received += 1
                if num_results_received == results_expected:
                    await ws.close()
                    break

        # Start the two coroutines and await till they finish
        await asyncio.gather(send_chunks(), receive_results())


def cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./config.json")
    pp_group = parser.add_mutually_exclusive_group()
    pp_group.add_argument("--json", help="Print as JSON", action="store_true", default=False)
    pp_group.add_argument("--csv", help="Print grids as CSV", action="store_true", default=False)

    subparser_top = parser.add_subparsers(dest="command", required=True)
    subparser_orgs = subparser_top.add_parser("orgs", help="Organizations").add_subparsers(dest="subcommand",
                                                                                           required=True)
    register_parser = subparser_orgs.add_parser("register", help="Register device")
    register_parser.add_argument("license_key", help="DFX API Organization License")
    unregister_parser = subparser_orgs.add_parser("unregister", help="Unregister device")

    subparser_users = subparser_top.add_parser("users", help="Users").add_subparsers(dest="subcommand", required=True)
    login_parser = subparser_users.add_parser("login", help="User login")
    login_parser.add_argument("email", help="Email address")
    login_parser.add_argument("password", help="Password")
    logout_parser = subparser_users.add_parser("logout", help="User logout")

    subparser_studies = subparser_top.add_parser("studies", help="Studies").add_subparsers(dest="subcommand",
                                                                                           required=True)
    study_list_parser = subparser_studies.add_parser("list", help="List existing studies")
    study_get_parser = subparser_studies.add_parser("get", help="Retrieve a study's information")
    study_get_parser.add_argument("study_id",
                                  nargs="?",
                                  help="ID of study to retrieve (default: selected study)",
                                  type=str)
    study_list_parser = subparser_studies.add_parser("select", help="Select a study to use")
    study_list_parser.add_argument("study_id", help="ID of study to use", type=str)

    subparser_meas = subparser_top.add_parser("meas", help="Measurements").add_subparsers(dest="subcommand",
                                                                                          required=True)
    make_parser = subparser_meas.add_parser("make", help="Make a measurement")
    make_parser.add_argument("video_path", help="Path to video file", type=str)
    make_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.01)
    make_parser.add_argument("--no_render", help="Disable video rendering", action="store_true", default=False)
    make_parser.add_argument("--debug_save_chunks_path", help="Save SDK chunks to folder", type=str, default=None)
    list_parser = subparser_meas.add_parser("list", help="List existing measurements")
    list_parser.add_argument("--limit", help="Number of measurements to retrieve (default : 10)", type=int, default=10)
    get_parser = subparser_meas.add_parser("get", help="Retrieve a measurement")
    get_parser.add_argument("measurement_id",
                            nargs="?",
                            help="ID of measurement to retrieve (default: last measurement)",
                            default=None)

    args = parser.parse_args()

    # asyncio.run(main(args))  # https://github.com/aio-libs/aiohttp/issues/4324

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    loop.run_until_complete(asyncio.sleep(0.25))  # https://github.com/aio-libs/aiohttp/issues/1925
    loop.close()


if __name__ == '__main__':
    cmdline()
