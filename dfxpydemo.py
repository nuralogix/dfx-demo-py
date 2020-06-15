import argparse
import asyncio
import base64
import glob
import json
import math
import os.path
import pkg_resources
import random
import string

import aiohttp
import cv2
import numpy as np

import libdfx as dfxsdk
import dfx_apiv2_client as dfxapi

from dfxpydemoutils import (DlibTracker, Renderer, NullRenderer, dfx_face_from_json, find_video_rotation,
                            sdk_result_to_dict, print_sdk_result, print_meas, print_pretty, read_next_frame, save_chunk)

try:
    _version = f"v{pkg_resources.require('dfxpydemo')[0].version}"
except Exception:
    _version = ""


async def main(args):
    # Load config
    config = load_config(args.config_file)

    # Check API status
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        api_status = await dfxapi.General.api_status(session)
        if not api_status["StatusID"] == "ACTIVE":
            print(f"DFX API Status: {api_status['StatusID']} ({dfxapi.Settings.rest_url})")

            return

    # Handle various command line subcommands

    # Handle "orgs" (Organizations) commands - "register" and "unregister"
    if args.command == "orgs":
        if args.subcommand == "unregister":
            success = await unregister(config, args.config_file)
        else:
            success = await register(config, args.config_file, args.license_key)

        if success:
            save_config(config, args.config_file)
        return

    # Handle "users" commands - "login" and "logout"
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

    # Handle "studies" commands - "get", "list" and "select"
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

    # Handle "meas" (Measurements) commands - "get" and "list"
    if args.command == "meas" and "make" not in args.subcommand:
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

    # Handle "meas" (Measurements) commands - "make" and "debug_make_from_chunks"
    assert args.command == "meas" and "make" in args.subcommand

    # Verify preconditions
    if not config["selected_study"]:
        print("Please select a study first using 'study select'")
        return

    # Prepare for making a measurement
    if args.subcommand == "make":
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

        if args.debug_study_cfg_file is None:
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
        else:
            with open(args.debug_study_cfg_file, 'rb') as f:
                study_cfg_bytes = f.read()

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

    elif args.subcommand == "debug_make_from_chunks":
        payload_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "payload*.bin")))
        meta_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "metadata*.bin")))
        prop_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "properties*.json")))
        number_files = min(len(payload_files), len(meta_files), len(prop_files))
        if number_files <= 0:
            print(f"No payload files found in {args.debug_chunks_folder}")
            return
        with open(prop_files[0], 'r') as pr:
            props = json.load(pr)
            number_chunks = props["number_chunks"]
            duration_pr = props["duration_s"]
        if number_chunks != number_files:
            print(f"Number of chunks in properties.json {number_chunks} != Number of payload files {number_files}")
            return
        if duration_pr * number_chunks > 120:
            print(f"Total payload duration {duration_pr * number_chunks} seconds is more than 120 seconds")
            return

        # Create DFX SDK factory (just so we can have a collector for decoding results)
        factory = dfxsdk.Factory()
        print("Created DFX Factory:", factory.getVersion())
        collector = factory.createCollector()
        print("Created DFX Collector for results decoding only")
    else:
        print("Unknown subcommand to 'meas'. This should never happen")
        return

    # Make a measurement
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        # Create a measurement on the API and get the measurement ID
        measurement_id = await dfxapi.Measurements.create(session, config["selected_study"])
        print(f"Created measurement {measurement_id}")

        # Use the session to connect to the WebSocket
        async with session.ws_connect(dfxapi.Settings.ws_url) as ws:
            # Subscribe to results
            results_request_id = generate_reqid()
            await dfxapi.Measurements.ws_subscribe_to_results(ws, results_request_id, measurement_id)

            # Queue to pass chunks between coroutines
            chunk_queue = asyncio.Queue(number_chunks)

            # When we receive `results_expected` results, we close the WebSocket in the receive loop
            results_expected = number_chunks

            # Coroutine to produce chunks and put then in chunk_queue
            if args.subcommand == "make":
                renderer = Renderer(_version, os.path.basename(args.video_path), frames_to_process, fps, measurement_id,
                                    number_chunks, 0.5) if not args.no_render else NullRenderer()
                renderer.set_message("Press Esc to cancel")
                produce_chunks_coro = extract_video(
                    chunk_queue,  # Chunks will be put into this queue
                    (videocap, fps, rotation, frames_to_process, frame_duration_ns),  # Video capture
                    tracker,  # Face tracker
                    collector,  # DFX SDK collector needed to create chunks
                    renderer  # Rendering
                )
            else:  # args.subcommand == "debug_make_from_chunks":
                produce_chunks_coro = read_folder_chunks(chunk_queue, payload_files, meta_files, prop_files)
                renderer = NullRenderer()

            # Coroutine to get chunks from chunk_queue and send chunk using WebSocket
            async def send_chunks():
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
                    renderer.set_sent(chunk.chunk_number)

                    # Save chunk (for debugging purposes)
                    if "debug_save_chunks_folder" in args and args.debug_save_chunks_folder:
                        save_chunk(chunk, args.debug_save_chunks_folder)
                        print(f"Saved chunk {chunk.chunk_number} in '{args.debug_save_chunks_folder}'")

                    chunk_queue.task_done()

            # Coroutine to receive responses using Websocket
            async def receive_results():
                num_results_received = 0
                async for msg in ws:
                    status, request_id, payload = dfxapi.Measurements.ws_decode(msg)
                    if request_id == results_request_id and len(payload) > 0:
                        sdk_result = collector.decodeMeasurementResult(payload)
                        result = sdk_result_to_dict(sdk_result)
                        renderer.set_results(result.copy())
                        print_sdk_result(result)
                        num_results_received += 1
                    if num_results_received == results_expected:
                        await ws.close()
                        break

            # Coroutine for rendering
            async def render():
                if type(renderer) == NullRenderer:
                    return

                cancelled = await renderer.render()
                if cancelled:
                    tracker.stop()
                    t = asyncio.current_task()
                    t.cancel()
                cv2.destroyAllWindows()

            # Start the three coroutines and await till they finish
            try:
                await asyncio.gather(produce_chunks_coro, send_chunks(), receive_results(), render())
            except Exception as e:
                print(e)
                print(f"Measurement {measurement_id} failed")
                return

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


async def extract_video(chunk_queue, video_opts, tracker, collector, renderer):
    videocap, fps, rotation, frames_to_process, frame_duration_ns = video_opts

    # Start the DFX SDK collection
    collector.startCollection()

    # Read frames from the video, track faces and extract using collector
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

        await renderer.put_nowait((image, (dfx_frame, frame_number)))

    # Stop the tracker
    tracker.stop()

    # Signal to send_chunks that we are done
    await chunk_queue.put(None)

    # Signal to render_queue that we are done
    renderer.keep_render_last_frame()


async def read_folder_chunks(chunk_queue, payload_files, meta_files, prop_files):
    for payload_file, meta_file, prop_file in zip(payload_files, meta_files, prop_files):
        with open(payload_file, 'rb') as p, open(meta_file, 'rb') as m, open(prop_file, 'r') as pr:
            payload_bytes = p.read()
            meta_bytes = m.read()
            props = json.load(pr)

            chunk = dfxsdk.Payload()
            chunk.valid = props["valid"]
            chunk.start_frame = props["start_frame"]
            chunk.end_frame = props["end_frame"]
            chunk.chunk_number = props["chunk_number"]
            chunk.number_chunks = props["number_chunks"]
            chunk.first_chunk_start_time_s = props["first_chunk_start_time_s"]
            chunk.start_time_s = props["start_time_s"]
            chunk.end_time_s = props["end_time_s"]
            chunk.duration_s = props["duration_s"]
            chunk.number_payload_bytes = len(payload_bytes)
            chunk.payload_data = payload_bytes
            chunk.number_metadata_bytes = len(meta_bytes)
            chunk.metadata = meta_bytes

            await chunk_queue.put(chunk)

            # Sleep to simulate a live measurement and not hit the rate limit
            sleep_time = props["duration_s"]
            await asyncio.sleep(sleep_time)

    await chunk_queue.put(None)


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
    list_parser = subparser_meas.add_parser("list", help="List existing measurements")
    list_parser.add_argument("--limit", help="Number of measurements to retrieve (default : 10)", type=int, default=10)
    get_parser = subparser_meas.add_parser("get", help="Retrieve a measurement")
    get_parser.add_argument("measurement_id",
                            nargs="?",
                            help="ID of measurement to retrieve (default: last measurement)",
                            default=None)
    make_parser = subparser_meas.add_parser("make", help="Make a measurement from a video file")
    make_parser.add_argument("video_path", help="Path to video file", type=str)
    make_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.01)
    make_parser.add_argument("--no_render", help="Disable video rendering", action="store_true", default=False)
    make_parser.add_argument("--debug_study_cfg_file",
                             help="Study config file to use instead of data from API (debugging)",
                             type=str,
                             default=None)
    make_parser.add_argument("--debug_save_chunks_folder",
                             help="Save SDK chunks to folder (debugging)",
                             type=str,
                             default=None)
    mk_ch_parser = subparser_meas.add_parser("debug_make_from_chunks",
                                             help="Make a measurement from saved SDK chunks (debugging)")
    mk_ch_parser.add_argument("debug_chunks_folder", help="Folder containing SDK chunks", type=str)
    args = parser.parse_args()

    # asyncio.run(main(args))  # https://github.com/aio-libs/aiohttp/issues/4324

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    loop.run_until_complete(asyncio.sleep(0.25))  # https://github.com/aio-libs/aiohttp/issues/1925
    loop.close()


if __name__ == '__main__':
    cmdline()
