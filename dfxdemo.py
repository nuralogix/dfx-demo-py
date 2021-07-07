import argparse
import asyncio
import base64
import copy
import glob
import json
import math
import os.path
import random
import string

import aiohttp
import cv2
import dfx_apiv2_client as dfxapi
import libdfx as dfxsdk
import pkg_resources

from dfxutils.app import AppState, MeasurementStep
from dfxutils.dlib_tracker import DlibTracker
from dfxutils.opencvhelpers import CameraReader, VideoReader
from dfxutils.prettyprint import PrettyPrinter as PP
from dfxutils.renderer import NullRenderer, Renderer
from dfxutils.sdkhelpers import DfxSdkHelpers

try:
    _version = f"v{pkg_resources.require('dfxdemo')[0].version}"
except Exception:
    _version = ""


async def main(args):
    # Load config
    config = load_config(args.config_file)

    # Check API status
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        _, api_status = await dfxapi.General.api_status(session)
        if not api_status["StatusID"] == "ACTIVE":
            print(f"DFX API Status: {api_status['StatusID']} ({dfxapi.Settings.rest_url})")

            return

    # Handle various command line subcommands

    # Handle "orgs" (Organizations) commands - "register" and "unregister"
    if args.command in ["o", "org", "orgs"]:
        if args.subcommand == "unregister":
            success = await unregister(config)
        else:
            success = await register(config, args.license_key)

        if success:
            save_config(config, args.config_file)
        return

    # Handle "users" commands - "login" and "logout"
    if args.command in ["u", "user", "users"]:
        if args.subcommand == "logout":
            success = logout(config)
        else:
            success = await login(config, args.email, args.password)

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

    # Handle "profiles" commands - "create", "update", "remove", "get" and "list"
    if args.command in ["p", "profile", "profiles"]:
        async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
            if args.subcommand == "create":
                _, profile_id = await dfxapi.Profiles.create(session, args.name, args.email)
                print(json.dumps(profile_id)) if args.json else PP.print_pretty(profile_id, args.csv)
            elif args.subcommand == "update":
                _, body = await dfxapi.Profiles.update(session, args.profile_id, args.name, args.email, args.status)
                print(json.dumps(body)) if args.json else PP.print_pretty(body, args.csv)
            elif args.subcommand == "remove":
                _, body = await dfxapi.Profiles.delete(session, args.profile_id)
                print(json.dumps(body)) if args.json else PP.print_pretty(body, args.csv)
            elif args.subcommand == "get":
                _, profile = await dfxapi.Profiles.retrieve(session, args.profile_id)
                print(json.dumps(profile)) if args.json else PP.print_pretty(profile, args.csv)
            elif args.subcommand == "list":
                _, profile_list = await dfxapi.Profiles.list(session)
                print(json.dumps(profile_list)) if args.json else PP.print_pretty(profile_list, args.csv)
        return

    # Handle "studies" commands - "get", "list" and "select"
    if args.command in ["s", "study", "studies"]:
        async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
            if args.subcommand == "get":
                study_id = config["selected_study"] if args.study_id is None else args.study_id
                if not study_id or study_id.isspace():
                    print("Please select a study or pass a study id")
                    return
                _, study = await dfxapi.Studies.retrieve(session, study_id, raise_for_status=False)
                print(json.dumps(study)) if args.json else PP.print_pretty(study, args.csv)
            elif args.subcommand == "list":
                _, studies = await dfxapi.Studies.list(session)
                print(json.dumps(studies)) if args.json else PP.print_pretty(studies, args.csv)
            elif args.subcommand == "select":
                status, response = await dfxapi.Studies.retrieve(session, args.study_id, raise_for_status=False)
                if status >= 400:
                    PP.print_pretty(response)
                    return
                config["selected_study"] = args.study_id
                save_config(config, args.config_file)
        return

    # Handle "measure" (Measurements) commands - "get" and "list"
    if args.command in ["m", "measure", "measurements"] and "make" not in args.subcommand:
        async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
            if args.subcommand == "get":
                measurement_id = config["last_measurement"] if args.measurement_id is None else args.measurement_id
                if not measurement_id or measurement_id.isspace():
                    print("Please complete a measurement first or pass a measurement id")
                    return
                _, results = await dfxapi.Measurements.retrieve(session, measurement_id)
                print(json.dumps(results)) if args.json else PP.print_result(results, args.csv)
            elif args.subcommand == "list":
                _, measurements = await dfxapi.Measurements.list(session,
                                                                 limit=args.limit,
                                                                 user_profile_id=args.profile_id,
                                                                 partner_id=args.partner_id)
                print(json.dumps(measurements)) if args.json else PP.print_pretty(measurements, args.csv)
        return

    # Handle "measure" (Measurements) commands - "make" and "debug_make_from_chunks"
    assert args.command in ["m", "measure", "measurements"] and "make" in args.subcommand

    # Verify preconditions
    # 1. Make sure a study is selected
    if not config["selected_study"]:
        print("Please select a study first using 'study select'")
        return

    # 2. Check if the profile_id if selected, actually exists
    if args.profile_id != "":
        try:
            async with aiohttp.ClientSession(headers=headers, raise_for_status=False) as session:
                status, body = await dfxapi.Profiles.retrieve(session, args.profile_id)
                if status >= 400:
                    print(f"Could not verify that profile ${args.profile_id} exists")
                    print(body)
                    return
        except Exception as e:  # This try catch code path can be removed once the bug on the API is fixed
            print(f"Could not verify that profile ${args.profile_id} exists")
            print(e)
            return

    # Prepare to make a measurement..
    app = AppState()

    if args.subcommand == "make" or args.subcommand == "make_camera":
        # ..using a video or camera
        app.is_camera = args.subcommand == "make_camera"
        image_src_name = f"Camera {args.camera}" if app.is_camera else os.path.basename(args.video_path)
        try:
            # Open the camera or video
            imreader = CameraReader(args.camera, mirror=True) if app.is_camera else VideoReader(
                args.video_path, args.start_time, args.end_time, rotation=args.rotation, fps=args.fps)

            # Open the demographics file if provided
            if args.demographics is not None:
                with open(args.demographics, "r") as f:
                    app.demographics = json.load(f)

            # Create a face tracker
            tracker = DlibTracker()

            # Create DFX SDK factory
            factory = dfxsdk.Factory()
            print("Created DFX Factory:", factory.getVersion())
            sdk_id = factory.getSdkId()

            # Get study config data..
            if args.debug_study_cfg_file is None:
                # ..from API required to initialize DFX SDK collector (or FAIL)
                study_cfg_bytes = await retrieve_sdk_config(headers, config, args.config_file, sdk_id)
            else:
                # .. or from a file
                with open(args.debug_study_cfg_file, 'rb') as f:
                    study_cfg_bytes = f.read()
        except Exception as e:
            print(e)
            return

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
        frames_per_chunk = math.ceil(chunk_duration_s * imreader.fps)
        if app.is_camera:
            app.number_chunks = math.ceil(args.measurement_duration_s / args.chunk_duration_s)
            app.end_frame = math.ceil(args.measurement_duration_s * imreader.fps)
        else:
            app.number_chunks = math.ceil(imreader.frames_to_process / frames_per_chunk)
            app.begin_frame = imreader.start_frame
            app.end_frame = imreader.stop_frame

        # Set collector config
        collector.setTargetFPS(imreader.fps)
        collector.setChunkDurationSeconds(chunk_duration_s)
        collector.setNumberChunks(app.number_chunks)
        print(f"    mode: {factory.getMode()}")
        print(f"    number chunks: {collector.getNumberChunks()}")
        print(f"    chunk duration: {collector.getChunkDurationSeconds()}s")
        for constraint in collector.getEnabledConstraints():
            print(f"    enabled constraint: {constraint}")

        # Set the demographics
        if app.demographics is not None:
            print("    Setting user demographics:")
            for k, v in app.demographics.items():
                collector.setProperty(f"{k}:1", str(v))  # The :1 is because we only care about one face in this demo
                print(f"       {k}: {v}")

        # Set the collector constraints config
        if app.is_camera:
            app.constraints_cfg = DfxSdkHelpers.ConstraintsConfig(collector.getConstraintsConfig("json"))
            app.constraints_cfg.minimumFps = 10
            collector.setConstraintsConfig("json", str(app.constraints_cfg))

    elif args.subcommand == "debug_make_from_chunks":
        # .. or using previously saved chunks
        if not args.alt:
            payload_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "payload*.bin")))
            meta_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "metadata*.bin")))
            prop_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "properties*.json")))
            number_files = min(len(payload_files), len(meta_files), len(prop_files))
            if number_files <= 0:
                print(f"No payload files found in {args.debug_chunks_folder}")
                return
            with open(prop_files[0], 'r') as pr:
                props = json.load(pr)
                app.number_chunks = props["number_chunks"]
                duration_pr = props["duration_s"]
            if app.number_chunks != number_files:
                print(
                    f"Number of chunks in properties.json {app.number_chunks} != Number of payload files {number_files}"
                )
                return
            if duration_pr * app.number_chunks > 120:
                print(f"Total payload duration {duration_pr * app.number_chunks} seconds is more than 120 seconds")
                return
        else:
            payload_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "payload*.bin")))
            meta_files = sorted(glob.glob(os.path.join(args.debug_chunks_folder, "meta*.json")))
            prop_files = meta_files
            number_files = min(len(payload_files), len(meta_files), len(prop_files))
            if number_files <= 0:
                print(f"No payload files found in {args.debug_chunks_folder}")
                return
            app.number_chunks = number_files

            # Guess the duration
            duration_pr = None
            with open(prop_files[0], 'r') as pr:
                props = json.load(pr)
                if "ChunkEndFrameTime_ms" in props:
                    with open(prop_files[1], 'r') as pr2:
                        props2 = json.load(pr2)
                        duration_pr = (props2["ChunkEndFrameTime_ms"] - props["ChunkEndFrameTime_ms"]) / 1000.0
            # but allow duration override...
            if args.chunk_duration_s is not None:
                duration_pr = args.chunk_duration_s
            if duration_pr is None:
                print("Could not determine chunk duration from files. Please use the --chunk_duration option")
                return
            if duration_pr * app.number_chunks > 120:
                print(f"Total payload duration {duration_pr * app.number_chunks} seconds is more than 120 seconds")
                return
            app.chunk_duration_s = int(duration_pr)

        # Create DFX SDK factory (just so we can have a collector for decoding results)
        factory = dfxsdk.Factory()
        print("Created DFX Factory:", factory.getVersion())
        collector = factory.createCollector()
        print("Created DFX Collector for results decoding only")

        app.step = MeasurementStep.USER_STARTED
    else:
        print("Unknown subcommand to 'meas'. This should never happen")
        return

    # Make a measurement
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        # Create a measurement on the API and get the measurement ID
        _, response = await dfxapi.Measurements.create(session,
                                                       config["selected_study"],
                                                       user_profile_id=args.profile_id,
                                                       partner_id=args.partner_id)
        app.measurement_id = response["ID"]
        print(f"Created measurement {app.measurement_id}")

        # Use the session to connect to the WebSocket
        async with session.ws_connect(dfxapi.Settings.ws_url) as ws:
            # Subscribe to results
            results_request_id = generate_reqid()
            await dfxapi.Measurements.ws_subscribe_to_results(ws, generate_reqid(), app.measurement_id,
                                                              results_request_id)

            # Queue to pass chunks between coroutines
            chunk_queue = asyncio.Queue(app.number_chunks)

            # When we receive the last chunk from the SDK, we can check for measurement completion
            app.last_chunk_sent = False

            # Coroutine to produce chunks and put then in chunk_queue
            if args.subcommand == "make" or args.subcommand == "make_camera":
                renderer = Renderer(
                    _version,
                    image_src_name,
                    imreader.fps,
                    app,
                    0.5 if imreader.height >= 720 else 1.0,
                ) if app.is_camera or not args.no_render else NullRenderer()
                if not app.is_camera:
                    print("Extraction started")
                else:
                    print("Waiting to start")
                produce_chunks_coro = extract_from_imgs(
                    chunk_queue,  # Chunks will be put into this queue
                    imreader,  # Image reader
                    tracker,  # Face tracker
                    collector,  # DFX SDK collector needed to create chunks
                    renderer,  # Rendering
                    app)  # App
            else:  # args.subcommand == "debug_make_from_chunks":
                renderer = NullRenderer()
                produce_chunks_coro = read_folder_chunks(chunk_queue, payload_files, meta_files, prop_files, args.alt,
                                                         app)

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
                    await dfxapi.Measurements.ws_add_data(ws, generate_reqid(), app.measurement_id, chunk.chunk_number,
                                                          action, chunk.start_time_s, chunk.end_time_s,
                                                          chunk.duration_s, chunk.metadata, chunk.payload_data)
                    print(f"Sent chunk {chunk.chunk_number}")
                    renderer.set_sent(chunk.chunk_number)

                    # Update data needed to check for completion
                    app.number_chunks_sent += 1
                    app.last_chunk_sent = action == 'LAST::PROCESS'

                    # Save chunk (for debugging purposes)
                    if "debug_save_chunks_folder" in args and args.debug_save_chunks_folder:
                        DfxSdkHelpers.save_chunk(copy.copy(chunk), args.debug_save_chunks_folder)
                        print(f"Saved chunk {chunk.chunk_number} in '{args.debug_save_chunks_folder}'")

                    chunk_queue.task_done()

                app.step = MeasurementStep.WAITING_RESULTS
                print("Extraction complete, waiting for results")

            # Coroutine to receive responses using the Websocket
            async def receive_results():
                num_results_received = 0
                async for msg in ws:
                    status, request_id, payload = dfxapi.Measurements.ws_decode(msg)
                    if request_id == results_request_id:
                        sdk_result = collector.decodeMeasurementResult(payload)
                        result = DfxSdkHelpers.sdk_result_to_dict(sdk_result)
                        renderer.set_results(result.copy())
                        PP.print_sdk_result(result)
                        num_results_received += 1
                    # We are done if the last chunk is sent and number of results received equals number of chunks sent
                    if app.last_chunk_sent and num_results_received == app.number_chunks_sent:
                        await ws.close()
                        break

                app.step = MeasurementStep.COMPLETED
                print("Measurement complete")

            # Coroutine for rendering
            async def render():
                if type(renderer) == NullRenderer:
                    return

                cancelled = await renderer.render()
                cv2.destroyAllWindows()
                if cancelled:
                    tracker.stop()
                    raise ValueError("Measurement was cancelled by user.")

            # Wrap the coroutines in tasks, start them and wait till they finish
            tasks = [
                asyncio.create_task(produce_chunks_coro),
                asyncio.create_task(send_chunks()),
                asyncio.create_task(receive_results()),
                asyncio.create_task(render())
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for p in pending:  # If there were any pending coroutines, cancel them here...
                p.cancel()
            if len(pending) > 0:  # If we had pending coroutines, it means something went wrong in the 'done' ones
                for d in done:
                    e = d.exception()
                    if e is not None and type(e) != asyncio.CancelledError:
                        print(e)
                print(f"Measurement {app.measurement_id} failed")
            else:
                config["last_measurement"] = app.measurement_id
                save_config(config, args.config_file)
                print(f"Measurement {app.measurement_id} completed")
                print(f"Use 'python {os.path.basename(__file__)} measure get' to get comprehensive results")


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
    if "rest_url" in config and config["rest_url"]:
        dfxapi.Settings.rest_url = config["rest_url"]
    if "ws_url" in config and config["ws_url"]:
        dfxapi.Settings.ws_url = config["ws_url"]

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


async def register(config, license_key):
    if dfxapi.Settings.device_token:
        print("Already registered")
        return False

    async with aiohttp.ClientSession() as session:
        status, body = await dfxapi.Organizations.register_license(session, license_key, "LINUX", "DFX Example",
                                                                   "DFXCLIENT", "0.0.1")
        if status < 400:
            config["device_id"] = dfxapi.Settings.device_id
            config["device_token"] = dfxapi.Settings.device_token
            config["role_id"] = dfxapi.Settings.role_id
            config["user_token"] = dfxapi.Settings.user_token
            print(f"Register successful with new device id {config['device_id']}")
            return True
        else:
            print(f"Register failed {status}: {body}")
            return False


async def unregister(config):
    if not dfxapi.Settings.device_token:
        print("Not registered")
        return False

    headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        status, body = await dfxapi.Organizations.unregister_license(session)
        if status < 400:
            print(f"Unregister successful for device id {config['device_id']}")
            config["device_id"] = ""
            config["device_token"] = ""
            config["role_id"] = ""
            return True
        else:
            print(f"Unregister failed {status}: {body}")


async def login(config, email, password):
    if dfxapi.Settings.user_token:
        print("Already logged in")
        return False

    if not dfxapi.Settings.device_token:
        print("Please register first to obtain a device_token")
        return False

    headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        status, body = await dfxapi.Users.login(session, email, password)
        if status < 400:
            config["user_token"] = dfxapi.Settings.user_token
            print("Login successful")
            return True
        else:
            print(f"Login failed {status}: {body}")
            return False


def logout(config):
    config["user_token"] = ""
    config["user_id"] = ""
    print("Logout successful")
    return True


async def retrieve_sdk_config(headers, config, config_file, sdk_id):
    async with aiohttp.ClientSession(headers=headers) as session:
        status, response = await dfxapi.Studies.retrieve_sdk_config_data(session, config["selected_study"], sdk_id,
                                                                         config["study_cfg_hash"])
        if status == 304:  # Our hash and data are already correct nothing to do
            pass
        elif status == 200:  # Got a new hash and data
            config["study_cfg_hash"] = response["MD5Hash"]
            config["study_cfg_data"] = response["ConfigFile"]
            print(f"Retrieved new DFX SDK config data with md5: {config['study_cfg_hash']}")
            save_config(config, config_file)
        else:
            raise RuntimeError(f"Could not retrieve DFX SDK config data for Study ID {config['selected_study']}. "
                               "Please contact Nuralogix")

        return base64.standard_b64decode(config["study_cfg_data"])


async def extract_from_imgs(chunk_queue, imreader, tracker, collector, renderer, app):
    # Read frames from the image source, track faces and extract using collector
    while True:
        # Grab a frame
        read, image, frame_number, frame_timestamp_ns = await imreader.read_next_frame()
        if not read or image is None:
            # Video ended, so grab what should be the last, possibly truncated chunk
            collector.forceComplete()
            chunk_data = collector.getChunkData()
            if chunk_data is not None:
                chunk = chunk_data.getChunkPayload()
                await chunk_queue.put(chunk)
                break

        # Start the DFX SDK collection if we received a start command
        if app.step == MeasurementStep.USER_STARTED:
            collector.startCollection()
            app.step = MeasurementStep.MEASURING
            if app.is_camera:
                app.begin_frame = frame_number
                app.end_frame = frame_number + app.end_frame

        # Track faces
        tracked_faces = tracker.trackFaces(image, frame_number, frame_timestamp_ns / 1000000.0)

        # Create a DFX VideoFrame, then a DFX Frame from the DFX VideoFrame and add DFX faces to it
        dfx_video_frame = dfxsdk.VideoFrame(image, frame_number, frame_timestamp_ns,
                                            dfxsdk.ChannelOrder.CHANNEL_ORDER_BGR)
        dfx_frame = collector.createFrame(dfx_video_frame)
        if len(tracked_faces) > 0:
            tracked_face = next(iter(tracked_faces.values()))  # We only care about the first face in this demo
            dfx_face = DfxSdkHelpers.dfx_face_from_json(collector, tracked_face)
            dfx_frame.addFace(dfx_face)

        # For cameras, check constraints and provide users actionable feedback
        if app.is_camera:
            c_result, c_details = collector.checkConstraints(dfx_frame)

            # Change renderer state
            renderer.set_constraints_feedback(DfxSdkHelpers.user_feedback_from_constraints(c_details))

            # Change the app step
            if app.step in [MeasurementStep.NOT_READY, MeasurementStep.READY]:
                if c_result == dfxsdk.ConstraintResult.GOOD:
                    app.step = MeasurementStep.READY
                else:
                    app.step = MeasurementStep.NOT_READY
            elif app.step == MeasurementStep.MEASURING:
                if c_result == dfxsdk.ConstraintResult.ERROR:
                    app.step = MeasurementStep.FAILED
                    reasons = DfxSdkHelpers.failure_causes_from_constraints(c_details)
                    print(reasons)
        else:
            if app.step == MeasurementStep.NOT_READY and len(tracked_faces) > 0:
                app.step = MeasurementStep.USER_STARTED

        # Extract bloodflow if the measurement has started
        if app.step == MeasurementStep.MEASURING:
            collector.defineRegions(dfx_frame)
            result = collector.extractChannels(dfx_frame)

            # Grab a chunk and check if we are finished
            if result == dfxsdk.CollectorState.CHUNKREADY or result == dfxsdk.CollectorState.COMPLETED:
                chunk_data = collector.getChunkData()
                if chunk_data is not None:
                    chunk = chunk_data.getChunkPayload()
                    await chunk_queue.put(chunk)
                if result == dfxsdk.CollectorState.COMPLETED:
                    if app.is_camera:
                        imreader.stop()
                    break

        await renderer.put_nowait((image, (dfx_frame, frame_number, frame_timestamp_ns)))

    # Stop the tracker
    tracker.stop()

    # Close the camera
    imreader.close()

    # Signal to send_chunks that we are done
    await chunk_queue.put(None)

    # Signal to render_queue that we are done
    renderer.keep_render_last_frame()


async def read_folder_chunks(chunk_queue, payload_files, meta_files, prop_files, alt, app):
    for i, (payload_file, meta_file, prop_file) in enumerate(zip(payload_files, meta_files, prop_files)):
        with open(payload_file, 'rb') as p, open(meta_file, 'rb') as m, open(prop_file, 'r') as pr:
            payload_bytes = p.read()
            meta_bytes = m.read()
            props = json.load(pr)

            chunk = dfxsdk.Payload()
            chunk.number_payload_bytes = len(payload_bytes)
            chunk.payload_data = payload_bytes
            chunk.number_metadata_bytes = len(meta_bytes)
            chunk.metadata = meta_bytes
            if not alt:
                chunk.valid = props["valid"]
                chunk.start_frame = props["start_frame"]
                chunk.end_frame = props["end_frame"]
                chunk.chunk_number = props["chunk_number"]
                chunk.number_chunks = props["number_chunks"]
                chunk.first_chunk_start_time_s = props["first_chunk_start_time_s"]
                chunk.start_time_s = props["start_time_s"]
                chunk.end_time_s = props["end_time_s"]
                chunk.duration_s = props["duration_s"]
            else:
                chunk.valid = 1
                chunk.chunk_number = i
                chunk.number_chunks = app.number_chunks
                chunk.duration_s = app.chunk_duration_s
                if "ChunkEndFrameTime_ms" in props:
                    chunk.end_time_s = int(props["ChunkEndFrameTime_ms"] / 1000.0)
                    chunk.start_time_s = int(chunk.end_time_s - chunk.duration_s)
                elif type(props["StartTime"]) in [int, float]:
                    chunk.start_time_s = int(props["StartTime"])
                    chunk.end_time_s = int(chunk.start_time_s + chunk.duration_s)
                elif type(props["StartTime"]) == dict:  # TODO: Verify
                    chunk.start_time_s = props["StartTime"]["low"]
                    chunk.end_time_s = props["EndTime"]["low"]
            await chunk_queue.put(chunk)

            # Sleep to simulate a live measurement and not hit the rate limit
            sleep_time = chunk.duration_s
            await asyncio.sleep(sleep_time)

    await chunk_queue.put(None)


def cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v",
                        "--version",
                        action="version",
                        version=f"%(prog)s {_version} (libdfx v{dfxsdk.__version__})")
    parser.add_argument("-c", "--config_file", help="Path to config file", default="./config.json")
    pp_group = parser.add_mutually_exclusive_group()
    pp_group.add_argument("--json", help="Print as JSON", action="store_true", default=False)
    pp_group.add_argument("--csv", help="Print grids as CSV", action="store_true", default=False)

    subparser_top = parser.add_subparsers(dest="command", required=True)
    subparser_orgs = subparser_top.add_parser("orgs", aliases=["o", "org"],
                                              help="Organizations").add_subparsers(dest="subcommand", required=True)
    register_parser = subparser_orgs.add_parser("register", help="Register device")
    register_parser.add_argument("license_key", help="DFX API Organization License")
    unregister_parser = subparser_orgs.add_parser("unregister", help="Unregister device")

    subparser_users = subparser_top.add_parser("users", aliases=["u", "user"],
                                               help="Users").add_subparsers(dest="subcommand", required=True)
    login_parser = subparser_users.add_parser("login", help="User login")
    login_parser.add_argument("email", help="Email address")
    login_parser.add_argument("password", help="Password")
    logout_parser = subparser_users.add_parser("logout", help="User logout")

    subparser_profiles = subparser_top.add_parser("profiles", aliases=["p", "profile"],
                                                  help="Profiles").add_subparsers(dest="subcommand", required=True)
    profile_create_parser = subparser_profiles.add_parser("create", help="Create profile")
    profile_create_parser.add_argument("name", help="Name (unique)", type=str)
    profile_create_parser.add_argument("--email", help="Email", type=str, default="no_email_provided")
    profile_update_parser = subparser_profiles.add_parser("update", help="Update profile")
    profile_update_parser.add_argument("profile_id", help="Profile ID to update", type=str)
    profile_update_parser.add_argument("name", help="New Name", type=str, default="")
    profile_update_parser.add_argument("email", help="New Email", type=str, default="")
    profile_update_parser.add_argument("status", help="New Status", type=str, default="")
    profile_remove_parser = subparser_profiles.add_parser("remove", help="Remove profile")
    profile_remove_parser.add_argument("profile_id", help="Profile ID to remove", type=str)
    profile_get_parser = subparser_profiles.add_parser("get", help="Retrieve profile")
    profile_get_parser.add_argument("profile_id", help="Profile ID to retrieve", type=str)
    profile_list_parser = subparser_profiles.add_parser("list", help="List profiles")

    subparser_studies = subparser_top.add_parser("studies", aliases=["s", "study"],
                                                 help="Studies").add_subparsers(dest="subcommand", required=True)
    study_list_parser = subparser_studies.add_parser("list", help="List existing studies")
    study_get_parser = subparser_studies.add_parser("get", help="Retrieve a study's information")
    study_get_parser.add_argument("study_id",
                                  nargs="?",
                                  help="ID of study to retrieve (default: selected study)",
                                  type=str)
    study_list_parser = subparser_studies.add_parser("select", help="Select a study to use")
    study_list_parser.add_argument("study_id", help="ID of study to use", type=str)

    subparser_meas = subparser_top.add_parser("measure", aliases=["m", "measurements"],
                                              help="Measurements").add_subparsers(dest="subcommand", required=True)
    list_parser = subparser_meas.add_parser("list", help="List existing measurements")
    list_parser.add_argument("--limit", help="Number of measurements to retrieve (default : 10)", type=int, default=10)
    list_parser.add_argument("--profile_id", help="Filter list by Profile ID", type=str, default="")
    list_parser.add_argument("--partner_id", help="Filter list by Partner ID", type=str, default="")
    get_parser = subparser_meas.add_parser("get", help="Retrieve a measurement")
    get_parser.add_argument("measurement_id",
                            nargs="?",
                            help="ID of measurement to retrieve (default: last measurement)",
                            default=None)
    make_parser = subparser_meas.add_parser("make", help="Make a measurement from a video file")
    make_parser.add_argument("video_path", help="Path to video file", type=str)
    make_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.0)
    make_parser.add_argument("-t", "--start_time", help="Video segment start time (seconds)", type=float, default=None)
    make_parser.add_argument("-T", "--end_time", help="Video segment end time (seconds)", type=float, default=None)
    make_parser.add_argument("--fps",
                             help="Use this framerate instead of detecting from video",
                             type=float,
                             default=None)
    make_parser.add_argument("--rotation",
                             help="Use this rotation instead of detecting from video (Must be 0, 90, 180 or 270)",
                             type=float,
                             default=None)
    make_parser.add_argument("--no_render", help="Disable video rendering", action="store_true", default=False)
    make_parser.add_argument("--profile_id", help="Set the Profile ID (Participant ID)", type=str, default="")
    make_parser.add_argument("--partner_id", help="Set the PartnerID", type=str, default="")
    make_parser.add_argument("-dg",
                             "--demographics",
                             help="Path to JSON file containing user demographics",
                             default=None)
    make_parser.add_argument("--debug_study_cfg_file",
                             help="Study config file to use instead of data from API (debugging)",
                             type=str,
                             default=None)
    make_parser.add_argument("--debug_save_chunks_folder",
                             help="Save SDK chunks to folder (debugging)",
                             type=str,
                             default=None)

    camera_parser = subparser_meas.add_parser("make_camera", help="Make a measurement from a camera")
    camera_parser.add_argument("--camera", help="Camera ID", type=int, default=0)
    camera_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.0)
    camera_parser.add_argument("-md",
                               "--measurement_duration_s",
                               help="Measurement duration (seconds)",
                               type=float,
                               default=30)
    camera_parser.add_argument("--profile_id", help="Set the Profile ID (Participant ID)", type=str, default="")
    camera_parser.add_argument("--partner_id", help="Set the PartnerID", type=str, default="")
    camera_parser.add_argument("-dg",
                               "--demographics",
                               help="Path to JSON file containing user demographics",
                               default=None)
    camera_parser.add_argument("--debug_study_cfg_file",
                               help="Study config file to use instead of data from API (debugging)",
                               type=str,
                               default=None)
    camera_parser.add_argument("--debug_save_chunks_folder",
                               help="Save SDK chunks to folder (debugging)",
                               type=str,
                               default=None)

    mk_ch_parser = subparser_meas.add_parser("debug_make_from_chunks",
                                             help="Make a measurement from saved SDK chunks (debugging)")
    mk_ch_parser.add_argument("debug_chunks_folder", help="Folder containing SDK chunks", type=str)
    mk_ch_parser.add_argument("--alt", help="Alternate payload format", action="store_true", default=False)
    mk_ch_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=None)
    mk_ch_parser.add_argument("--profile_id", help="Set the Profile ID (Participant ID)", type=str, default="")
    mk_ch_parser.add_argument("--partner_id", help="Set the PartnerID", type=str, default="")
    mk_ch_parser.add_argument("-dg",
                              "--demographics",
                              help="Path to JSON file containing user demographics",
                              default=None)
    args = parser.parse_args()

    # asyncio.run(main(args))  # https://github.com/aio-libs/aiohttp/issues/4324

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    loop.run_until_complete(asyncio.sleep(0.25))  # https://github.com/aio-libs/aiohttp/issues/1925
    loop.close()


if __name__ == '__main__':
    cmdline()
