import argparse
import asyncio
import base64
import copy
import json
import math
import os.path
import platform

import cv2
import libdfx as dfxsdk
import pkg_resources

from dfxutils.app import AppState, MeasurementStep
from dfxutils.opencvhelpers import CameraReader, VideoReader
from dfxutils.renderer import NullRenderer, Renderer
from dfxutils.sdkhelpers import DfxSdkHelpers

FT_CHOICES = []
try:
    from dfxutils.visage_tracker import VisageTracker
    FT_CHOICES.append("visage")
except ImportError:
    pass

try:
    from dfxutils.mediapipe_tracker import MediaPipeTracker
    FT_CHOICES.append("mediapipe")
except ImportError:
    pass

try:
    from dfxutils.dlib_tracker import DlibTracker
    FT_CHOICES.append("dlib")
except ImportError:
    pass

if not FT_CHOICES:
    raise ImportError("Could not import any face tracker")

try:
    _version = f"v{pkg_resources.require('dfxdemo')[0].version}"
except Exception:
    _version = ""


async def main(args):
    # Handle various command line subcommands
    assert args.command in ["video", "camera"]

    # Prepare to process a video or camera
    app = AppState()
    app.extract_only = True
    app.is_camera = args.command == "camera"
    app.is_infrared = args.infrared
    app.virtual = args.virtual
    headless = cv2.version.headless or "headless" in args and args.headless
    image_src_name = f"Camera {args.camera}" if app.is_camera else os.path.basename(args.video_path)
    try:
        # Open the camera or video
        width, height, fps = None, None, None
        if app.virtual is not None:
            width, height, fps = map(int, app.virtual.replace('@', 'x').split('x'))
        imreader = CameraReader(args.camera, mirror=True, fps=fps, width=width,
                                height=height) if app.is_camera else VideoReader(
                                    args.video_path,
                                    args.start_time,
                                    args.end_time,
                                    rotation=args.rotation,
                                    fps=args.fps,
                                    use_video_timestamps=args.use_video_timestamps)

        # Open the demographics file if provided
        if args.demographics is not None:
            with open(args.demographics, "r") as f:
                app.demographics = json.load(f)

        # Create a face tracker
        if args.face_tracker == "visage":
            tracker = VisageTracker(args.visage_license,
                                    1,
                                    imreader.width,
                                    imreader.height,
                                    use_analyser=args.analyser,
                                    track_in_background=app.is_camera)
        elif args.face_tracker == "dlib":
            tracker = DlibTracker()
        else:
            tracker = MediaPipeTracker(1, track_in_background=app.is_camera)

        # Create DFX SDK factory
        factory = dfxsdk.Factory()
        print("Created DFX Factory:", factory.getVersion())
        sdk_id = factory.getSdkId()

        # Get study config data from a .dat file or from a dfxdemo config.json file
        with open(args.study_cfg_file, 'rb') as f:
            file_bytes = f.read()
            try:
                # Try to interpret as a config.json
                config = json.loads(file_bytes)
                study_cfg_bytes = base64.standard_b64decode(config["study_cfg_data"])
            except Exception:
                # Just pass along the raw bytes
                study_cfg_bytes = file_bytes
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

    print(f"Face Tracker: {args.face_tracker}")
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

    # Set the demographics
    if app.demographics is not None:
        print("    Setting user demographics:")
        if not DfxSdkHelpers.set_user_demographics(collector, app.demographics):
            print("Failed to set user demographics because " + collector.getLastErrorMessage())
            return

    # Set the collector constraints config
    if app.is_camera:
        app.constraints_cfg = DfxSdkHelpers.ConstraintsConfig(collector.getConstraintsConfig("json"))
        app.constraints_cfg.minimumFps = 10
        collector.setConstraintsConfig("json", str(app.constraints_cfg))

    # Print the enabled constraints
    print("Constraints:")
    for constraint in collector.getEnabledConstraints():
        print(f"    enabled: {constraint}")

    # Extract and save
    # Queue to pass chunks between coroutines
    chunk_queue = asyncio.Queue(app.number_chunks)

    # When we receive the last chunk from the SDK, we can check for measurement completion
    app.last_chunk_sent = False

    # Coroutine to produce chunks and put then in chunk_queue
    renderer = Renderer(
        _version,
        image_src_name,
        imreader.fps,
        app,
        0.5 if imreader.height >= 720 else 1.0,
    ) if app.is_camera or not headless else NullRenderer()
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

    # Coroutine to get chunks from chunk_queue and send chunk using WebSocket
    async def save_chunks():
        while True:
            chunk = await chunk_queue.get()
            if chunk is None:
                chunk_queue.task_done()
                break

            # Update renderer
            renderer.set_sent(chunk.chunk_number)

            # Save chunk (for debugging purposes)
            DfxSdkHelpers.save_chunk(copy.copy(chunk), args.save_chunks_folder)
            print(f"Saved chunk {chunk.chunk_number} in '{args.save_chunks_folder}'")

            chunk_queue.task_done()

        app.step = MeasurementStep.COMPLETED
        print("Saving chunks complete")

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
        asyncio.create_task(save_chunks()),
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
                # raise e  # Uncomment this to see a stack trace
        print(f"Extraction failed")
    else:
        print("Extraction complete")


async def extract_from_imgs(chunk_queue, imreader, tracker, collector, renderer, app):
    # Set channel order based on is_infrared, is_camera and is_virtual
    channelOrder = dfxsdk.ChannelOrder.CHANNEL_ORDER_BGR
    if app.is_infrared:
        if app.is_camera:
            if app.virtual is not None:
                channelOrder = dfxsdk.ChannelOrder.CHANNEL_ORDER_INFRARED888
            else:
                channelOrder = dfxsdk.ChannelOrder.CHANNEL_ORDER_INFRARED
        else:
            channelOrder = dfxsdk.ChannelOrder.CHANNEL_ORDER_INFRARED888

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
        dfx_video_frame = dfxsdk.VideoFrame(image, frame_number, frame_timestamp_ns, channelOrder)
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
            elif result == dfxsdk.CollectorState.ERROR:
                app.step = MeasurementStep.FAILED
                reasons = "Failed because " + dfxsdk.Collector.getLastErrorMessage()

        await renderer.put_nowait((image, (dfx_frame, frame_number, frame_timestamp_ns)))

    # Stop the tracker
    tracker.stop()

    # Close the camera
    imreader.close()

    # Signal to send_chunks that we are done
    await chunk_queue.put(None)

    # Signal to render_queue that we are done
    renderer.keep_render_last_frame()


def cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s{' (headless) ' if cv2.version.headless else ''} {_version} (libdfx v{dfxsdk.__version__})")
    parser.add_argument("-c", "--config_file", help="Path to config file", default="./config.json")

    subparser_top = parser.add_subparsers(dest="command", required=True)

    # extract - video, camera
    video_parser = subparser_top.add_parser("video", help="Extract and save chunks from a video file (offline)")
    video_parser.add_argument("study_cfg_file", help="Study config file to use", type=str, default=None)
    video_parser.add_argument("save_chunks_folder", help="Save chunks to this folder", type=str, default=None)
    video_parser.add_argument("video_path", help="Path to video file", type=str)
    video_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.0)
    video_parser.add_argument("-t", "--start_time", help="Video segment start time (seconds)", type=float, default=None)
    video_parser.add_argument("-T", "--end_time", help="Video segment end time (seconds)", type=float, default=None)
    video_parser.add_argument("--fps",
                              help="Use this framerate instead of detecting from video",
                              type=float,
                              default=None)
    video_parser.add_argument("--rotation",
                              help="Use this rotation instead of detecting from video (Must be 0, 90, 180 or 270)",
                              type=float,
                              default=None)
    video_parser.add_argument("--use-video-timestamps",
                              help="Use timestamps embedded in video instead of calculating from frame numbers "
                              "(doesn't work on all videos)",
                              action="store_true",
                              default=False)
    video_parser.add_argument("--infrared",
                              help="Assume video is from infrared camera",
                              action="store_true",
                              default=False)
    if not cv2.version.headless:
        video_parser.add_argument("--headless", help="Disable video rendering", action="store_true", default=False)
    video_parser.add_argument("-dg",
                              "--demographics",
                              help="Path to JSON file containing user demographics",
                              default=None)
    video_parser.add_argument("-ft",
                              "--face_tracker",
                              help=f"Face tracker to use. (default: {FT_CHOICES[0]})",
                              default=FT_CHOICES[0],
                              choices=FT_CHOICES)
    if "visage" in FT_CHOICES:
        video_parser.add_argument("-vl",
                                  "--visage_license",
                                  help="Path to folder containing Visage License",
                                  default="")
        video_parser.add_argument("-va",
                                  "--analyser",
                                  help="Use Visage Face Analyser module",
                                  action="store_true",
                                  default=False)
    if not cv2.version.headless:
        camera_parser = subparser_top.add_parser("camera", help="Extract and save chunks from a camera (offline)")
        camera_parser.add_argument("study_cfg_file", help="Study config file to use", type=str, default=None)
        camera_parser.add_argument("save_chunks_folder", help="Save chunks to this folder", type=str, default=None)
        camera_parser.add_argument("--camera", help="Camera ID", type=int, default=0)
        camera_parser.add_argument("-cd",
                                   "--chunk_duration_s",
                                   help="Chunk duration (seconds)",
                                   type=float,
                                   default=5.0)
        camera_parser.add_argument("-md",
                                   "--measurement_duration_s",
                                   help="Measurement duration (seconds)",
                                   type=float,
                                   default=30)
        camera_parser.add_argument("--infrared", help="Assume infrared camera", action="store_true", default=False)
        camera_parser.add_argument("--virtual",
                                   help="Assume virtual camera if set to WxH@fps e.g. 564x682@30",
                                   type=str,
                                   default=None)
        camera_parser.add_argument("-dg",
                                   "--demographics",
                                   help="Path to JSON file containing user demographics",
                                   default=None)
        camera_parser.add_argument("-ft",
                                   "--face_tracker",
                                   help=f"Face tracker to use (default: {FT_CHOICES[0]})",
                                   default=FT_CHOICES[0],
                                   choices=FT_CHOICES)
        if "visage" in FT_CHOICES:
            camera_parser.add_argument("-vl",
                                       "--visage_license",
                                       help="Path to folder containing Visage License",
                                       default="")
            camera_parser.add_argument("-va",
                                       "--analyser",
                                       help="Use Visage Analysis module",
                                       action="store_true",
                                       default=False)

    args = parser.parse_args()

    # https://github.com/aio-libs/aiohttp/issues/4324
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args))


if __name__ == '__main__':
    cmdline()
