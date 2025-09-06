import argparse
import asyncio
import base64
import copy
import glob
import json
import math
import os.path
import platform
import random
import string
from dfxdemo import main
import aiohttp
import cv2
import dfx_apiv2_client as dfxapi
import libdfx as dfxsdk
from pygame import Surface
import yarl
from kioskrequest import KioskRequest
from default_args import GetDefaultArgs
from defaults import getdefaults
from helpers import *
import sys
from dfxutils.app import AppState, MeasurementStep
from dfxutils.opencvhelpers import CameraReader, VideoReader
from dfxutils.prettyprint import PrettyPrinter as PP
from dfxutils.renderer import NullRenderer, Renderer
from pygamerenderer import *
from dfxutils.sdkhelpers import DfxSdkHelpers

from quart import Quart
from kettuspinner import KettuSpinner
FT_CHOICES = []

try:
    from dfxutils.mp_tasksvision_tracker import MediaPipeTasksVisionTracker
    FT_CHOICES.append("taskvision")
except ImportError:
    pass

try:
    from dfxutils.mediapipe_tracker import MediaPipeTracker
    FT_CHOICES.append("mediapipe")
except ImportError:
    pass


try:
    import importlib.metadata
    _version = f"v{importlib.metadata.version('dfxdemo')}"
except Exception:
    _version = ""



class Empty:
    pass



async def run_job(screen: Surface, data: KioskRequest = None, useDemoRenderer = True):
    kettu = KettuSpinner("/Users/alexkivikoski/Documents/hfkiosk/kettuanim")
    await kettu.FadeInOut(screen,(screen.get_width()/2-128,screen.get_height()/2-128))

    # Load config


    args = GetDefaultArgs(["measure", "make_camera"])
    args.camera = 1
    #await main(args,screen)
    #return
    
    

    config = load_config(args.config_file)

    # Verify and if necessary, attempt to renew the token
    using_user_token = bool(dfxapi.Settings.user_token)
    verified, renewed, headers, new_config = await verify_renew_token(config, False)
    if not verified:
        if new_config is not None:
            save_config(new_config, args.config_file)
        if not renewed:
            return

    # Prepare to make a measurement..
    app = AppState()
    # ..using a video or camera
    app.is_camera = True #args.subcommand == "make_camera"
    app.is_infrared = False
    app.virtual = None
    headless = False

    if (data is not None):
        app.demographics = data.get_demographics()

    image_src_name = f"Camera {args.camera}"
    try:
        # Open the camera or video
        width, height, fps = 1280, 720, 30
        
        imreader = CameraReader(args.camera, mirror=False, fps=fps, width=1280,
                                height=720)
        
       

        # Open the demographics file if provided
        if args.demographics is not None:
            with open(args.demographics, "r") as f:
                app.demographics = json.load(f)

        # Create a face tracker
        tracker = MediaPipeTasksVisionTracker(1, track_in_background=app.is_camera)

        # Create DFX SDK factory
        factory = dfxsdk.Factory()
        print("Created DFX Factory:", factory.getVersion())
        sdk_id = factory.getSdkId()

        # ..from API required to initialize DFX SDK collector (or FAIL)
        study_cfg_bytes = await retrieve_sdk_config(headers, config, args.config_file, sdk_id)
    
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
    
    app.number_chunks = math.ceil(args.measurement_duration_s / args.chunk_duration_s)
    app.end_frame = math.ceil(args.measurement_duration_s * imreader.fps)
   
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
    app.constraints_cfg = DfxSdkHelpers.ConstraintsConfig(collector.getConstraintsConfig("json"))
    app.constraints_cfg.minimumFps = 10
    collector.setConstraintsConfig("json", str(app.constraints_cfg))

    # Print the enabled constraints
    print("Constraints:")
    for constraint in collector.getEnabledConstraints():
        print(f"    enabled: {constraint}")

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
        async with dfxapi.Measurements.ws_connect(session) as ws:
            # Auth using `ws_auth_with_token` if headers cannot be manipulated, normally you don't need to do this
            # if "Authorization" not in session.headers:
            #     await dfxapi.Organizations.ws_auth_with_token(ws, generate_reqid())
            #     await ws.receive()  # Wait to receive response before proceeding..

            # Subscribe to results
            results_request_id = generate_reqid()
            await dfxapi.Measurements.ws_subscribe_to_results(ws, generate_reqid(), app.measurement_id,
                                                              results_request_id)

            # Queue to pass chunks between coroutines
            chunk_queue = asyncio.Queue(app.number_chunks)

            # When we receive the last chunk from the SDK, we can check for measurement completion
            app.last_chunk_sent = False

            # Coroutine to produce chunks and put then in chunk_queue
            
            renderer = PygameRenderer(
                _version,
                image_src_name,
                imreader.fps,
                app,
                screen, 
                useDemoRenderer, 
                0.5 if imreader.height >= 720 else 1.0,
            )
            
            print("Waiting to start")
            produce_chunks_coro = extract_from_imgs(
                chunk_queue,  # Chunks will be put into this queue
                imreader,  # Image reader
                tracker,  # Face tracker
                collector,  # DFX SDK collector needed to create chunks
                renderer,  # Rendering
                app)  # App

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
                    await dfxapi.Measurements.ws_add_data(ws,
                                                          generate_reqid(),
                                                          app.measurement_id,
                                                          action,
                                                          chunk.payload_data,
                                                          chunk_order=chunk.chunk_number,
                                                          start_time_s=chunk.start_time_s,
                                                          end_time_s=chunk.end_time_s,
                                                          duration_s=chunk.duration_s,
                                                          metadata=chunk.metadata)
                    print(f"Sent chunk {chunk.chunk_number}")
                    renderer.set_sent(chunk.chunk_number)

                    # Update data needed to check for completion
                    app.number_chunks_sent += 1
                    app.last_chunk_sent = action == 'LAST::PROCESS'

                    chunk_queue.task_done()

                app.step = MeasurementStep.WAITING_RESULTS
                print("Extraction complete, waiting for results")

            # Coroutine to receive responses using the Websocket
            async def receive_results():
                num_results_received = 0
                try: 
                    async for msg in ws:
                        status, request_id, payload = dfxapi.Measurements.ws_decode(msg)
                        if request_id == results_request_id:
                            json_result = json.loads(payload)
                            result = DfxSdkHelpers.json_result_to_dict(json_result)
                            renderer.set_results(result.copy())
                            print(payload) if args.json else PP.print_sdk_result(result)
                            num_results_received += 1
                            renderer.setResultsCount(num_results_received)
                        # We are done if the last chunk is sent and number of results received equals number of chunks sent
                        if app.last_chunk_sent and num_results_received == app.number_chunks_sent:
                            print("Last chunk sent, closing websocket")
                            await ws.close()
                            break
                    if (app.last_chunk_sent):
                        app.step = MeasurementStep.COMPLETED
                        print("Measurement complete")
                    else:
                        print("Websocket closed too early")
                        app.step = MeasurementStep.FAILED
                except Exception:
                    app.step = MeasurementStep.FAILED
                

            # Coroutine for rendering
            async def render():
                if type(renderer) == NullRenderer:
                    return

                cancelled = await renderer.render()
                #cv2.destroyAllWindows()
                pygame.quit()
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
                        # raise e  # Uncomment this to see a stack trace
                print(f"Measurement {app.measurement_id} failed")
            else:
                config["last_measurement"] = app.measurement_id
                save_config(config, args.config_file)
                print(f"Measurement {app.measurement_id} completed")
                print(f"Use 'dfxdemo measure get' to get comprehensive results")
    return renderer._results

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1080/2,1920/2))
    asyncio.run(run_job(screen,None,True))