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

import aiohttp
import cv2
import dfx_apiv2_client as dfxapi
import libdfx as dfxsdk
import yarl

from defaults import getdefaults

from dfxutils.app import AppState, MeasurementStep
from dfxutils.opencvhelpers import CameraReader, VideoReader
from dfxutils.prettyprint import PrettyPrinter as PP
from dfxutils.renderer import NullRenderer, Renderer
#from hfrenderer import HfRenderer
from dfxutils.sdkhelpers import DfxSdkHelpers

FT_CHOICES = []


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
    
def load_config(config_file):
    config = {
        "device_id": "",
        "device_token": "",
        "device_refresh_token": "",
        "role_id": "",
        "user_id": "",
        "user_token": "",
        "user_refresh_token": "",
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
    dfxapi.Settings.device_refresh_token = config["device_refresh_token"]
    dfxapi.Settings.role_id = config["role_id"]
    dfxapi.Settings.user_id = config["user_id"]
    dfxapi.Settings.user_token = config["user_token"]
    dfxapi.Settings.user_refresh_token = config["user_refresh_token"]
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
    # Set channel order based on is_infrared, is_camera and virtual
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
                try:
                    reasons = "Failed because " + dfxsdk.Collector.getLastErrorMessage()
                except Exception:
                    reasons = "Failed to get reason"

        await renderer.put_nowait((image, (dfx_frame, frame_number, frame_timestamp_ns)))

    # Stop the tracker
    tracker.stop()

    # Close the camera
    imreader.close()

    # Signal to send_chunks that we are done
    await chunk_queue.put(None)

    # Signal to render_queue that we are done
    renderer.keep_render_last_frame()

async def verify_renew_token(config, using_user_token):
    token_type = 'user' if using_user_token else 'device'
    # Use the token to create the headers
    if using_user_token:
        headers = {"Authorization": f"Bearer {dfxapi.Settings.user_token}"}
    else:
        headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}

    # Verify that our token is still valid and renew if it's not
    async with aiohttp.ClientSession(headers=headers, raise_for_status=False) as session:
        status, body = await dfxapi.General.verify_token(session)
        if status < 400:
            return True, False, headers, None

        # Something other than an expired token went wrong, so bail without touching saved tokens
        if not (status == 401 and "Code" in body and body["Code"] == "TOKEN_EXPIRED"):
            # Show error from verify_token failure
            print(f"Your {token_type} token could not be verified.")
            print(f"{status}: {body}")
            return False, False, None, None

        # Token has expired, attempt to renew it...
        if using_user_token:
            renew_status, renew_body = await dfxapi.Auths.renew_user_token(session)
        else:
            renew_status, renew_body = await dfxapi.Auths.renew_device_token(session)

        # Renew failed
        if renew_status >= 400:
            # Show error from verify_token failure
            print(f"Your {token_type} token could not be verified.")
            print(f"{status}: {body}")

            # Show error from renew_token failure
            print(f"Attempted {token_type} token refresh but failed, please register and/or login again!")
            print(f"{renew_status}: {renew_body}")

            # Something other than an expired token went wrong, so bail without touching saved tokens
            if not (renew_status == 401 and "Code" in renew_body and renew_body["Code"] == "TOKEN_EXPIRED"):
                return False, False, None, None

            # Erase saved tokens
            if using_user_token:
                config["user_token"] = ""
                config["user_refresh_token"] = ""
            else:
                config["device_id"] = ""
                config["device_token"] = ""
                config["device_refresh_token"] = ""
                config["user_token"] = ""  # User tokens are also invalid if device tokens are
                config["user_refresh_token"] = ""
                config["role_id"] = ""
                config["user_id"] = ""

            # Exit since we cannot continue
            return False, False, None, config

        # Renew worked, so save new tokens
        if using_user_token:
            config["user_token"] = dfxapi.Settings.user_token
            config["user_refresh_token"] = dfxapi.Settings.user_refresh_token

            # Adjust headers
            headers = {"Authorization": f"Bearer {dfxapi.Settings.user_token}"}
        else:
            config["device_token"] = dfxapi.Settings.device_token
            config["device_refresh_token"] = dfxapi.Settings.device_refresh_token

            # Adjust headers
            headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}

        # Continue
        print(f"Refreshed {token_type} token. Continuing with command...")

        return False, True, headers, config
    
def determine_action(chunk_number, number_chunks):
    action = 'CHUNK::PROCESS'
    if chunk_number == 0 and number_chunks > 1:
        action = 'FIRST::PROCESS'
    elif chunk_number == number_chunks - 1:
        action = 'LAST::PROCESS'
    return action