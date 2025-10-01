import asyncio

import cv2
from picamera2 import Picamera2
import libcamera


def _rotate(frame, rotation):
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

    return frame

class PiCameraReader:

    def __init__(self, camera_id=0, mirror=True, fps=None, width=None, height=None, rotation=None) -> None:
        self.picam2 = Picamera2(camera_num=camera_id)

        if width is None:
            width = 640
        if height is None:
            height = 480
        if fps is None:
            fps = 30

        if mirror:
            transform = libcamera.Transform(hflip=1)
        else:
            transform = libcamera.Transform()

        config = self.picam2.create_video_configuration(
            main={
                "size": (width, height),
                "format": "RGB888",
            },
            controls={
                "FrameRate": fps,
                # "AeEnable": True,
                # "AwbEnable": True,
                # "AeFlickerMode": libcamera.controls.AeFlickerModeEnum.Off,
                # "NoiseReductionMode": libcamera.controls.draft.NoiseReductionModeEnum.Off,
            },
            buffer_count=10,
            transform=transform, # Handle mirroring
        )
        self.picam2.align_configuration(config)
        self.picam2.configure(config)
        self.picam2.start()
        self.rotation = 0 if rotation is None else rotation

        self._frame_number = 0
        self._stop = False

        # Get actual width, height
        self.width = config["main"]["size"][0]
        self.height = config["main"]["size"][1]

        # Set fps and frame duration
        self.fps = fps
        self._frame_duration_s = 1.0 / self.fps

    def start(self):
        self._stop = False
        if not self.picam2.started:
            self.picam2.start()

    def stop(self):
        self._stop = True
        if self.picam2.started:
            self.picam2.stop()

    def close(self):
        if self.picam2.started:
            self.picam2.stop()
        self.picam2.close()

    async def read_next_frame(self):
        # Sleep for the smallest amount of time, just to keep asyncio happy
        await asyncio.sleep(self._frame_duration_s / 100)
        with self.picam2.captured_request() as request:
            frame = request.make_array("main")
            metadata = request.get_metadata()
            frame_number = self._frame_number
            self._frame_number += 1

            # Try to get timestamp from metadata
            frame_timestamp_ns = metadata["SensorTimestamp"]
            read = frame is not None and frame.size != 0 and not self._stop

            if read:
                frame = _rotate(frame, self.rotation)

            return read, frame, frame_number, frame_timestamp_ns

    @classmethod
    def list(cls):
        cameras = Picamera2.global_camera_info()
        return cameras
