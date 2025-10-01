import asyncio
import time

import cv2


def _mirror_and_rotate(frame, mirror, rotation):
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

    return frame


class CameraReader:
    def __init__(self, camera_id, mirror=True, fps=None, width=None, height=None, rotation=None) -> None:
        self._videocap = cv2.VideoCapture(int(camera_id))
        if not self._videocap.isOpened():
            raise RuntimeError(f"Could not open {int(camera_id)}")

        # Set the fps, width and height if passed (usually for virtual cameras)

        if fps is not None:
            self._videocap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self._videocap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise RuntimeError(f"Camera framerate {self.fps} is invalid.")

        self._frame_number = 0
        self.frame_duration_s = 1.0 / self.fps

        self.mirror = mirror
        self.rotation = 0 if rotation is None else rotation

        if width is not None:
            self._videocap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.width = int(self._videocap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if height is not None:
            self._videocap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.height = int(self._videocap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._stop = False

    def start(self):
        self._stop = False

    def stop(self):
        self._stop = True

    def close(self):
        if self._videocap.isOpened():
            self._videocap.release()

    async def read_next_frame(self):
        await asyncio.sleep(self.frame_duration_s)

        frame_number = self._frame_number
        read = self._videocap.grab()
        # This is not the best way to get a timestamp as asyncio may suspend this anytime. Ideally the camera SDK should
        # provide one with the frame itself. But since OpenCV does not, this is better than nothing.
        frame_timestamp_ns = time.monotonic_ns()
        read, frame = self._videocap.retrieve()
        self._frame_number += 1

        if read and frame is not None and frame.size != 0 and not self._stop:
            frame = _mirror_and_rotate(frame, self.mirror, self.rotation)
        else:
            # This is needed because OpenCV doesn't do a good job of ensuring this.
            read = False
            frame = None

        return read, frame, frame_number, frame_timestamp_ns


class VideoReader:
    def __init__(self,
                 video_path,
                 start_time=None,
                 stop_time=None,
                 rotation=None,
                 fps=None,
                 use_video_timestamps=False,
                 max_seconds_to_process=120) -> None:
        self._videocap = cv2.VideoCapture(video_path)
        if not self._videocap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")

        self.fps = self._videocap.get(cv2.CAP_PROP_FPS) if fps is None else fps
        if self.fps <= 0:
            raise RuntimeError(f"Video framerate {self.fps} is invalid. Please override using '--fps' parameter.")

        self.rotation = 0 if rotation is None else rotation
        self.frame_duration_ns = 1000_000_000.0 / self.fps
        self.frame_duration_s = 1.0 / self.fps

        frames_in_source = int(self._videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._videocap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._videocap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.mirror = False

        self.start_frame = 0
        self.stop_frame = frames_in_source - 1
        if start_time is not None:
            self.start_frame = max(self.start_frame, int(start_time * self.fps))
        if stop_time is not None:
            self.stop_frame = min(self.stop_frame, int(stop_time * self.fps))

        self.frames_to_process = self.stop_frame - self.start_frame
        extraction_duration = self.frames_to_process / self.fps
        if extraction_duration > max_seconds_to_process:
            print(
                f"Extraction duration of {extraction_duration:.1f} seconds is longer than {max_seconds_to_process}"
                f" seconds; processing first {max_seconds_to_process} seconds only."
            )
            self.stop_frame = int(self.fps * max_seconds_to_process)
            self.frames_to_process = self.stop_frame - self.start_frame
        if self.start_frame > 0:
            self._videocap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.use_video_timestamps = use_video_timestamps

    # def _find_video_rotation(self, video_path):
    #     try:
    #         mi = MediaInfo.parse(video_path)
    #         for track in mi.tracks:
    #             if track.track_type == "Video":
    #                 return track.rotation
    #     except Exception:
    #         print("Could not determine rotation, using 0")
    #     return 0

    def close(self):
        if self._videocap.isOpened():
            self._videocap.release()

    async def read_next_frame(self):
        await asyncio.sleep(self.frame_duration_s)

        frame_number = int(self._videocap.get(cv2.CAP_PROP_POS_FRAMES))
        read, frame = self._videocap.read()

        frame_timestamp_ns = frame_number * self.frame_duration_ns

        # Get timestamps from video if asked to
        if self.use_video_timestamps:
            # Currently OpenCV doesn't handle videos with variable frame rates correctly but perhaps in the future this
            # code will make sense
            frame_timestamp_ns = self._videocap.get(cv2.CAP_PROP_POS_MSEC) * 1000_000

            # On some videos, timestamps get read as 0.0 once you reach the end of the file
            if frame_timestamp_ns <= 0.0 and frame_number > self.start_frame:
                read = False
                frame = None

        if read and frame is not None and frame.size != 0 and frame_number <= self.stop_frame:
            frame = _mirror_and_rotate(frame, self.mirror, self.rotation)
        else:
            # This is needed because OpenCV doesn't do a good job of ensuring this.
            read = False
            frame = None

        return read, frame, frame_number, frame_timestamp_ns
