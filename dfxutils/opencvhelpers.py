import asyncio

import cv2
from pymediainfo import MediaInfo


class VideoReader:
    def __init__(self, video_path, start_time=None, end_time=None, mirror=False) -> None:
        self._videocap = cv2.VideoCapture(video_path)
        if not self._videocap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")

        self.fps = self._videocap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise RuntimeError(f"Video framerate {self.fps} is invalid. Please override using '--fps' parameter.")
        self.rotation = self._find_video_rotation(video_path)
        self.mirror = mirror
        self.frame_duration_ns = 1000000000.0 / self.fps

        self.start_frame = 0
        self.end_frame = int(self._videocap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if start_time is not None:
            self.start_frame = max(self.start_frame, int(start_time * self.fps))
        if end_time is not None:
            self.end_frame = min(self.end_frame, int(end_time * self.fps))

        self.frames_to_process = self.end_frame - self.start_frame
        if self.frames_to_process / self.fps > 120:
            print(
                f"Video segment duration {self.frames_to_process / self.fps:.1f}s is longer than 120s, processing first 120s only."
            )
            self.end_frame = int(self.fps * 120)
            self.frames_to_process = self.end_frame - self.start_frame
        if self.start_frame > 0:
            self._videocap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def _find_video_rotation(self, video_path):
        mi = MediaInfo.parse(video_path)
        for track in mi.tracks:
            if track.track_type == "Video":
                return track.rotation
        return 0

    async def read_next_frame(self):
        if self.fps > 0:
            await asyncio.sleep(1.0 / self.fps)
        else:
            await asyncio.sleep(0.033)

        frame_number = int(self._videocap.get(cv2.CAP_PROP_POS_FRAMES))
        read, frame = self._videocap.read()

        if read and frame is not None and frame.size != 0 and frame_number <= self.end_frame:
            # Mirror frame if necessary
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Rotate frame if necessary
            if self.rotation == 90:
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame, 1)
            elif self.rotation == 180:
                frame = cv2.flip(frame, -1)
            elif self.rotation == 270:
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame, 0)
            else:  # 0 or some other weird result
                pass
        else:
            # This is needed because OpenCV doesn't do a good job of ensuring this.
            read = False
            frame = None

        return read, frame, frame_number
