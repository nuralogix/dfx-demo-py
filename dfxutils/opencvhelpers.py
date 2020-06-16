import asyncio

import cv2


class OpenCvHelpers:
    @staticmethod
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

    @staticmethod
    async def read_next_frame(video_cap, target_fps, rotation, mirror):
        if target_fps > 0:
            await asyncio.sleep(1.0 / target_fps)
        else:
            await asyncio.sleep(0.033)

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
