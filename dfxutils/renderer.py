import asyncio
import datetime

import cv2
import numpy as np


class Renderer():
    def __init__(self, version, image_src_name, first_frame, last_frame, fps, measurement_id, total_chunks, sf=1.0):
        self._render_queue = asyncio.Queue(1)
        self._version = version
        self._image_src_name = image_src_name
        self._first_frame = first_frame
        self._last_frame = last_frame
        self._fps = fps
        self._measurement_id = measurement_id
        self._total_chunks = total_chunks
        self._measuring = True
        self._message = ""
        self._results = {}
        self._sf = sf if sf > 0 else 1.0
        self._rendering_last = False
        self._recv_chunk = None
        self._sent_chunk = None

    async def render(self):
        render_image, meta = None, None

        cancelled = False
        while not self._rendering_last:
            try:
                render_image, meta = self._render_queue.get_nowait()

                render_image_copy = np.copy(render_image)
                self._draw_on_image(render_image_copy, meta)
                cv2.imshow(f"dfxdemo {self._version}", render_image_copy)
                k = cv2.waitKey(1)
                if k == 'q' or k == 27:
                    cancelled = True
                    break
            except asyncio.QueueEmpty:
                pass
            finally:
                await asyncio.sleep(0)

        if cancelled:
            return cancelled

        # Keep rendering the last frame at 10fps as we display results
        while self._rendering_last:
            await asyncio.sleep(0.1)

            render_image_copy = np.copy(render_image)
            self._draw_on_image(render_image_copy, meta)
            cv2.imshow(f"dfxdemo {self._version}", render_image_copy)
            k = cv2.waitKey(1)
            if k == 'q' or k == 27:
                break

        return False

    async def put_nowait(self, render_info):
        try:
            image, meta = render_info
            if self._sf == 1.0:
                rimage = np.copy(image)
            elif self._sf < 1.0:
                rimage = cv2.resize(image, (0, 0), fx=self._sf, fy=self._sf, interpolation=cv2.INTER_AREA)
            else:
                rimage = cv2.resize(image, (0, 0), fx=self._sf, fy=self._sf, interpolation=cv2.INTER_LINEAR)

            self._render_queue.put_nowait((rimage, meta))
        except asyncio.QueueFull:
            pass

    def keep_render_last_frame(self):
        self._rendering_last = True

    def set_message(self, message):
        self._message = message

    def set_results(self, results):
        recv_chunk = int(results["chunk_number"])
        if self._recv_chunk is None or recv_chunk > self._recv_chunk:
            self._recv_chunk = recv_chunk
            del results["chunk_number"]
            self._results = results

    def set_sent(self, sent_number):
        self._sent_chunk = int(sent_number)

    def _draw_on_image(self, render_image, image_meta):
        dfxframe, frame_number = image_meta
        # Render the face polygons
        for faceID in dfxframe.getFaceIdentifiers():
            for regionID in dfxframe.getRegionNames(faceID):
                if dfxframe.getRegionIntProperty(faceID, regionID, "draw") != 0:
                    polygon = dfxframe.getRegionPolygon(faceID, regionID)
                    cv2.polylines(render_image, [np.round(np.array(polygon) * self._sf).astype(int)],
                                  isClosed=True,
                                  color=(255, 255, 0),
                                  thickness=1,
                                  lineType=cv2.LINE_AA)
        # Render filename and framerate
        c = 2
        r = 15
        msg = f"{self._image_src_name} ({self._fps:.2f} fps)"
        r = self._draw_text(msg, render_image, (c, r))

        # Render the message
        if self._message:
            r = self._draw_text(self._message, render_image, (c, r), fg=(255, 0, 0))

        r += 10
        # Render progress
        if self._measuring:
            if frame_number < self._last_frame:
                if self._first_frame > 0:
                    r = self._draw_text(
                        f"Processing frame {frame_number} of {self._first_frame} to {self._last_frame + 1}",
                        render_image, (c, r))
                else:
                    r = self._draw_text(f"Processing frame {frame_number} of {self._last_frame + 1}", render_image,
                                        (c, r))
            else:
                if self._first_frame > 0:
                    r = self._draw_text(
                        f"Processed all {self._last_frame - self._first_frame + 1} frames from {self._first_frame} to {self._last_frame + 1}",
                        render_image, (c, r))
                else:
                    r = self._draw_text(f"Processed all {self._last_frame - self._first_frame + 1} frames",
                                        render_image, (c, r))

        # Render chunk numbers and results
        if self._sent_chunk is not None:
            r = self._draw_text(f"Sent chunk: {self._sent_chunk + 1} of {self._total_chunks}", render_image, (c, r))
        if self._results:
            r = self._draw_text(f"Received result: {self._recv_chunk + 1} of {self._total_chunks}", render_image,
                                (c, r))
            for k, v in self._results.items():
                r = self._draw_text(f"{k}: {v}", render_image, (c + 10, r), fs=0.8)

        # Render the current time (so user knows things aren't frozen)
        now = datetime.datetime.now()
        self._draw_text(f"{now.strftime('%X')}",
                        render_image, (render_image.shape[1] - 70, 15),
                        fg=(0, 128, 0) if now.second % 2 == 0 else (0, 0, 0))

    def _draw_text(self, msg, render_image, origin, fs=None, fg=None, bg=None):
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        THICK = 1
        PAD = 3
        fs = 0.45 if fs is None else fs * 0.45
        fg = (0, 0, 0) if fg is None else fg
        bg = (255, 255, 255) if bg is None else bg

        sz, baseline = cv2.getTextSize(msg, FONT, fs, THICK)
        cv2.rectangle(render_image, (origin[0] - PAD, origin[1] - sz[1] - PAD),
                      (origin[0] + sz[0] + PAD, origin[1] + sz[1] - baseline * 2 + PAD),
                      bg,
                      thickness=-1)
        cv2.putText(render_image, msg, origin, FONT, fs, fg, THICK, AA)

        return origin[1] + sz[1] + baseline + 1


class NullRenderer():
    async def render(self):
        pass

    async def put_nowait(self, _):
        pass

    def keep_render_last_frame(self):
        pass

    def set_message(self, _):
        pass

    def set_results(self, _):
        pass

    def set_sent(self, _):
        pass