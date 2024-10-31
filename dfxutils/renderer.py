import asyncio
import datetime

from collections import deque

import cv2
import numpy as np

from .app import MeasurementStep

_message_action = {
    MeasurementStep.NOT_READY: "",
    MeasurementStep.READY: "Press 's' to start",
    MeasurementStep.USER_STARTED: "",
    MeasurementStep.MEASURING: "Press Esc to cancel",
    MeasurementStep.WAITING_RESULTS: "Press Esc to cancel",
    MeasurementStep.COMPLETED: "Press Esc to exit",
    MeasurementStep.USER_CANCELLED: "",
    MeasurementStep.FAILED: "Press Esc to exit"
}

_message_state = {
    MeasurementStep.NOT_READY: "Not ready to measure",
    MeasurementStep.READY: "Ready to measure",
    MeasurementStep.USER_STARTED: "Starting",
    MeasurementStep.MEASURING: "",
    MeasurementStep.WAITING_RESULTS: "Waiting for results",
    MeasurementStep.COMPLETED: "Measurement completed successfully",
    MeasurementStep.USER_CANCELLED: "Measurement cancelled by user",
    MeasurementStep.FAILED: "Measurement failed"
}


class Renderer():
    def __init__(self, version, image_src_name, fps, app, sf=1.0):
        self._render_queue = asyncio.Queue(1)
        self._version = version
        self._image_src_name = image_src_name
        self._fps = fps
        self._app = app
        self._feedback = ""
        self._results = {}
        self._sf = sf if sf > 0 else 1.0
        self._rendering_last = False
        self._recv_chunk = None
        self._sent_chunk = None
        self._timestamps_ns = deque(maxlen=11)
        self._last_frame_number = None
        if self._app.extract_only:
            for k in _message_state:
                _message_state[k] = _message_state[k].replace("Measurement", "Extraction").replace("measure", "extract")

    async def render(self):
        render_image, meta = None, None

        cancelled = False
        while not self._rendering_last:
            try:
                render_image, meta = self._render_queue.get_nowait()
                _, frame_number, frame_timestamp_ns = meta
                if not self._rendering_last:
                    self._timestamps_ns.append(frame_timestamp_ns)

                render_image_copy = np.copy(render_image)
                self._draw_on_image(render_image_copy, meta)
                cv2.imshow(f"dfxdemo {self._version}", render_image_copy)
                k = cv2.waitKey(1)
                if k in [ord('q'), 27]:
                    cancelled = True
                    self._app.step = MeasurementStep.USER_CANCELLED
                    break
                if self._app.step == MeasurementStep.READY and k in [ord('s'), ord(' ')]:
                    self._app.step = MeasurementStep.USER_STARTED
                elif self._app.step == MeasurementStep.FAILED and k in [ord('r')]:
                    self._app.step = MeasurementStep.NOT_READY
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
            if k in [ord('q'), 27]:
                if self._app.step == MeasurementStep.WAITING_RESULTS:
                    self._app.step = MeasurementStep.USER_CANCELLED
                    cancelled = True
                break

        return cancelled

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

    def set_constraints_feedback(self, feedback):
        self._feedback = feedback

    def set_results(self, results):
        recv_chunk = int(results["chunk_number"])
        if self._recv_chunk is None or recv_chunk > self._recv_chunk:
            self._recv_chunk = recv_chunk
            del results["chunk_number"]
            self._results = results

    def set_sent(self, sent_number):
        self._sent_chunk = int(sent_number)

    def _draw_on_image(self, render_image, image_meta):
        dfxframe, frame_number, frame_timestamp_ns = image_meta
        # Render the target_rect
        if self._app.constraints_cfg is not None:
            w = render_image.shape[1] * self._app.constraints_cfg.boxWidth_pct / 100
            h = render_image.shape[0] * self._app.constraints_cfg.boxHeight_pct / 100
            xc = render_image.shape[1] * self._app.constraints_cfg.boxCenterX_pct / 100
            yc = render_image.shape[0] * self._app.constraints_cfg.boxCenterY_pct / 100
            cv2.rectangle(render_image, (int(xc - w / 2), int(yc - h / 2)), (int(xc + w / 2), int(yc + h / 2)),
                          color=(255, 0, 0),
                          thickness=1,
                          lineType=cv2.LINE_AA)

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

        # Render the current time (so user knows things aren't frozen)
        now = datetime.datetime.now()
        self._draw_text(f"{now.strftime('%X')}",
                        render_image, (render_image.shape[1] - 70, 15),
                        fg=(0, 128, 0) if now.second % 2 == 0 else (0, 0, 0))

        # Render filename, framerate of last 10 frames and expected framerate
        c = 2
        r = 15
        if not self._app.is_camera:
            msg = f"{self._image_src_name}: {self._fps:.2f} fps"
        else:
            if len(self._timestamps_ns) >= 2:
                deltas = [self._timestamps_ns[i + 1] - self._timestamps_ns[i] for i in range(len(self._timestamps_ns) - 1)]
                avg_delta = sum(deltas) / len(deltas)
                fps_now = 1000_000_000.0 / avg_delta
            else:
                fps_now = self._fps
            msg = f"{self._image_src_name}: {fps_now:.2f} fps (Expected {self._fps:.2f} fps)"
        r = self._draw_text(msg, render_image, (c, r))

        # Render the message
        message_action = _message_action[self._app.step]
        if message_action:
            r = self._draw_text(message_action, render_image, (c, r), fg=(255, 0, 0))

        # Render progress
        if self._app.step == MeasurementStep.MEASURING:
            if self._app.begin_frame > 0:
                r = self._draw_text(
                    f"Extracting frame {frame_number} of {self._app.begin_frame} to {self._app.end_frame + 1}",
                    render_image, (c, r))
            else:
                r = self._draw_text(f"Extracting frame {frame_number} of {self._app.end_frame + 1}", render_image,
                                    (c, r))
        elif self._app.step == MeasurementStep.WAITING_RESULTS:
            if self._app.begin_frame > 0:
                r = self._draw_text(
                    f"Extracted all {self._app.end_frame - self._app.begin_frame + 1} frames from "
                    f"{self._app.begin_frame} to {self._app.end_frame + 1}", render_image, (c, r))
            else:
                r = self._draw_text(f"Extracted all {self._app.end_frame - self._app.begin_frame + 1} frames",
                                    render_image, (c, r))
            dots = "..." if now.second % 2 == 0 else ""
            r = self._draw_text(_message_state[self._app.step] + dots, render_image, (c, r))
        else:
            r = self._draw_text(_message_state[self._app.step], render_image, (c, r))

        # Render the constraints feedback
        if self._feedback:
            r = self._draw_text(self._feedback, render_image, (c, r), fg=(0, 0, 255))

        # Render chunk numbers and results
        if self._sent_chunk is not None:
            msg = "Saved" if self._app.extract_only else "Sent"
            r = self._draw_text(f"{msg} chunk: {self._sent_chunk + 1} of {self._app.number_chunks}", render_image,
                                (c, r))
        if self._results:
            r = self._draw_text(f"Received result: {self._recv_chunk + 1} of {self._app.number_chunks}", render_image,
                                (c, r))
            for k, v in self._results.items():
                r = self._draw_text(f"{k}: {v}", render_image, (c + 10, r), fs=0.8)

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

    def set_constraints_feedback(self, _):
        pass

    def set_results(self, _):
        pass

    def set_sent(self, _):
        pass
