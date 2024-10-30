import enum


class MeasurementStep(enum.Enum):
    NOT_READY = 0
    READY = 1,
    USER_STARTED = 2,
    MEASURING = 3,
    WAITING_RESULTS = 4,
    COMPLETED = 5,
    USER_CANCELLED = 6,
    FAILED = 7


class AppState:
    step = MeasurementStep(MeasurementStep.NOT_READY)
    is_camera = False
    measurement_id = ""
    number_chunks = 0
    number_chunks_sent = 0
    last_chunk_sent = False
    begin_frame = 0
    end_frame = 0
    constraints_cfg = None
    demographics = None
    is_infrared = False
