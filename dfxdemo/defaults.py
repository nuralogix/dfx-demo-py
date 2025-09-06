import argparse
import cv2
import libdfx as dfxsdk

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

def getdefaults():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s{' (headless) ' if cv2.version.headless else ''} {_version} (libdfx v{dfxsdk.__version__})")
    parser.add_argument("-c", "--config_file", help="Path to config file", default="./config.json")
    pp_group = parser.add_mutually_exclusive_group()
    pp_group.add_argument("--json", help="Print as JSON", action="store_true", default=False)
    pp_group.add_argument("--csv", help="Print grids as CSV", action="store_true", default=False)

    subparser_top = parser.add_subparsers(dest="command")

    
    # measure - list, get, make, make_camera, debug_make_from_chunks
    subparser_meas = subparser_top.add_parser("measure", aliases=["m", "measurements"],
                                              help="Measurements").add_subparsers(dest="subcommand")
    
    if FT_CHOICES and not cv2.version.headless:
        camera_parser = subparser_meas.add_parser("make_camera", help="Make a measurement from a camera")
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
        camera_parser.add_argument("--profile_id", help="Set the Profile ID (Participant ID)", type=str, default="")
        camera_parser.add_argument("--partner_id", help="Set the PartnerID", type=str, default="")
        camera_parser.add_argument("--infrared", help="Assume infrared camera", action="store_true", default=False)
        camera_parser.add_argument("--virtual",
                                   help="Assume virtual camera if set to WxH@fps e.g. 564x682@30",
                                   type=str,
                                   default=None)
        camera_parser.add_argument("-dg",
                                   "--demographics",
                                   help="Path to JSON file containing user demographics",
                                   default=None)
        camera_parser.add_argument("--debug_study_cfg_file",
                                   help="Study config file to use instead of data from API (debugging)",
                                   type=str,
                                   default=None)
        camera_parser.add_argument("--debug_save_chunks_folder",
                                   help="Save SDK chunks to folder (debugging)",
                                   type=str,
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

    mk_ch_parser = subparser_meas.add_parser("debug_make_from_chunks",
                                             help="Make a measurement from saved SDK chunks (debugging)")
    mk_ch_parser.add_argument("debug_chunks_folder", help="Folder containing SDK chunks", type=str)
    mk_ch_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.0)
    mk_ch_parser.add_argument("--profile_id", help="Set the Profile ID (Participant ID)", type=str, default="")
    mk_ch_parser.add_argument("--partner_id", help="Set the PartnerID", type=str, default="")
    return parser.parse_args()