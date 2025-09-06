import argparse

import cv2


def GetDefaultArgs(input = None):
    parser = argparse.ArgumentParser()
    FT_CHOICES = ["mediapipe"]
    parser.add_argument("-c", "--config_file", help="Path to config file", default="./config.json")
    pp_group = parser.add_mutually_exclusive_group()
    pp_group.add_argument("--json", help="Print as JSON", action="store_true", default=False)
    pp_group.add_argument("--csv", help="Print grids as CSV", action="store_true", default=False)

    subparser_top = parser.add_subparsers(dest="command", required=True)

    # orgs - register, unregister
    subparser_orgs = subparser_top.add_parser("orgs", aliases=["o", "org"],
                                              help="Organizations").add_subparsers(dest="subcommand", required=True)
    register_parser = subparser_orgs.add_parser("register", help="Register device")
    register_parser.add_argument("license_key", help="DFX API Organization License")
    register_parser.add_argument("--rest-url", help="Connect to DFX API using this REST URL", default=None)
    unregister_parser = subparser_orgs.add_parser("unregister", help="Unregister device")
    o_login_parser = subparser_orgs.add_parser("login", help="Adminstrative login (no measurements)")
    o_login_parser.add_argument("org_key", help="Organization Key")
    o_login_parser.add_argument("email", help="Email address")
    o_login_parser.add_argument("password", help="Password")
    o_list_parser = subparser_orgs.add_parser("list-measurements",
                                              aliases=[
                                                  "lm",
                                              ],
                                              help="List existing measurements across org")
    o_list_parser.add_argument("--limit",
                               help="Number of measurements to retrieve (default: 10, max: 50)",
                               type=int,
                               default=10)
    o_list_parser.add_argument("--offset",
                               help="Offset of the measurements to retrieve (default: 0)",
                               type=int,
                               default=0)
    o_list_parser.add_argument("--profile_id", help="Filter list by Profile ID", type=str, default="")
    o_list_parser.add_argument("--partner_id", help="Filter list by Partner ID", type=str, default="")
    o_list_parser.add_argument("--study_id", help="Filter list by Study ID", type=str, default="")
    o_list_parser.add_argument("--email", help="Filter list by Email", type=str, default="")
    o_get_parser = subparser_orgs.add_parser("get-measurement",
                                             aliases=[
                                                 "gm",
                                             ],
                                             help="Retrieve a measurement across org")
    o_get_parser.add_argument("measurement_id", help="ID of measurement to retrieve")
    o_get_parser.add_argument("--expand", help="Retrieve vector results per signal", action="store_true", default=False)

    # users - login, logout
    subparser_users = subparser_top.add_parser("users", aliases=["u", "user"],
                                               help="Users").add_subparsers(dest="subcommand", required=True)
    login_parser = subparser_users.add_parser("login", help="User login")
    login_parser.add_argument("email", help="Email address")
    login_parser.add_argument("password", help="Password")
    logout_parser = subparser_users.add_parser("logout", help="User logout")

    # profiles - create, update, remove, get, list
    subparser_profiles = subparser_top.add_parser("profiles", aliases=["p", "profile"],
                                                  help="Profiles").add_subparsers(dest="subcommand", required=True)
    profile_create_parser = subparser_profiles.add_parser("create", help="Create profile")
    profile_create_parser.add_argument("name", help="Name (unique)", type=str)
    profile_create_parser.add_argument("--email", help="Email", type=str, default="no_email_provided")
    profile_update_parser = subparser_profiles.add_parser("update", help="Update profile")
    profile_update_parser.add_argument("profile_id", help="Profile ID to update", type=str)
    profile_update_parser.add_argument("name", help="New Name", type=str, default="")
    profile_update_parser.add_argument("email", help="New Email", type=str, default="")
    profile_update_parser.add_argument("status", help="New Status", type=str, default="")
    profile_remove_parser = subparser_profiles.add_parser("remove", help="Remove profile")
    profile_remove_parser.add_argument("profile_id", help="Profile ID to remove", type=str)
    profile_get_parser = subparser_profiles.add_parser("get", help="Retrieve profile")
    profile_get_parser.add_argument("profile_id", help="Profile ID to retrieve", type=str)
    profile_list_parser = subparser_profiles.add_parser("list", help="List profiles")

    # studies - list, get, select
    subparser_studies = subparser_top.add_parser("studies", aliases=["s", "study"],
                                                 help="Studies").add_subparsers(dest="subcommand", required=True)
    study_list_parser = subparser_studies.add_parser("list", help="List existing studies")
    study_get_parser = subparser_studies.add_parser("get", help="Retrieve a study's information")
    study_get_parser.add_argument("study_id",
                                  nargs="?",
                                  help="ID of study to retrieve (default: selected study)",
                                  type=str)
    study_select_parser = subparser_studies.add_parser("select", help="Select a study to use")
    study_select_parser.add_argument("study_id", help="ID of study to use", type=str)

    # measure - list, get, make, make_camera, debug_make_from_chunks
    subparser_meas = subparser_top.add_parser("measure", aliases=["m", "measurements"],
                                              help="Measurements").add_subparsers(dest="subcommand", required=True)
    list_parser = subparser_meas.add_parser("list", help="List existing measurements")
    list_parser.add_argument("--limit",
                             help="Number of measurements to retrieve (default: 10, max: 50)",
                             type=int,
                             default=10)
    list_parser.add_argument("--offset",
                             help="Offset of the measurements to retrieve (default: 0)",
                             type=int,
                             default=0)
    list_parser.add_argument("--profile_id", help="Filter list by Profile ID", type=str, default="")
    list_parser.add_argument("--partner_id", help="Filter list by Partner ID", type=str, default="")
    get_parser = subparser_meas.add_parser("get", help="Retrieve a measurement")
    get_parser.add_argument("measurement_id",
                            nargs="?",
                            help="ID of measurement to retrieve (default: last measurement)",
                            default=None)
    get_parser.add_argument("--expand", help="Retrieve vector results per signal", action="store_true", default=False)
    
    make_parser = subparser_meas.add_parser("make", help="Make a measurement from a video file")
    make_parser.add_argument("video_path", help="Path to video file", type=str)
    make_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.0)
    make_parser.add_argument("-t",
                                "--start_time",
                                help="Video segment start time (seconds)",
                                type=float,
                                default=None)
    make_parser.add_argument("-T", "--end_time", help="Video segment end time (seconds)", type=float, default=None)
    make_parser.add_argument("--fps",
                                help="Use this framerate instead of detecting from video",
                                type=float,
                                default=None)
    make_parser.add_argument("--rotation",
                                help="Use this rotation instead of detecting from video (Must be 0, 90, 180 or 270)",
                                type=float,
                                default=None)
    make_parser.add_argument("--use-video-timestamps",
                                help="Use timestamps embedded in video instead of calculating from frame numbers "
                                "(doesn't work on all videos)",
                                action="store_true",
                                default=False)
    make_parser.add_argument("--infrared",
                                help="Assume video is from infrared camera",
                                action="store_true",
                                default=False)
    if not cv2.version.headless:
        make_parser.add_argument("--headless", help="Disable video rendering", action="store_true", default=False)
    make_parser.add_argument("--profile_id", help="Set the Profile ID (Participant ID)", type=str, default="")
    make_parser.add_argument("--partner_id", help="Set the PartnerID", type=str, default="")
    make_parser.add_argument("-dg",
                                "--demographics",
                                help="Path to JSON file containing user demographics",
                                default=None)
    make_parser.add_argument("--debug_study_cfg_file",
                                help="Study config file to use instead of data from API (debugging)",
                                type=str,
                                default=None)
    make_parser.add_argument("--debug_save_chunks_folder",
                                help="Save SDK chunks to folder (debugging)",
                                type=str,
                                default=None)
    make_parser.add_argument("-ft",
                                "--face_tracker",
                                help=f"Face tracker to use. (default: {FT_CHOICES[0]})",
                                default=FT_CHOICES[0],
                                choices=FT_CHOICES)
       
    if not cv2.version.headless:
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
                                   help=f"Face tracker to use",
                                   default="mediapipe")
        

    mk_ch_parser = subparser_meas.add_parser("debug_make_from_chunks",
                                             help="Make a measurement from saved SDK chunks (debugging)")
    mk_ch_parser.add_argument("debug_chunks_folder", help="Folder containing SDK chunks", type=str)
    mk_ch_parser.add_argument("-cd", "--chunk_duration_s", help="Chunk duration (seconds)", type=float, default=5.0)
    mk_ch_parser.add_argument("--profile_id", help="Set the Profile ID (Participant ID)", type=str, default="")
    mk_ch_parser.add_argument("--partner_id", help="Set the PartnerID", type=str, default="")

    make_parser.add_argument("--healthfox", help="For HealthFOX", action="store_true", default=False)

    args1 = parser.parse_args(input)
    return args1