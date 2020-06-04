import argparse
import asyncio
import glob
import json
import os.path
import random
import string

import aiohttp

import dfx_apiv2_client as dfxapi

from dfxpydemoutils import print_pretty, print_meas, DlibTracker


async def main(args):
    # Load config
    config = load_config(args.config_file)

    # Check API status
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        api_status = await dfxapi.General.api_status(session)
        if not api_status["StatusID"] == "ACTIVE":
            print(f"DFX API Status: {api_status['StatusID']} ({dfxapi.Settings.rest_url})")

            return

    # Register or unregister
    if args.command == "org":
        if args.subcommand == "unregister":
            success = await unregister(config, args.config_file)
        else:
            success = await register(config, args.config_file, args.license_key)

        if success:
            save_config(config, args.config_file)
        return

    # Login or logout
    if args.command == "user":
        if args.subcommand == "logout":
            success = logout(config, args.config_file)
        else:
            success = await login(config, args.config_file, args.email, args.password)

        if success:
            save_config(config, args.config_file)
        return

    # The commands below need a token, so make sure we are registered and/or logged in
    if not dfxapi.Settings.device_token and not dfxapi.Settings.user_token:
        print("Please register and/or login first to obtain a token")
        return

    # Use the token to create the headers
    token = dfxapi.Settings.user_token if dfxapi.Settings.user_token else dfxapi.Settings.device_token
    headers = {"Authorization": f"Bearer {token}"}

    # Retrieve or list studies
    if args.command == "study":
        async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
            if args.subcommand == "get":
                study_id = config["selected_study"] if args.study_id is None else args.study_id
                if not study_id or study_id.isspace():
                    print("Please select a study or pass a study id")
                    return
                study = await dfxapi.Studies.retrieve(session, study_id)
                print(json.dumps(study)) if args.json else print_pretty(study, args.csv)
            elif args.subcommand == "list":
                studies = await dfxapi.Studies.list(session)
                print(json.dumps(studies)) if args.json else print_pretty(studies, args.csv)
            elif args.subcommand == "select":
                config["selected_study"] = args.study_id
                save_config(config, args.config_file)
        return

    # Retrieve or list measurements
    if args.command == "measure" and args.subcommand != "make":
        async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
            if args.subcommand == "get":
                measurement_id = config["last_measurement"] if args.measurement_id is None else args.measurement_id
                if not measurement_id or measurement_id.isspace():
                    print("Please complete a measurement first or pass a measurement id")
                    return
                results = await dfxapi.Measurements.retrieve(session, measurement_id)
                print(json.dumps(results)) if args.json else print_meas(results, args.csv)
            elif args.subcommand == "list":
                measurements = await dfxapi.Measurements.list(session, limit=args.limit)
                print(json.dumps(measurements)) if args.json else print_pretty(measurements, args.csv)
        return

    # Make a measurement
    assert args.command == "measure" and args.subcommand == "make"

    # Verify preconditions
    if not config["selected_study"]:
        print("Please select a study first using 'study select'")
        return
    payload_files = sorted(glob.glob(os.path.join(args.payloads_folder, "payload*.bin")))
    meta_files = sorted(glob.glob(os.path.join(args.payloads_folder, "metadata*.bin")))
    prop_files = sorted(glob.glob(os.path.join(args.payloads_folder, "properties*.json")))
    number_files = min(len(payload_files), len(meta_files), len(prop_files))
    if number_files <= 0:
        print(f"No payload files found in {args.payloads_folder}")
        return
    with open(prop_files[0], 'r') as pr:
        props = json.load(pr)
        number_chunks_pr = props["number_chunks"]
        duration_pr = props["duration_s"]
    if number_chunks_pr != number_files:
        print(f"Number of chunks in properties.json {number_chunks_pr} != Number of payload files {number_files}")
        return
    if duration_pr * number_chunks_pr > 120:
        print(f"Total payload duration {duration_pr * number_chunks_pr} seconds is more than 120 seconds")
        return

    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        # Create a measurement
        measurement_id = await dfxapi.Measurements.create(session, config["selected_study"])
        print(f"Created measurement {measurement_id}")

        measurement_files = zip(payload_files, meta_files, prop_files)
        await measure_websocket(session, measurement_id, measurement_files, number_chunks_pr)

        print(f"Measurement {measurement_id} complete")
        config["last_measurement"] = measurement_id
        save_config(config, args.config_file)


def load_config(config_file):
    config = {
        "device_id": "",
        "device_token": "",
        "role_id": "",
        "user_id": "",
        "user_token": "",
        "selected_study": "",
        "last_measurement": ""
    }
    if os.path.isfile(config_file):
        with open(config_file, "r") as c:
            read_config = json.loads(c.read())
            config = {**config, **read_config}

    dfxapi.Settings.device_id = config["device_id"]
    dfxapi.Settings.device_token = config["device_token"]
    dfxapi.Settings.role_id = config["role_id"]
    dfxapi.Settings.role_id = config["role_id"]
    dfxapi.Settings.user_token = config["user_token"]

    return config


def save_config(config, config_file):
    with open(config_file, "w") as c:
        c.write(json.dumps(config, indent=4))
        print(f"Config updated in {config_file}")


def generate_reqid():
    return "".join(random.choices(string.ascii_letters, k=10))


def determine_action(chunk_number, number_chunks):
    action = 'CHUNK::PROCESS'
    if chunk_number == 0 and number_chunks > 1:
        action = 'FIRST::PROCESS'
    elif chunk_number == number_chunks - 1:
        action = 'LAST::PROCESS'
    return action


async def register(config, config_file, license_key):
    if dfxapi.Settings.device_token:
        print("Already registered")
        return False

    # TODO: Handle 404 properly here...
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        await dfxapi.Organizations.register_license(session, license_key, "LINUX", "DFX Example", "DFXCLIENT", "0.0.1")
        config["device_id"] = dfxapi.Settings.device_id
        config["device_token"] = dfxapi.Settings.device_token
        config["role_id"] = dfxapi.Settings.role_id
        config["user_token"] = dfxapi.Settings.user_token
        print(f"Register successful with new device id {config['device_id']}")
    return True


async def unregister(config, config_file):
    if not dfxapi.Settings.device_token:
        print("Not registered")
        return False

    headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        await dfxapi.Organizations.unregister_license(session)
        print(f"Unregister successful for device id {config['device_id']}")
        config["device_id"] = ""
        config["device_token"] = ""
        config["role_id"] = ""
    return True


async def login(config, config_file, email, password):
    if dfxapi.Settings.user_token:
        print("Already logged in")
        return False

    if not dfxapi.Settings.device_token:
        print("Please register first to obtain a device_token")
        return False

    headers = {"Authorization": f"Bearer {dfxapi.Settings.device_token}"}
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        await dfxapi.Users.login(session, email, password)
        config["user_token"] = dfxapi.Settings.user_token
        print("Login successful")
    return True


def logout(config, config_file):
    config["user_token"] = ""
    config["user_id"] = ""
    print("Logout successful")
    return True


async def measure_websocket(session, measurement_id, measurement_files, number_chunks):
    # Use the session to connect to the WebSocket
    async with session.ws_connect(dfxapi.Settings.ws_url) as ws:
        # Subscribe to results
        results_request_id = generate_reqid()
        await dfxapi.Measurements.ws_subscribe_to_results(ws, results_request_id, measurement_id)

        # Use this to close WebSocket in the receive loop
        results_expected = number_chunks + 1

        async def send_chunks():
            # Coroutine to iterate through the payload files and send chunks using WebSocket
            for payload_file, meta_file, prop_file in measurement_files:
                with open(payload_file, 'rb') as p, open(meta_file, 'rb') as m, open(prop_file, 'r') as pr:
                    payload_bytes = p.read()
                    meta_bytes = m.read()
                    props = json.load(pr)

                    # Determine action and request id
                    action = determine_action(props["chunk_number"], props["number_chunks"])
                    request_id = generate_reqid()

                    # Add data
                    await dfxapi.Measurements.ws_add_data(ws, generate_reqid(), measurement_id, props["chunk_number"],
                                                          action, props["start_time_s"], props["end_time_s"],
                                                          props["duration_s"], meta_bytes, payload_bytes)
                    sleep_time = props["duration_s"]
                    print(
                        f"Sent chunk req#:{request_id} - {action} ...waiting {sleep_time:.0f} seconds instead of {props['duration_s']:.0f}..."
                    )

                    # Sleep to simulate a live measurement and not hit the rate limit
                    await asyncio.sleep(sleep_time)

        async def receive_results():
            # Coroutine to receive results
            num_results_received = 0
            async for msg in ws:
                status, request_id, payload = dfxapi.Measurements.ws_decode(msg)
                if request_id == results_request_id:
                    print(f"  Received result - {len(payload)} bytes {payload[:80]}")
                    num_results_received += 1
                if num_results_received == results_expected:
                    await ws.close()
                    break

        # Start the two coroutines and await till they finish
        await asyncio.gather(send_chunks(), receive_results())


def cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./config.json")
    pp_group = parser.add_mutually_exclusive_group()
    pp_group.add_argument("--json", help="Print as JSON", action="store_true", default=False)
    pp_group.add_argument("--csv", help="Print grids as CSV", action="store_true", default=False)

    subparser_top = parser.add_subparsers(dest="command", required=True)
    subparser_orgs = subparser_top.add_parser("org", help="Organizations").add_subparsers(dest="subcommand",
                                                                                          required=True)
    register_parser = subparser_orgs.add_parser("register", help="Register device")
    register_parser.add_argument("license_key", help="DFX API Organization License")
    unregister_parser = subparser_orgs.add_parser("unregister", help="Unregister device")

    subparser_users = subparser_top.add_parser("user", help="Users").add_subparsers(dest="subcommand", required=True)
    login_parser = subparser_users.add_parser("login", help="User login")
    login_parser.add_argument("email", help="Email address")
    login_parser.add_argument("password", help="Password")
    logout_parser = subparser_users.add_parser("logout", help="User logout")

    subparser_studies = subparser_top.add_parser("study", help="Studies").add_subparsers(dest="subcommand",
                                                                                         required=True)
    study_list_parser = subparser_studies.add_parser("list", help="List existing studies")
    study_get_parser = subparser_studies.add_parser("get", help="Retrieve a study's information")
    study_get_parser.add_argument("study_id",
                                  nargs="?",
                                  help="ID of study to retrieve (default: selected study)",
                                  type=str)
    study_list_parser = subparser_studies.add_parser("select", help="Select a study to use")
    study_list_parser.add_argument("study_id", help="ID of study to use", type=str)

    subparser_meas = subparser_top.add_parser("measure", help="Measurements").add_subparsers(dest="subcommand",
                                                                                             required=True)
    make_parser = subparser_meas.add_parser("make", help="Make a measurement")
    make_parser.add_argument("payloads_folder", help="Folder containing payloads", type=str)
    list_parser = subparser_meas.add_parser("list", help="List existing measurements")
    list_parser.add_argument("--limit", help="Number of measurements to retrieve (default : 10)", type=int, default=10)
    get_parser = subparser_meas.add_parser("get", help="Retrieve a measurement")
    get_parser.add_argument("measurement_id",
                            nargs="?",
                            help="ID of measurement to retrieve (default: last measurement)",
                            default=None)

    args = parser.parse_args()

    # asyncio.run(main(args))  # https://github.com/aio-libs/aiohttp/issues/4324

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    loop.run_until_complete(asyncio.sleep(0.25))  # https://github.com/aio-libs/aiohttp/issues/1925
    loop.close()


if __name__ == '__main__':
    cmdline()
