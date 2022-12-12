import datetime
import shutil

TIMESTAMP_KEYS = ["Created", "Updated"]
TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size()
MIN_COL_WIDTH = 5


class PrettyPrinter():
    @staticmethod
    def print_pretty(x, csv=False, indent=0) -> None:
        if type(x) == list:
            PrettyPrinter._print_list(x, csv, indent)
        elif type(x) == dict:
            PrettyPrinter._print_dict(x, csv, indent)
        else:
            print(x)

    @staticmethod
    def print_sdk_result(dict_result):
        if not dict_result:
            print("Received invalid sdk_result from DFX SDK collector decode!!")
            return

        print(f"Received chunk {dict_result['chunk_number']}")
        del dict_result["chunk_number"]

        PrettyPrinter.print_pretty(dict_result, indent=2)

    @staticmethod
    def print_result(measurement_results, csv=False):
        if "Results" in measurement_results and measurement_results["Results"] is not None:
            grid_results = []
            notes = {}
            for signal_id, signal_name in measurement_results["SignalNames"].items():
                result_data = measurement_results["Results"][signal_id][0]
                units = measurement_results["SignalUnits"][signal_id]
                description = measurement_results["SignalDescriptions"][signal_id]
                result_value = sum(result_data["Data"]) / len(result_data["Data"]) / result_data["Multiplier"] if len(
                    result_data["Data"]) > 0 and result_data["Multiplier"] > 0 else None
                notes_present = None
                if "Notes" in result_data and len(result_data["Notes"]) > 0:
                    notes_present = "Yes"
                    notes[signal_id] = ", ".join(note for note in (note.replace("NOTE_", "")
                                                                   for note in result_data["Notes"]))
                grid_result = {
                    "ID": signal_id,
                    "Name": signal_name,
                    "Value": result_value,
                    "Unit": units if units is not None else "",
                    "Notes": notes_present,
                    "Category": measurement_results["SignalConfig"][signal_id]["category"],
                    "Description": description
                }
                grid_results.append(grid_result)
            measurement_results["Results"] = grid_results
            if len(notes) > 0:
                measurement_results["Notes"] = notes
            del measurement_results["SignalNames"]
            del measurement_results["SignalUnits"]
            del measurement_results["SignalDescriptions"]
            del measurement_results["SignalConfig"]
        PrettyPrinter.print_pretty(measurement_results, csv)

    @staticmethod
    def _print_dict(dict_, csv, indent) -> None:
        sep = "," if csv else ": "
        for k, v in dict_.items():
            if type(v) == list:
                print(indent * " " + f"{k}{sep}")
                PrettyPrinter._print_list(v, csv, indent + 2)
            elif type(v) == dict:
                print(indent * " " + f"{k}{sep}")
                PrettyPrinter._print_dict(v, csv, indent + 2)
            else:
                if v is None:
                    vv = ""
                elif k in TIMESTAMP_KEYS:
                    vv = datetime.datetime.fromtimestamp(v)
                else:
                    vv = v
                print(indent * " " + f"{k}{sep}{vv}")

    @staticmethod
    def _print_list(list_, csv, indent):
        if len(list_) > 0 and type(list_[0]) == dict:
            PrettyPrinter._print_list_of_dicts(list_, csv, indent)
            return

        for item in list_:
            if type(item) == list:
                PrettyPrinter._print_list(item, csv, indent + 2)
            elif type(item) == dict:
                PrettyPrinter._print_dict(item, csv, indent + 2)
            else:
                print(indent * " " + f"{item}")

    @staticmethod
    def _print_list_of_dicts(list_of_dicts, csv, indent):
        if len(list_of_dicts) <= 0:
            return

        for dict_ in list_of_dicts:
            for k, v in dict_.items():
                if v is None:
                    dict_[k] = ""
                elif k in TIMESTAMP_KEYS:
                    ts = datetime.datetime.fromtimestamp(v)
                    if csv:
                        dict_[k] = str(ts)
                    else:
                        dict_[k] = ts.strftime("%Y-%m-%d")

        if csv:
            print(indent * " " + ",".join([f"{key}" for key in list_of_dicts[0].keys()]))
            for dict_ in list_of_dicts:
                print(indent * " " + ",".join([f"{value}" for value in dict_.values()]))
            return

        # Find the column widths
        col_widths = [len(str(k)) for k in list_of_dicts[0].keys()]
        for dict_ in list_of_dicts:
            for i, v in enumerate(dict_.values()):
                col_widths[i] = max(col_widths[i], len(str(v)))

        # If the sum of column widths exceeds the width of the terminal
        spaces_between_cols = len(list_of_dicts[0].keys())
        excess = sum(col_widths) + indent + spaces_between_cols - TERM_WIDTH
        if excess > 0 and "Description" in list_of_dicts[0].keys():
            # Shrink the "Description" column by the minimum amount possible
            idx = list(list_of_dicts[0].keys()).index("Description")
            col_widths[idx] = max(col_widths[idx] - excess, MIN_COL_WIDTH)
            for dict_ in list_of_dicts:
                if len(dict_["Description"]) > col_widths[idx]:
                    dict_["Description"] = dict_["Description"][:col_widths[idx] - 3] + "..."

        # Print the table header and table
        print(indent * " " + "".join([f"{str(key):{cw}} " for (cw, key) in zip(col_widths, list_of_dicts[0].keys())]))
        for dict_ in list_of_dicts:
            print(indent * " " + "".join([f"{str(value):{cw}} " for (cw, value) in zip(col_widths, dict_.values())]))
