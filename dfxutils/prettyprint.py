import datetime

TIMESTAMP_KEYS = ["Created", "Updated"]


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
        if measurement_results["Results"]:
            grid_results = []
            for signal_id, signal_name in measurement_results["SignalNames"].items():
                result_data = measurement_results["Results"][signal_id][0]
                units = measurement_results["SignalUnits"][signal_id]
                description = measurement_results["SignalDescriptions"][signal_id]
                if not csv and len(description) > 100:
                    description = description[:100] + "..."
                result_value = sum(result_data["Data"]) / len(result_data["Data"]) / result_data["Multiplier"] if len(
                    result_data["Data"]) > 0 and result_data["Multiplier"] > 0 else None
                grid_result = {
                    "ID": signal_id,
                    "Name": signal_name,
                    "Value": result_value,
                    "Unit": units if units is not None else "",
                    "Category": measurement_results["SignalConfig"][signal_id]["category"],
                    "Description": description
                }
                grid_results.append(grid_result)
            measurement_results["Results"] = grid_results
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
                print(indent * " " + item)

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

        col_widths = [len(str(k)) for k in list_of_dicts[0].keys()]
        for dict_ in list_of_dicts:
            for i, v in enumerate(dict_.values()):
                col_widths[i] = max(col_widths[i], len(str(v)))
        print(indent * " " + "".join([f"{str(key):{cw}} " for (cw, key) in zip(col_widths, list_of_dicts[0].keys())]))
        for dict_ in list_of_dicts:
            print(indent * " " + "".join([f"{str(value):{cw}} " for (cw, value) in zip(col_widths, dict_.values())]))
