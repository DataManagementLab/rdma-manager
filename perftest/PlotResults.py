import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
from matplotlib.backends.backend_pdf import PdfPages
from os import walk

# [ HOW TO USE ]
# -----------------------------------------------
#
# python PlotResults.py <csvFile> <outputFormat>
#   csvFile:        File name or . to automatically find csvFile (optional)
#   outputFormat:   pdf, jpg, png, emf, eps, ps, raw, rgba, svg, svgz
#
# -----------------------------------------------

CSV_DELIMITER = ','
CSV_LINEBREAK = '\n'
CSV_COMMA = '.'

mpl.rcParams["path.simplify"] = True  # lines will be simplified
mpl.rcParams["path.simplify_threshold"] = 1.0  # how much lines should be simplified
plt.style.use("fast")  # automatically chunks lines and speeds up rendering


class TestParameters:
    is_server = False
    threads = None
    buffer_slots = None
    packet_size = None
    memory_size = None
    local_memory_type = None
    remote_memory_type = None
    iterations = None
    write_mode = None

    def __str__(self):
        return self.__dict__.__str__()

    def to_file_str(self) -> str:
        return "thr=" + str(self.threads) + "-bs=" + str(self.buffer_slots) + "-ps=" + \
               str(self.packet_size) + "-ms=" + str(self.memory_size) + "-mt=" + \
               str(self.local_memory_type) + "_to_" + str(self.remote_memory_type) + "-itr=" + \
               str(self.iterations) + "-wm=" + str(self.write_mode)


def find_csv_file():
    for dirpath, dirnames, filenames in walk("../"):
        for filename in filenames:
            if filename.endswith(".csv"):
                return dirpath + filename
    return None


def get_line_style(column_name: str) -> {}:
    column_name = column_name.lower()
    output = {
        "color": "k",
        "linestyle": "-",
        "marker": "None",
        "alpha": 0.75,
        "linewidth": 1.5,
    }
    if column_name.find("write") >= 0:
        output["color"] = "b"
    elif column_name.find("read") >= 0:
        output["color"] = "r"
    elif column_name.find("send") >= 0:
        output["color"] = "g"

    if column_name.find("min") >= 0:
        output["linestyle"] = ":"
        output["marker"] = "^"
        output["markersize"] = 2.0
        output["alpha"] = 0.25
        output["linewidth"] = 0.5
    elif column_name.find("max") >= 0:
        output["linestyle"] = ":"
        output["marker"] = "v"
        output["markersize"] = 2.0
        output["alpha"] = 0.25
        output["linewidth"] = 0.5
    elif column_name.find("av") >= 0:
        output["linestyle"] = "-."
        output["alpha"] = 0.5
        output["linewidth"] = 0.85
    elif column_name.find("med") >= 0:
        output["linestyle"] = "--"
        output["alpha"] = 0.5
        output["linewidth"] = 0.85

    return output


def group_columns(columns: [{}]) -> [{}]:
    cols = []
    cols.extend(columns)
    tmp = []

    def process_columns_by_name(search: str):
        for column in columns:
            if str(column["name"]).lower().find(search) >= 0:
                cols.remove(column)
                tmp.append(column, )

    process_columns_by_name("write")
    process_columns_by_name("read")
    process_columns_by_name("send")
    process_columns_by_name("fetch")
    process_columns_by_name("comp")
    output = []
    output.extend(cols)
    output.extend(tmp)
    return output


def plot_bandwidth(test_params, test_columns, output_file_name, output_format):
    print("Plotting Bandwidth results ...")
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    y_global_min = sys.maxsize
    y_global_max = -sys.maxsize - 1
    for entry in range(entry_count):
        columns = group_columns(test_columns[entry])
        test_columns[entry] = columns
        for column in columns:
            if str(column["name"]).lower().startswith("min"):
                for val in column["values"]:
                    if val < y_global_min:
                        y_global_min = val
            elif str(column["name"]).lower().startswith("max"):
                for val in column["values"]:
                    if val > y_global_max:
                        y_global_max = val

    # x-axis = Packet Size | y-axis =
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        print(params)  # TODO REMOVE
        print(columns)  # TODO REMOVE
        title = params.local_memory_type + "→" + params.remote_memory_type + " (thrs=" + \
            str(params.threads) + "; bufslots=" + \
            str(params.buffer_slots) + "; itrs=" + str(params.iterations) + \
            "; wm=" + str(params.write_mode) + ")"
        y_label = "Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        for column in columns:
            col_label = column["name"]
            if col_label.rfind("/s") >= 0:
                unit_off = col_label.rfind("[")
                if unit_off > 0:
                    if y_label_update:
                        y_label = "Bandwidth " + col_label[unit_off:]
                        y_label_update = False
                    col_label = col_label[0:unit_off].strip()
                lines = ax.plot(x_values, np.array(column["values"]), label=col_label,
                                **get_line_style(col_label))
                lines[0].set_antialiased(False)

        # ylim=[y_global_min, y_global_max]
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=title)
        ax.legend()

        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "BANDWIDTH-" + params.to_file_str() + "-Raw" + "." +
                        output_format)
        plt.close(fig)

    # x-axis = Packet Size | y-axis = Medians compared to thread counts


def plot_latency(test_params, test_columns, output_file_name, output_format):
    print("Plotting Latency results ...")
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    y_global_min = sys.maxsize
    y_global_max = -sys.maxsize - 1
    for entry in range(entry_count):
        columns = group_columns(test_columns[entry])
        test_columns[entry] = columns
        for column in columns:
            if str(column["name"]).lower().startswith("min"):
                for val in column["values"]:
                    if val < y_global_min:
                        y_global_min = val
            elif str(column["name"]).lower().startswith("max"):
                for val in column["values"]:
                    if val > y_global_max:
                        y_global_max = val

    # x-axis = Packet Size | y-axis =
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        print(params)  # TODO REMOVE
        print(columns)  # TODO REMOVE
        title = params.local_memory_type + "→" + params.remote_memory_type + " (thrs=" + \
            str(params.threads) + "; bufslots=" + \
            str(params.buffer_slots) + "; itrs=" + str(params.iterations) + \
            "; wm=" + str(params.write_mode) + ")"
        y_label = "Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        for column in columns:
            col_label = column["name"]
            if col_label.rfind("sec") >= 0:
                unit_off = col_label.rfind("[")
                if unit_off > 0:
                    if y_label_update:
                        y_label = "Latency " + col_label[unit_off:]
                        y_label_update = False
                    col_label = col_label[0:unit_off].strip()
                lines = ax.plot(x_values, np.array(column["values"]), label=col_label,
                                **get_line_style(col_label))
                lines[0].set_antialiased(False)

        # ylim=[y_global_min, y_global_max]
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=title)
        ax.legend()

        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "LATENCY-" + params.to_file_str() + "-Raw" + "." +
                        output_format)
        plt.close(fig)

    # x-axis = Packet Size | y-axis = Medians compared to thread counts


def plot_operations_count(test_params, test_columns, output_file_name, output_format):
    print("Plotting OperationsCount results ...")
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    y_global_min = sys.maxsize
    y_global_max = -sys.maxsize - 1
    for entry in range(entry_count):
        columns = group_columns(test_columns[entry])
        test_columns[entry] = columns
        for column in columns:
            if str(column["name"]).lower().startswith("min"):
                for val in column["values"]:
                    if val < y_global_min:
                        y_global_min = val
            elif str(column["name"]).lower().startswith("max"):
                for val in column["values"]:
                    if val > y_global_max:
                        y_global_max = val

    # x-axis = Packet Size | y-axis =
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        print(params)  # TODO REMOVE
        print(columns)  # TODO REMOVE
        title = params.local_memory_type + "→" + params.remote_memory_type + " (thrs=" + \
            str(params.threads) + "; bufslots=" + str(params.buffer_slots) + "; itrs=" + \
            str(params.iterations) + "; wm=" + str(params.write_mode) + ")"
        y_label = "Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        for column in columns:
            col_label = column["name"]
            if col_label.rfind("/s") >= 0:
                unit_off = col_label.rfind("[")
                if unit_off > 0:
                    if y_label_update:
                        y_label = "Operations/sec " + col_label[unit_off:]
                        y_label_update = False
                    col_label = col_label[0:unit_off].strip()
                lines = ax.plot(x_values, np.array(column["values"]), label=col_label,
                                **get_line_style(col_label))
                lines[0].set_antialiased(False)

        # ylim=[y_global_min, y_global_max]
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=title)
        ax.legend()

        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "OPERATIONS_COUNT-" + params.to_file_str() + "-Raw" +
                        "." + output_format)
        plt.close(fig)

    # x-axis = Packet Size | y-axis = Medians compared to thread counts


def plot_atomics_bandwidth(test_params, test_columns, output_file_name, output_format):
    print("Plotting AtomicsBandwidth results ...")


def plot_atomics_latency(test_params, test_columns, output_file_name, output_format):
    print("Plotting AtomicsLatency results ...")


def plot_atomics_operations_count(test_params, test_columns, output_file_name, output_format):
    print("Plotting AtomicsOperationsCount results ...")


def parse_test_parameters(test_params: [str]) -> TestParameters:
    def parse_int_value(key_value: str) -> int:
        start_index = key_value.find("=") + 1
        if start_index <= 0:
            return None
        end_index = key_value.rfind("(")
        if end_index < 0:
            end_index = len(key_value)
        try:
            return int(key_value[start_index:end_index].strip())
        except ValueError:
            return None

    def parse_str_value(key_value: str) -> str:
        try:
            return key_value[key_value.index("=") + 1:].strip()
        except ValueError:
            return None

    p = TestParameters()
    for raw_param in test_params:
        param = str(raw_param).strip().lower()
        i = parse_int_value(param)
        if i:
            if param.startswith("thr"):
                p.threads = i
            elif param.startswith("buf"):
                p.buffer_slots = i
            elif param.startswith("pack"):
                p.packet_size = i
            elif param.startswith("mem"):
                p.memory_size = i
            elif param.startswith("it"):
                p.iterations = i
            else:
                sys.stderr.write("Unknown parameter '" + param + " with value '" + i + "'\n")
        else:
            s = parse_str_value(raw_param)
            if s:
                if param.startswith("mem"):
                    s = s.split("->")
                    p.local_memory_type = s[0].strip()
                    if len(s) > 1:
                        p.remote_memory_type = s[1].strip()
                elif param.startswith("w"):
                    p.write_mode = s
            elif param.startswith("cli"):
                p.is_server = False
            elif param.startswith("serv"):
                p.is_server = True
            else:
                sys.stderr.write("Unknown parameter '" + param + "\n")
    return p


def plot_test_values(test_name: str, test_params: [TestParameters], test_columns: {},
                     output_file_name, output_format):
    test = str(test_name).strip().replace(" ", "").lower()
    if test == "bandwidth":
        plot_bandwidth(test_params, test_columns, output_file_name, output_format)
    elif test == "latency":
        plot_latency(test_params, test_columns, output_file_name, output_format)
    elif test == "operationscount":
        plot_operations_count(test_params, test_columns, output_file_name, output_format)
    elif test == "atomicsbandwidth":
        plot_atomics_bandwidth(test_params, test_columns, output_file_name, output_format)
    elif test == "atomicslatency":
        plot_atomics_latency(test_params, test_columns, output_file_name, output_format)
    elif test == "atomicsoperationscount":
        plot_atomics_operations_count(test_params, test_columns, output_file_name, output_format)
    else:
        sys.stderr.write("Cannot print unknown test '%s'\n" % test_name)


def plot_csv_file(csv_file_name, output_file_name, output_format):
    """
    Plots the values of a given CSV file and stores plots in an output file
    :param csv_file_name: the values should be read from
    :param output_file_name: name of the output file without format at end
    :param output_format: format of the output (will be appended on file name)
    :return:
    """
    with open(csv_file_name, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_LINEBREAK)
        test_index = 0
        parse_labels = True
        prev_test_name = ""
        test_name = ""
        test_params = []  # [test_index: [params]]
        test_columns = {}  # {test_index: [{ name=str, values=[values] }]}
        if output_format == "pdf":
            output_file_name = PdfPages(output_file_name + "." + output_format)
        else:
            try:
                os.mkdir(output_file_name)
            except OSError:
                pass
            output_file_name = output_file_name + "/"
        for row in reader:
            if len(row) == 0:  # prepare for next test block
                prev_test_name = test_name.lower()
                test_name = ""
                continue
            if len(test_name) == 0:
                test_name = str(row[0]).strip()
                if test_name.lower() != prev_test_name:  # reset because new test name
                    if test_index > 0:
                        plot_test_values(prev_test_name, test_params, test_columns,
                                         output_file_name, output_format)
                    test_index = 0
                    test_params = []
                    test_columns = {}
                else:
                    test_index = test_index + 1
                test_params.append(parse_test_parameters(row[1:]), )
                parse_labels = True
                continue
            if parse_labels:
                parse_labels = False
                test_columns[test_index] = []
                for i in range(len(row)):
                    test_columns[test_index].append({
                        "name": str(row[i]).strip(),
                        "values": []
                    })
                continue
            for i in range(len(row)):
                row_value = str(row[i])
                if CSV_COMMA in row_value:
                    try:
                        row_value = float(row_value)
                    except ValueError:
                        pass
                else:
                    try:
                        row_value = int(row_value)
                    except ValueError:
                        pass
                column = test_columns[test_index]
                column[i]["values"].append(row_value)
                test_columns[test_index] = column
        if test_index > 0:
            plot_test_values(test_name, test_params, test_columns, output_file_name, output_format)

        if output_file_name and isinstance(output_file_name, PdfPages):
            output_file_name.close()


# Parse line arguments
csvFileName = None
outputFormat = None
if len(sys.argv) <= 1:
    csvFileName = find_csv_file()
else:
    csvFileName = sys.argv[1]
    if csvFileName == ".":
        csvFileName = find_csv_file()
if len(sys.argv) <= 2:
    outputFormat = "pdf"
else:
    outputFormat = sys.argv[2].strip().lower()

# Check validity of line arguments
if csvFileName is None:
    raise FileNotFoundError("No CSV file found and also not provided as line argument")
if not os.path.exists(csvFileName):
    raise FileNotFoundError("File '%s' does not exist" % csvFileName)
if not os.path.isfile(csvFileName):
    raise FileNotFoundError("File '%s' is not a file" % csvFileName)
if outputFormat not in plt.figure().canvas.get_supported_filetypes():
    raise LookupError("Output format '%s' is not supported" % outputFormat)

lastDot = csvFileName.rfind(".")
outputFileName = (csvFileName if lastDot < 0 else csvFileName[0:lastDot])

print("Plotting CSV file '%s' as %s('%s') ..." % (csvFileName, outputFormat.upper(),
                                                  outputFileName,))

plot_csv_file(csvFileName, outputFileName, outputFormat)
