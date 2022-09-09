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

MAX_LINES_PER_PLOT_BEFORE_SPLITTING = 12

mpl.rcParams["path.simplify"] = True  # lines will be simplified
mpl.rcParams["path.simplify_threshold"] = 1.0  # how much lines should be simplified
plt.style.use("fast")  # automatically chunks lines and speeds up rendering

# Constant that is depending on implementation (DO NOT CHANGE)
MIN_LINES_PER_COMPARE_PLOT = 7  # x-axis + y-axis (avg & median * write & read & send/recv)


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
        return self.__class__.__name__ + self.__dict__.__str__()

    def __repr__(self):
        return self.__str__()

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


def parse_bracket_value(label: str) -> str:
    start = label.find("(")
    label = label[(start + 1):] if start >= 0 else label
    end = label.rfind(")")
    label = label[:end] if end >= 0 else label
    numbers = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    num = ""
    for i in range(len(label)):
        if label[i] in numbers:
            num = num + label[i]
        else:
            break
    return label if len(num) <= 0 else int(num)


def get_line_color(label: str, ordered_possible_values: []) -> str:
    h1 = "ff"
    h2 = "00"
    value = parse_bracket_value(label)
    values_len = len(ordered_possible_values)
    for i in range(values_len):
        if value == ordered_possible_values[i]:
            h1 = hex(75 + int((values_len - i) * 180 / values_len))[2:]  # remove '0x' at
            # beginning
            while len(h1) < 2:
                h1 = "0" + h1
            h2 = hex((i % 4) * 85)[2:]  # remove '0x' at beginning
            while len(h2) < 2:
                h2 = "0" + h2

    if label.find("write") >= 0:
        return "#00" + h2 + h1
    elif label.find("read") >= 0:
        return "#" + h1 + h2 + "00"
    elif label.find("send") >= 0:
        return "#" + h2 + h1 + "00"
    elif label.find("fet") >= 0:
        return "#" + h2 + h1 + h1
    elif label.find("swap") >= 0:
        return "#" + h1 + h1 + h2
    h = hex(255 - int(h1, 16))[2:]  # remove '0x' at beginning
    while len(h) < 2:
        h = "0" + h
    return "#" + h + h + h


def get_bandwidth_raw_line_style(column_name: str) -> {}:
    column_name = column_name.lower()
    output = {
        "color": get_line_color(column_name, []),
        "linestyle": "-",
        "marker": "None",
        "alpha": 0.75,
        "linewidth": 1.0,
    }
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
        output["linewidth"] = 0.75
    elif column_name.find("med") >= 0:
        output["linestyle"] = "--"
        output["alpha"] = 0.5
        output["linewidth"] = 0.75
    return output


def get_compare_line_style(column_name: str, ordered_possible_values: [int]):
    """ Used for comparing by bandwidth, latency and operations/sec """
    column_name = column_name.lower()
    output = {
        "color": get_line_color(column_name, ordered_possible_values),
        "linestyle": "-",
        "marker": "None",
        "alpha": 0.75,
        "linewidth": 1.0,
    }
    if column_name.find("av") >= 0:
        output["linestyle"] = "-"
    elif column_name.find("med") >= 0:
        output["linestyle"] = "--"
    return output


def get_latency_raw_line_style(column_name: str) -> {}:
    column_name = column_name.lower()
    output = {
        "color": get_line_color(column_name, []),
        "linestyle": "-",
        "marker": "None",
        "alpha": 0.75,
        "linewidth": 1.0,
    }
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
    elif column_name.find("med") >= 0:
        output["linestyle"] = "--"
    return output


def get_operations_count_raw_line_style(column_name: str) -> {}:
    column_name = column_name.lower()
    output = {
        "color": get_line_color(column_name, []),
        "linestyle": "-",
        "marker": "None",
        "alpha": 0.75,
        "linewidth": 1.0,
    }
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
    elif column_name.find("med") >= 0:
        output["linestyle"] = "--"
    return output


def remove_unit_from_label(label: str) -> (str, str):
    start = label.find("[")
    if start < 0:
        return label.strip(), None
    end = label.rfind("]")
    if end < 0:
        return label[:start].strip(), "[" + label[(start + 1):].strip() + "]"
    return (label[:start] + label[(end + 1):].strip()).strip(), \
        "[" + label[(start + 1):end].strip() + "]"


def filter_columns(columns: [{}], search: [str]) -> [{}]:
    cols = []
    for column in columns:
        name = str(column["name"]).lower()
        for s in search:
            if name.find(s) >= 0:
                cols.append(column)
    return cols


def sort_columns(columns: [{}]) -> [{}]:
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
    process_columns_by_name("swap")
    output = []
    output.extend(cols)
    output.extend(tmp)
    return output


def split_plot_by_write_read_send(plot: {}) -> [{}]:
    # plot is in the form of {title: str, columns: [{}] }
    title = plot["title"]
    columns = plot["columns"]

    def get_filtered_columns(search: str) -> [{}]:
        result = []
        first = True
        for column in columns:
            if first or str(column["name"]).lower().find(search) >= 0:
                result.append(column)
            first = False
        return result

    return [{
        "title": title + " for write",
        "columns": get_filtered_columns("write")
    }, {
        "title": title + " for read",
        "columns": get_filtered_columns("read")
    }, {
        "title": title + " for send/recv",
        "columns": get_filtered_columns("send")
    }]


def split_plot_by_fetch_compare(plot: {}) -> [{}]:
    # plot is in the form of {title: str, columns: [{}] }
    title = plot["title"]
    columns = plot["columns"]

    def get_filtered_columns(search: str) -> [{}]:
        result = []
        first = True
        for column in columns:
            if first or str(column["name"]).lower().find(search) >= 0:
                result.append(column)
            first = False
        return result

    return [{
        "title": title + " for fetch&add",
        "columns": get_filtered_columns("fetch")
    }, {
        "title": title + " for compare&swap",
        "columns": get_filtered_columns("swap")
    }]


def transform_and_sort_plots(entries: [{}]) -> [{}]:
    """ Input format:  [ {params:TestParameters, plots:[{title:str, columns:[{}] }] } ]
        Output format: [ {title:str, params:TestParameters, columns:[{}] } ]"""
    output = []
    entries = list(entries)

    def process_plots_by_name(ents: [{}], search: str) -> [{}]:
        temp = []
        while len(ents) > 0:
            entry = ents.pop(0)
            params = entry["params"]
            plots = entry["plots"]
            tmp = []
            while len(plots) > 0:
                plot = plots.pop(0)
                title = plot["title"]
                if str(title).lower().find(search) >= 0:
                    output.append({
                        "title": title,
                        "params": params,
                        "columns": sort_columns(plot["columns"])
                    }, )
                else:
                    tmp.append(plot)
            if len(tmp) > 0:
                entry["plots"] = tmp
                temp.append(entry)
        return temp

    entries = process_plots_by_name(entries, "write")
    entries = process_plots_by_name(entries, "read")
    entries = process_plots_by_name(entries, "send")
    entries = process_plots_by_name(entries, "fetch")
    entries = process_plots_by_name(entries, "swap")
    process_plots_by_name(entries, "")  # append remaining
    return output


def plot_bandwidth(test_params: [TestParameters], test_columns: [{}], output_file_name,
                   output_format):
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    memory_types_possible_values = set()
    compare_memory_types = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    threads_possible_values = set()
    compare_threads = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    buffer_slots_possible_values = set()
    compare_buffer_slots = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    iterations_possible_values = set()
    compare_iterations = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    write_modes_possible_values = set()
    compare_write_modes = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    for entry in range(entry_count):
        params = test_params[entry]
        columns = sort_columns(filter_columns(test_columns[entry], ("/s", "size")))
        test_columns[entry] = columns

        # initialize compare memory types
        mem_type = params.local_memory_type + "→" + params.remote_memory_type
        memory_types_possible_values.add(mem_type.lower())
        compare_memory_types_key = str(params.threads) + "-" + str(params.buffer_slots) + "-" + \
            str(params.iterations) + "-" + str(params.write_mode)
        if compare_memory_types_key not in compare_memory_types:
            compare_memory_types[compare_memory_types_key] = {
                "params": params,
                "plots": [{
                    "title": "Bandwidth compare memory types",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare thread counts
        threads_possible_values.add(params.threads)
        compare_threads_key = str(params.buffer_slots) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_threads_key not in compare_threads:
            compare_threads[compare_threads_key] = {
                "params": params,
                "plots": [{
                    "title": "Bandwidth compare thread counts",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare buffer slots
        buffer_slots_possible_values.add(params.buffer_slots)
        compare_buffer_slots_key = str(params.threads) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_buffer_slots_key not in compare_buffer_slots:
            compare_buffer_slots[compare_buffer_slots_key] = {
                "params": params,
                "plots": [{
                    "title": "Bandwidth compare buffer slots",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare iterations
        iterations_possible_values.add(params.iterations)
        compare_iterations_key = str(params.threads) + "-" + str(params.buffer_slots) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_iterations_key not in compare_iterations:
            compare_iterations[compare_iterations_key] = {
                "params": params,
                "plots": [{
                    "title": "Bandwidth compare iterations",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare write modes
        write_modes_possible_values.add(params.write_mode)
        compare_write_modes_key = str(params.threads) + "-" + str(params.buffer_slots) + \
            "-" + str(params.iterations) + "-" + str(params.local_memory_type) + \
            "-" + str(params.remote_memory_type)
        if compare_write_modes_key not in compare_write_modes:
            compare_write_modes[compare_write_modes_key] = {
                "params": params,
                "plots": [{
                    "title": "Bandwidth compare write modes",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        for column in columns:
            # Process medians and averages
            if str(column["name"]).lower().startswith("med") or \
                    str(column["name"]).lower().startswith("av"):
                compare_memory_types[compare_memory_types_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + mem_type + ")",
                        "values": column["values"]
                    }
                )
                compare_threads[compare_threads_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.threads) + "x Thr)",
                        "values": column["values"]
                    }
                )
                compare_buffer_slots[compare_buffer_slots_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.buffer_slots) + "x BufSlots)",
                        "values": column["values"]
                    }
                )
                compare_iterations[compare_iterations_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.iterations) + "x Itrs)",
                        "values": column["values"]
                    }
                )
                compare_write_modes[compare_write_modes_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (wm: " + str(params.write_mode) + ")",
                        "values": column["values"]
                    }
                )

    for entry in compare_memory_types.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_memory_types = transform_and_sort_plots(compare_memory_types.values())  # TRANSFORMS !
    memory_types_possible_values = list(memory_types_possible_values)
    memory_types_possible_values.sort(reverse=True)

    for entry in compare_threads.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_threads = transform_and_sort_plots(compare_threads.values())  # TRANSFORMS !
    threads_possible_values = list(threads_possible_values)
    threads_possible_values.sort(reverse=True)

    for entry in compare_buffer_slots.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_buffer_slots = transform_and_sort_plots(compare_buffer_slots.values())  # TRANSFORMS !
    buffer_slots_possible_values = list(buffer_slots_possible_values)
    buffer_slots_possible_values.sort(reverse=True)

    for entry in compare_iterations.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_iterations = transform_and_sort_plots(compare_iterations.values())  # TRANSFORMS !
    iterations_possible_values = list(iterations_possible_values)
    iterations_possible_values.sort(reverse=True)

    for entry in compare_write_modes.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_write_modes = transform_and_sort_plots(compare_write_modes.values())  # TRANSFORMS !
    write_modes_possible_values = list(write_modes_possible_values)
    write_modes_possible_values.sort(reverse=True)

    # Plot raw values
    print("Plotting Bandwidth results RAW ...")
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        title = "Bandwidth (" + params.local_memory_type + "→" + params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots) + \
                   "  itrs=" + str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_bandwidth_raw_line_style(col_label))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "BANDWIDTH-" + params.to_file_str() + "-Raw" + "." +
                        output_format)
        plt.close(fig)

    # Plot values to compare memory types
    print("Plotting Bandwidth results to compare memory types ...")
    for plot in compare_memory_types:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"]
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots) + \
                   "  itrs=" + str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, memory_types_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "BANDWIDTH-" + params.to_file_str() + "-MemoryTypes" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare thread counts
    print("Plotting Bandwidth results to compare thread counts ...")
    for plot in compare_threads:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "bufslots=" + str(params.buffer_slots) + "  itrs=" + \
                   str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, threads_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "BANDWIDTH-" + params.to_file_str() + "-ThreadCounts" +
                        "." + output_format)
        plt.close(fig)

    # Plot values to compare buffer slots
    print("Plotting Bandwidth results to compare buffer slots ...")
    for plot in compare_buffer_slots:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  itrs=" + \
                   str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, buffer_slots_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "BANDWIDTH-" + params.to_file_str() + "-BufferSlots" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare iterations
    print("Plotting Bandwidth results to compare iterations ...")
    for plot in compare_iterations:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + \
                   str(params.buffer_slots) + "  wm=" + str(params.write_mode)
        y_label = "Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, iterations_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "BANDWIDTH-" + params.to_file_str() + "-Iterations" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare write modes
    print("Plotting Bandwidth results to compare write modes ...")
    for plot in compare_write_modes:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + \
                   str(params.buffer_slots) + "  itrs=" + str(params.iterations)
        y_label = "Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, write_modes_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "BANDWIDTH-" + params.to_file_str() + "-WriteModes" +
                "." + output_format)
        plt.close(fig)


def plot_latency(test_params: [TestParameters], test_columns: [{}], output_file_name,
                 output_format):
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    memory_types_possible_values = set()
    compare_memory_types = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    threads_possible_values = set()
    compare_threads = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    buffer_slots_possible_values = set()
    compare_buffer_slots = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    iterations_possible_values = set()
    compare_iterations = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    write_modes_possible_values = set()
    compare_write_modes = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    for entry in range(entry_count):
        params = test_params[entry]
        columns = sort_columns(test_columns[entry])  # no column filtering needed
        test_columns[entry] = columns

        # initialize compare memory types
        mem_type = params.local_memory_type + "→" + params.remote_memory_type
        memory_types_possible_values.add(mem_type.lower())
        compare_memory_types_key = str(params.threads) + "-" + str(params.buffer_slots) + "-" + \
                                   str(params.iterations) + "-" + str(params.write_mode)
        if compare_memory_types_key not in compare_memory_types:
            compare_memory_types[compare_memory_types_key] = {
                "params": params,
                "plots": [{
                    "title": "Latency compare memory types",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare thread counts
        threads_possible_values.add(params.threads)
        compare_threads_key = str(params.buffer_slots) + "-" + str(params.iterations) + \
                              "-" + str(params.local_memory_type) + "-" + str(
            params.remote_memory_type) + \
                              "-" + str(params.write_mode)
        if compare_threads_key not in compare_threads:
            compare_threads[compare_threads_key] = {
                "params": params,
                "plots": [{
                    "title": "Latency compare thread counts",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare buffer slots
        buffer_slots_possible_values.add(params.buffer_slots)
        compare_buffer_slots_key = str(params.threads) + "-" + str(params.iterations) + \
                                   "-" + str(params.local_memory_type) + "-" + str(
            params.remote_memory_type) + \
                                   "-" + str(params.write_mode)
        if compare_buffer_slots_key not in compare_buffer_slots:
            compare_buffer_slots[compare_buffer_slots_key] = {
                "params": params,
                "plots": [{
                    "title": "Latency compare buffer slots",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare iterations
        iterations_possible_values.add(params.iterations)
        compare_iterations_key = str(params.threads) + "-" + str(params.buffer_slots) + \
                                 "-" + str(params.local_memory_type) + "-" + str(
            params.remote_memory_type) + \
                                 "-" + str(params.write_mode)
        if compare_iterations_key not in compare_iterations:
            compare_iterations[compare_iterations_key] = {
                "params": params,
                "plots": [{
                    "title": "Latency compare iterations",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare write modes
        write_modes_possible_values.add(params.write_mode)
        compare_write_modes_key = str(params.threads) + "-" + str(params.buffer_slots) + \
                                  "-" + str(params.iterations) + "-" + str(
            params.local_memory_type) + "-" + \
                                  str(params.remote_memory_type)
        if compare_write_modes_key not in compare_write_modes:
            compare_write_modes[compare_write_modes_key] = {
                "params": params,
                "plots": [{
                    "title": "Latency compare write modes",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        for column in columns:
            # Process medians and averages
            if str(column["name"]).lower().startswith("med") or \
                    str(column["name"]).lower().startswith("av"):
                compare_memory_types[compare_memory_types_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + mem_type + ")",
                        "values": column["values"]
                    }
                )
                compare_threads[compare_threads_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.threads) + "x Thr)",
                        "values": column["values"]
                    }
                )
                compare_buffer_slots[compare_buffer_slots_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.buffer_slots) + "x BufSlots)",
                        "values": column["values"]
                    }
                )
                compare_iterations[compare_iterations_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.iterations) + "x Itrs)",
                        "values": column["values"]
                    }
                )
                compare_write_modes[compare_write_modes_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (wm: " + str(params.write_mode) + ")",
                        "values": column["values"]
                    }
                )

    for entry in compare_memory_types.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_memory_types = transform_and_sort_plots(compare_memory_types.values())  # TRANSFORMS !
    memory_types_possible_values = list(memory_types_possible_values)
    memory_types_possible_values.sort(reverse=True)

    for entry in compare_threads.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_threads = transform_and_sort_plots(compare_threads.values())  # TRANSFORMS !
    threads_possible_values = list(threads_possible_values)
    threads_possible_values.sort(reverse=True)

    for entry in compare_buffer_slots.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_buffer_slots = transform_and_sort_plots(compare_buffer_slots.values())  # TRANSFORMS !
    buffer_slots_possible_values = list(buffer_slots_possible_values)
    buffer_slots_possible_values.sort(reverse=True)

    for entry in compare_iterations.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_iterations = transform_and_sort_plots(compare_iterations.values())  # TRANSFORMS !
    iterations_possible_values = list(iterations_possible_values)
    iterations_possible_values.sort(reverse=True)

    for entry in compare_write_modes.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_write_modes = transform_and_sort_plots(compare_write_modes.values())  # TRANSFORMS !
    write_modes_possible_values = list(write_modes_possible_values)
    write_modes_possible_values.sort(reverse=True)

    # Plot raw values
    print("Plotting Latency results RAW ...")
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        title = "Latency (" + params.local_memory_type + "→" + params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots) + \
                   "  itrs=" + str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_latency_raw_line_style(col_label))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "LATENCY-" + params.to_file_str() + "-Raw" + "." +
                        output_format)
        plt.close(fig)

    # Plot values to compare memory types
    print("Plotting Latency results to compare memory types ...")
    for plot in compare_memory_types:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"]
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots) + \
                   "  itrs=" + str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, memory_types_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "LATENCY-" + params.to_file_str() + "-MemoryTypes" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare thread counts
    print("Plotting Latency results to compare thread counts ...")
    for plot in compare_threads:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "bufslots=" + str(params.buffer_slots) + "  itrs=" + \
                   str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, threads_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "LATENCY-" + params.to_file_str() + "-ThreadCounts" +
                        "." + output_format)
        plt.close(fig)

    # Plot values to compare buffer slots
    print("Plotting Latency results to compare buffer slots ...")
    for plot in compare_buffer_slots:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
                params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  itrs=" + \
                   str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, buffer_slots_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "LATENCY-" + params.to_file_str() + "-BufferSlots" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare iterations
    print("Plotting Latency results to compare iterations ...")
    for plot in compare_iterations:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + \
                   str(params.buffer_slots) + "  wm=" + str(params.write_mode)
        y_label = "Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, iterations_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "LATENCY-" + params.to_file_str() + "-Iterations" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare write modes
    print("Plotting Latency results to compare write modes ...")
    for plot in compare_write_modes:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + \
                   str(params.buffer_slots) + "  itrs=" + str(params.iterations)
        y_label = "Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, write_modes_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "LATENCY-" + params.to_file_str() + "-WriteModes" +
                "." + output_format)
        plt.close(fig)


def plot_operations_count(test_params: [TestParameters], test_columns: [{}], output_file_name,
                          output_format):
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    memory_types_possible_values = set()
    compare_memory_types = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    threads_possible_values = set()
    compare_threads = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    buffer_slots_possible_values = set()
    compare_buffer_slots = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    iterations_possible_values = set()
    compare_iterations = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    write_modes_possible_values = set()
    compare_write_modes = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    for entry in range(entry_count):
        params = test_params[entry]
        columns = sort_columns(filter_columns(test_columns[entry], ("/s", "pack")))
        test_columns[entry] = columns

        # initialize compare memory types
        mem_type = params.local_memory_type + "→" + params.remote_memory_type
        memory_types_possible_values.add(mem_type.lower())
        compare_memory_types_key = str(params.threads) + "-" + str(params.buffer_slots) + "-" + \
                                   str(params.iterations) + "-" + str(params.write_mode)
        if compare_memory_types_key not in compare_memory_types:
            compare_memory_types[compare_memory_types_key] = {
                "params": params,
                "plots": [{
                    "title": "Operations/sec compare memory types",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare thread counts
        threads_possible_values.add(params.threads)
        compare_threads_key = str(params.buffer_slots) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_threads_key not in compare_threads:
            compare_threads[compare_threads_key] = {
                "params": params,
                "plots": [{
                    "title": "Operations/sec compare thread counts",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare buffer slots
        buffer_slots_possible_values.add(params.buffer_slots)
        compare_buffer_slots_key = str(params.threads) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_buffer_slots_key not in compare_buffer_slots:
            compare_buffer_slots[compare_buffer_slots_key] = {
                "params": params,
                "plots": [{
                    "title": "Operations/sec compare buffer slots",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare iterations
        iterations_possible_values.add(params.iterations)
        compare_iterations_key = str(params.threads) + "-" + str(params.buffer_slots) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_iterations_key not in compare_iterations:
            compare_iterations[compare_iterations_key] = {
                "params": params,
                "plots": [{
                    "title": "Operations/sec compare iterations",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare write modes
        write_modes_possible_values.add(params.write_mode)
        compare_write_modes_key = str(params.threads) + "-" + str(params.buffer_slots) + \
            "-" + str(params.iterations) + "-" + str(params.local_memory_type) + \
            "-" + str(params.remote_memory_type)
        if compare_write_modes_key not in compare_write_modes:
            compare_write_modes[compare_write_modes_key] = {
                "params": params,
                "plots": [{
                    "title": "Operations/sec compare write modes",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        for column in columns:
            # Process medians and averages
            if str(column["name"]).lower().startswith("med") or \
                    str(column["name"]).lower().startswith("av"):
                compare_memory_types[compare_memory_types_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + mem_type + ")",
                        "values": column["values"]
                    }
                )
                compare_threads[compare_threads_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.threads) + "x Thr)",
                        "values": column["values"]
                    }
                )
                compare_buffer_slots[compare_buffer_slots_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.buffer_slots) + "x BufSlots)",
                        "values": column["values"]
                    }
                )
                compare_iterations[compare_iterations_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.iterations) + "x Itrs)",
                        "values": column["values"]
                    }
                )
                compare_write_modes[compare_write_modes_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (wm: " + str(params.write_mode) + ")",
                        "values": column["values"]
                    }
                )

    for entry in compare_memory_types.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_memory_types = transform_and_sort_plots(compare_memory_types.values())  # TRANSFORMS !
    memory_types_possible_values = list(memory_types_possible_values)
    memory_types_possible_values.sort(reverse=True)

    for entry in compare_threads.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_threads = transform_and_sort_plots(compare_threads.values())  # TRANSFORMS !
    threads_possible_values = list(threads_possible_values)
    threads_possible_values.sort(reverse=True)

    for entry in compare_buffer_slots.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_buffer_slots = transform_and_sort_plots(compare_buffer_slots.values())  # TRANSFORMS !
    buffer_slots_possible_values = list(buffer_slots_possible_values)
    buffer_slots_possible_values.sort(reverse=True)

    for entry in compare_iterations.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_iterations = transform_and_sort_plots(compare_iterations.values())  # TRANSFORMS !
    iterations_possible_values = list(iterations_possible_values)
    iterations_possible_values.sort(reverse=True)

    for entry in compare_write_modes.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_write_read_send(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_write_modes = transform_and_sort_plots(compare_write_modes.values())  # TRANSFORMS !
    write_modes_possible_values = list(write_modes_possible_values)
    write_modes_possible_values.sort(reverse=True)

    # Plot raw values
    print("Plotting Operations/sec results RAW ...")
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        title = "Operations/sec (" + params.local_memory_type + "→" + params.remote_memory_type + \
                ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots) + \
                   "  itrs=" + str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_operations_count_raw_line_style(col_label))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "OPERATIONS_COUNT-" + params.to_file_str() + "-Raw" +
                        "." + output_format)
        plt.close(fig)

    # Plot values to compare memory types
    print("Plotting Operations/sec results to compare memory types ...")
    for plot in compare_memory_types:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"]
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots) + \
                   "  itrs=" + str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, memory_types_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "OPERATIONS_COUNT-" + params.to_file_str() + "-MemoryTypes" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare thread counts
    print("Plotting Operations/sec results to compare thread counts ...")
    for plot in compare_threads:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "bufslots=" + str(params.buffer_slots) + "  itrs=" + \
                   str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, threads_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "OPERATIONS_COUNT-" + params.to_file_str() +
                        "-ThreadCounts" + "." + output_format)
        plt.close(fig)

    # Plot values to compare buffer slots
    print("Plotting Operations/sec results to compare buffer slots ...")
    for plot in compare_buffer_slots:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  itrs=" + \
                   str(params.iterations) + "  wm=" + str(params.write_mode)
        y_label = "Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, buffer_slots_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "OPERATIONS_COUNT-" + params.to_file_str() + "-BufferSlots" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare iterations
    print("Plotting Operations/sec results to compare iterations ...")
    for plot in compare_iterations:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + \
                   str(params.buffer_slots) + "  wm=" + str(params.write_mode)
        y_label = "Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, iterations_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "OPERATIONS_COUNT-" + params.to_file_str() + "-Iterations" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare write modes
    print("Plotting Operations/sec results to compare write modes ...")
    for plot in compare_write_modes:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + \
                   str(params.buffer_slots) + "  itrs=" + str(params.iterations)
        y_label = "Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, write_modes_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "OPERATIONS_COUNT-" + params.to_file_str() + "-WriteModes" +
                "." + output_format)
        plt.close(fig)


def plot_atomics_bandwidth(test_params: [TestParameters], test_columns: [{}], output_file_name,
                           output_format):
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    memory_types_possible_values = set()
    compare_memory_types = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    threads_possible_values = set()
    compare_threads = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    buffer_slots_possible_values = set()
    compare_buffer_slots = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    for entry in range(entry_count):
        params = test_params[entry]
        columns = sort_columns(filter_columns(test_columns[entry], ("/s", "iter")))
        test_columns[entry] = columns

        # initialize compare memory types
        mem_type = params.local_memory_type + "→" + params.remote_memory_type
        memory_types_possible_values.add(mem_type.lower())
        compare_memory_types_key = str(params.threads) + "-" + str(params.buffer_slots) + "-" + \
            str(params.iterations) + "-" + str(params.write_mode)
        if compare_memory_types_key not in compare_memory_types:
            compare_memory_types[compare_memory_types_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Bandwidth compare memory types",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare thread counts
        threads_possible_values.add(params.threads)
        compare_threads_key = str(params.buffer_slots) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_threads_key not in compare_threads:
            compare_threads[compare_threads_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Bandwidth compare thread counts",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare buffer slots
        buffer_slots_possible_values.add(params.buffer_slots)
        compare_buffer_slots_key = str(params.threads) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_buffer_slots_key not in compare_buffer_slots:
            compare_buffer_slots[compare_buffer_slots_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Bandwidth compare buffer slots",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        for column in columns:
            # Process medians and averages
            if str(column["name"]).lower().startswith("med") or \
                    str(column["name"]).lower().startswith("av"):
                compare_memory_types[compare_memory_types_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + mem_type + ")",
                        "values": column["values"]
                    }
                )
                compare_threads[compare_threads_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.threads) + "x Thr)",
                        "values": column["values"]
                    }
                )
                compare_buffer_slots[compare_buffer_slots_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.buffer_slots) + "x BufSlots)",
                        "values": column["values"]
                    }
                )

    for entry in compare_memory_types.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_memory_types = transform_and_sort_plots(compare_memory_types.values())  # TRANSFORMS !
    memory_types_possible_values = list(memory_types_possible_values)
    memory_types_possible_values.sort(reverse=True)

    for entry in compare_threads.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_threads = transform_and_sort_plots(compare_threads.values())  # TRANSFORMS !
    threads_possible_values = list(threads_possible_values)
    threads_possible_values.sort(reverse=True)

    for entry in compare_buffer_slots.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_buffer_slots = transform_and_sort_plots(compare_buffer_slots.values())  # TRANSFORMS !
    buffer_slots_possible_values = list(buffer_slots_possible_values)
    buffer_slots_possible_values.sort(reverse=True)

    # Plot raw values
    print("Plotting Atomics Bandwidth results RAW ...")
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        title = "Atomics Bandwidth (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_bandwidth_raw_line_style(col_label))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "ATOMICS_BANDWIDTH-" + params.to_file_str() + "-Raw" +
                        "." + output_format)
        plt.close(fig)

    # Plot values to compare memory types
    print("Plotting Atomics Bandwidth results to compare memory types ...")
    for plot in compare_memory_types:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"]
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, memory_types_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "ATOMICS_BANDWIDTH-" + params.to_file_str() + "-MemoryTypes" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare thread counts
    print("Plotting Atomics Bandwidth results to compare thread counts ...")
    for plot in compare_threads:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, threads_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "ATOMICS_BANDWIDTH-" + params.to_file_str() +
                        "-ThreadCounts" + "." + output_format)
        plt.close(fig)

    # Plot values to compare buffer slots
    print("Plotting Atomics Bandwidth results to compare buffer slots ...")
    for plot in compare_buffer_slots:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads)
        y_label = "Atomics Bandwidth"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Bandwidth " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, buffer_slots_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "ATOMICS_BANDWIDTH-" + params.to_file_str() + "-BufferSlots" +
                "." + output_format)
        plt.close(fig)


def plot_atomics_latency(test_params: [TestParameters], test_columns: [{}], output_file_name,
                         output_format):
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    memory_types_possible_values = set()
    compare_memory_types = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    threads_possible_values = set()
    compare_threads = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    buffer_slots_possible_values = set()
    compare_buffer_slots = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    for entry in range(entry_count):
        params = test_params[entry]
        columns = sort_columns(test_columns[entry])  # no filtering needed
        test_columns[entry] = columns

        # initialize compare memory types
        mem_type = params.local_memory_type + "→" + params.remote_memory_type
        memory_types_possible_values.add(mem_type.lower())
        compare_memory_types_key = str(params.threads) + "-" + str(params.buffer_slots) + "-" + \
            str(params.iterations) + "-" + str(params.write_mode)
        if compare_memory_types_key not in compare_memory_types:
            compare_memory_types[compare_memory_types_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Latency compare memory types",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare thread counts
        threads_possible_values.add(params.threads)
        compare_threads_key = str(params.buffer_slots) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_threads_key not in compare_threads:
            compare_threads[compare_threads_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Latency compare thread counts",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare buffer slots
        buffer_slots_possible_values.add(params.buffer_slots)
        compare_buffer_slots_key = str(params.threads) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_buffer_slots_key not in compare_buffer_slots:
            compare_buffer_slots[compare_buffer_slots_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Latency compare buffer slots",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        for column in columns:
            # Process medians and averages
            if str(column["name"]).lower().startswith("med") or \
                    str(column["name"]).lower().startswith("av"):
                compare_memory_types[compare_memory_types_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + mem_type + ")",
                        "values": column["values"]
                    }
                )
                compare_threads[compare_threads_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.threads) + "x Thr)",
                        "values": column["values"]
                    }
                )
                compare_buffer_slots[compare_buffer_slots_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.buffer_slots) + "x BufSlots)",
                        "values": column["values"]
                    }
                )

    for entry in compare_memory_types.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_memory_types = transform_and_sort_plots(compare_memory_types.values())  # TRANSFORMS !
    memory_types_possible_values = list(memory_types_possible_values)
    memory_types_possible_values.sort(reverse=True)

    for entry in compare_threads.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_threads = transform_and_sort_plots(compare_threads.values())  # TRANSFORMS !
    threads_possible_values = list(threads_possible_values)
    threads_possible_values.sort(reverse=True)

    for entry in compare_buffer_slots.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_buffer_slots = transform_and_sort_plots(compare_buffer_slots.values())  # TRANSFORMS !
    buffer_slots_possible_values = list(buffer_slots_possible_values)
    buffer_slots_possible_values.sort(reverse=True)

    # Plot raw values
    print("Plotting Atomics Latency results RAW ...")
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        title = "Atomics Latency (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, []))  # use compare style for raw data
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "ATOMICS_LATENCY-" + params.to_file_str() + "-Raw" +
                        "." + output_format)
        plt.close(fig)

    # Plot values to compare memory types
    print("Plotting Atomics Latency results to compare memory types ...")
    for plot in compare_memory_types:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"]
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, memory_types_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "ATOMICS_LATENCY-" + params.to_file_str() + "-MemoryTypes" +
                "." + output_format)
        plt.close(fig)

    # Plot values to compare thread counts
    print("Plotting Atomics Latency results to compare thread counts ...")
    for plot in compare_threads:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, threads_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "ATOMICS_LATENCY-" + params.to_file_str() +
                        "-ThreadCounts" + "." + output_format)
        plt.close(fig)

    # Plot values to compare buffer slots
    print("Plotting Atomics Latency results to compare buffer slots ...")
    for plot in compare_buffer_slots:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads)
        y_label = "Atomics Latency"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Latency " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, buffer_slots_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "ATOMICS_LATENCY-" + params.to_file_str() + "-BufferSlots" +
                "." + output_format)
        plt.close(fig)


def plot_atomics_operations_count(test_params: [TestParameters], test_columns: [{}],
                                  output_file_name, output_format):
    entry_count = len(test_params)

    # Group columns and find minimum and maximum y value
    memory_types_possible_values = set()
    compare_memory_types = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    threads_possible_values = set()
    compare_threads = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    buffer_slots_possible_values = set()
    compare_buffer_slots = {}  # {key: {params:[], plots:[{title:str, columns:[{}] }] }}  # (tmp)
    for entry in range(entry_count):
        params = test_params[entry]
        columns = sort_columns(filter_columns(test_columns[entry], ("/s", "iter")))
        test_columns[entry] = columns

        # initialize compare memory types
        mem_type = params.local_memory_type + "→" + params.remote_memory_type
        memory_types_possible_values.add(mem_type.lower())
        compare_memory_types_key = str(params.threads) + "-" + str(params.buffer_slots) + "-" + \
            str(params.iterations) + "-" + str(params.write_mode)
        if compare_memory_types_key not in compare_memory_types:
            compare_memory_types[compare_memory_types_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Operations/sec compare memory types",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare thread counts
        threads_possible_values.add(params.threads)
        compare_threads_key = str(params.buffer_slots) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_threads_key not in compare_threads:
            compare_threads[compare_threads_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Operations/sec compare thread counts",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        # initialize compare buffer slots
        buffer_slots_possible_values.add(params.buffer_slots)
        compare_buffer_slots_key = str(params.threads) + "-" + str(params.iterations) + \
            "-" + str(params.local_memory_type) + "-" + str(params.remote_memory_type) + \
            "-" + str(params.write_mode)
        if compare_buffer_slots_key not in compare_buffer_slots:
            compare_buffer_slots[compare_buffer_slots_key] = {
                "params": params,
                "plots": [{
                    "title": "Atomics Operations/sec compare buffer slots",
                    "columns": [columns[0], ]  # x-values, y-values will be appended
                }, ]  # possibly split up into multiple plots if too much columns
            }

        for column in columns:
            # Process medians and averages
            if str(column["name"]).lower().startswith("med") or \
                    str(column["name"]).lower().startswith("av"):
                compare_memory_types[compare_memory_types_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + mem_type + ")",
                        "values": column["values"]
                    }
                )
                compare_threads[compare_threads_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.threads) + "x Thr)",
                        "values": column["values"]
                    }
                )
                compare_buffer_slots[compare_buffer_slots_key]["plots"][0]["columns"].append(
                    {
                        "name": column["name"] + " (" + str(params.buffer_slots) + "x BufSlots)",
                        "values": column["values"]
                    }
                )

    for entry in compare_memory_types.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_memory_types = transform_and_sort_plots(compare_memory_types.values())  # TRANSFORMS !
    memory_types_possible_values = list(memory_types_possible_values)
    memory_types_possible_values.sort(reverse=True)

    for entry in compare_threads.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_threads = transform_and_sort_plots(compare_threads.values())  # TRANSFORMS !
    threads_possible_values = list(threads_possible_values)
    threads_possible_values.sort(reverse=True)

    for entry in compare_buffer_slots.values():
        if len(entry["plots"][0]["columns"]) > MAX_LINES_PER_PLOT_BEFORE_SPLITTING + 1:
            entry["plots"] = split_plot_by_fetch_compare(  # split columns into multiple plots
                entry["plots"][0]
            )
    compare_buffer_slots = transform_and_sort_plots(compare_buffer_slots.values())  # TRANSFORMS !
    buffer_slots_possible_values = list(buffer_slots_possible_values)
    buffer_slots_possible_values.sort(reverse=True)

    # Plot raw values
    print("Plotting Atomics Operations/sec results RAW ...")
    for entry in range(entry_count):
        params = test_params[entry]
        columns = test_columns[entry]
        title = "Atomics Operations/sec (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_operations_count_raw_line_style(col_label))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "ATOMICS_OPERATIONS_COUNT-" + params.to_file_str() +
                        "-Raw" + "." + output_format)
        plt.close(fig)

    # Plot values to compare memory types
    print("Plotting Atomics Operations/sec results to compare memory types ...")
    for plot in compare_memory_types:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"]
        subtitle = "thrs=" + str(params.threads) + "  bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, memory_types_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "ATOMICS_OPERATIONS_COUNT-" + params.to_file_str() +
                "-MemoryTypes" + "." + output_format)
        plt.close(fig)

    # Plot values to compare thread counts
    print("Plotting Atomics Operations/sec results to compare thread counts ...")
    for plot in compare_threads:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "bufslots=" + str(params.buffer_slots)
        y_label = "Atomics Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, threads_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(output_file_name + "ATOMICS_OPERATIONS_COUNT-" + params.to_file_str() +
                        "-ThreadCounts" + "." + output_format)
        plt.close(fig)

    # Plot values to compare buffer slots
    print("Plotting Atomics Operations/sec results to compare buffer slots ...")
    for plot in compare_buffer_slots:
        columns = plot["columns"]
        if len(columns) <= MIN_LINES_PER_COMPARE_PLOT:
            continue  # check if multiple avg & medians such that comparing makes sense
        params = plot["params"]
        title = plot["title"] + " (" + params.local_memory_type + "→" + \
            params.remote_memory_type + ")"
        subtitle = "thrs=" + str(params.threads)
        y_label = "Atomics Operations/sec"
        y_label_update = True
        x_values = columns[0]["values"]
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])
        x_values = np.array(x_values)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        first = True
        for column in columns:
            if first:  # skip x-values
                first = False
                continue
            col_label = column["name"]
            y_values = np.array(column["values"])
            tmp = remove_unit_from_label(col_label)
            col_label = tmp[0]
            if tmp[1] and y_label_update:
                y_label = "Atomics Operations/sec " + tmp[1]
                y_label_update = False
            ax.plot(x_values, y_values, label=col_label, antialiased=False,
                    **get_compare_line_style(col_label, buffer_slots_possible_values))
        ax.set(xlabel="Packet Size [Bytes]", ylabel=y_label, title=subtitle)
        ax.legend(fontsize="xx-small")
        if isinstance(output_file_name, PdfPages):
            output_file_name.savefig(figure=fig)
        else:
            plt.savefig(
                output_file_name + "ATOMICS_OPERATIONS_COUNT-" + params.to_file_str() +
                "-BufferSlots" + "." + output_format)
        plt.close(fig)


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
        has_values = False
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
            row = [str(x).strip() for x in row if len(str(x).strip()) > 0]
            if len(row) == 0:  # prepare for next test block
                prev_test_name = test_name.lower()
                test_name = ""
                continue
            if len(test_name) == 0:
                test_name = str(row[0]).strip()
                if test_name.lower() != prev_test_name:  # reset because new test name
                    if has_values:
                        plot_test_values(prev_test_name, test_params, test_columns,
                                         output_file_name, output_format)
                    has_values = False
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
            else:
                has_values = True
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
        if has_values:
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
