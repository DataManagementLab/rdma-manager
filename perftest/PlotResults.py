import sys
from os import walk
import os
import matplotlib.pyplot as matplot
import numpy as np
import csv

# [ HOW TO USE ]
# -----------------------------------------------
#
# python PlotResults.py <csvFile> <outputFormat>
#   csvFile:        File name or . to automatically find csvFile (optional)
#   outputFormat:   pdf
#
# -----------------------------------------------


CSV_DELIMITER = ','
CSV_LINEBREAK = '\n'
CSV_COMMA = '.'


def findCSVFile():
    for dirpath, dirnames, filenames in walk("../"):
        for filename in filenames:
            if filename.endswith(".csv"):
                return dirpath + filename
    return None


def plotBandwidth(test_params, test_columns):
    print("Plotting Bandwidth results ...")
    print(test_params)
    print(test_columns)


def plotLatency(test_params, test_columns):
    print("Plotting Latency results ...")


def plotOperationsCount(test_params, test_columns):
    print("Plotting OperationsCount results ...")


def plotAtomicsBandwidth(test_params, test_columns):
    print("Plotting AtomicsBandwidth results ...")


def plotAtomicsLatency(test_params, test_columns):
    print("Plotting AtomicsLatency results ...")


def plotAtomicsOperationsCount(test_params, test_columns):
    print("Plotting AtomicsOperationsCount results ...")


def plotTestValues(test_name: str, test_params: [[str]], test_columns: {}):
    test = str(test_name).strip().replace(" ", "").lower()
    if test == "bandwidth":
        plotBandwidth(test_params, test_columns)
    elif test == "latency":
        plotLatency(test_params, test_columns)
    elif test == "operationscount":
        plotOperationsCount(test_params, test_columns)
    elif test == "atomicsbandwidth":
        plotAtomicsBandwidth(test_params, test_columns)
    elif test == "atomicslatency":
        plotAtomicsLatency(test_params, test_columns)
    elif test == "atomicsoperationscount":
        plotAtomicsOperationsCount(test_params, test_columns)
    else:
        sys.stderr.write("Cannot print unknown test '%s'\n" % test_name)


def plotCSVFile(csv_file_name):
    if csv_file_name is None:
        raise FileNotFoundError("No CSV file found and also not provided as line argument")
    if not os.path.exists(csv_file_name):
        raise FileNotFoundError("File '%s' does not exist" % csv_file_name)
    if not os.path.isfile(csv_file_name):
        raise FileNotFoundError("File '%s' is not a file" % csv_file_name)

    with open(csv_file_name, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_LINEBREAK)
        test_index = 0
        parse_labels = True
        prev_test_name = ""
        test_name = ""
        test_params = []  # [test_index: [params]]
        test_columns = {}  # {test_index: [{ name=str, values=[values] }]}
        for row in reader:
            print(row)
            if len(row) == 0:  # prepare for next test block
                prev_test_name = test_name.lower()
                test_name = ""
                continue
            if len(test_name) == 0:
                test_name = str(row[0])
                if test_name.lower() != prev_test_name:  # reset because new test name
                    if test_index > 0:
                        plotTestValues(test_name, test_params, test_columns)
                    test_index = 0
                    test_params = []
                    test_columns = {}
                else:
                    test_index = test_index + 1
                test_params.append(row[1:], )
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
            plotTestValues(test_name, test_params, test_columns)


csvFileName = None
if len(sys.argv) <= 1:
    csvFileName = findCSVFile()
else:
    csvFileName = sys.argv[1]
    if csvFileName == ".":
        csvFileName = findCSVFile()
print("Plotting CSV file '%s' ..." % csvFileName)
plotCSVFile(csvFileName)
