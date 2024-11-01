import re
import numpy as np


"""
This script reads the dataset report that contains the descriptions of every variable
and retrieves, for every feature, a dictionary with abnormal values, such as 77 or 99,
together with their replacements.

The results are stored in a json file, and then they are manually converted to the dictionary
that can be found in the config.py file (config.ABNORMAL_FEATURE_VALUES)
"""


with open("codebook15_llcp.txt", "r") as f:
    lines = f.readlines()

pattern = r"SAS Variable Name:\s*(_?\w+)"

var_info = []
current_lines = []
for line in lines:
    new_line = line.strip()
    if new_line == "":
        if current_lines:
            var_info.append(current_lines)
            current_lines = []
        continue
    current_lines.append(new_line)

final_dict = {}
for variable_block in var_info:
    collecting = False
    variable_name = ""
    variable_values = {}
    for line in variable_block:
        match = re.findall(pattern, line)
        if match:
            variable_name = match[0]
        elif "Value Label" in line:
            collecting = True
            continue

        if collecting:
            if "Notes:" in line:
                variable_values[value] += " - " + line
            else:
                value = line.split(" ")[0]
                variable_values[value] = line[len(value) + 1 :]
                # variable_values[value] = re.split(r'\s{2,}', line[len(value)+1:])[0]
    final_dict[variable_name] = variable_values


abnormal_values = {}
texts_to_zero = ["None"]
texts_to_nan = ["Don't know", "Not Sure", "Refused", "missing", "Not asked"]
pattern = re.compile(
    "|".join([re.escape(term) for term in texts_to_nan]), re.IGNORECASE
)
for var, info in final_dict.items():
    # print(f"==={var}===")
    abnormal_values[var] = {}
    for value, line in info.items():
        # print(f"{value}: {line}")

        if not value.isnumeric():
            continue

        if any([text in line for text in texts_to_zero]):
            abnormal_values[var][float(value)] = 0
        elif re.findall(pattern, line):
            abnormal_values[var][float(value)] = np.nan

    print("**Abnormal values:**")
    for value, replacement in abnormal_values[var].items():
        print(f"{value}: {replacement}")

import json

with open("var_abnormal_values.json", "w") as f:
    json.dump(abnormal_values, f)
