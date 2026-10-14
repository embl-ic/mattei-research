# =============================================================================
# COPYRIGHT, ATTRIBUTION, AND LICENSE NOTICE
# =============================================================================
#
# Copyright (c) 2026 Linhua Tai,
# Mahamid Lab, Molecular Systems Biology Unit,
# European Molecular Biology Laboratory (EMBL), Heidelberg, Germany.
#
# SOFTWARE DEVELOPMENT:
# This script was developed by Linhua Tai in the Mahamid Lab,
# Molecular Systems Biology Unit, at the European Molecular Biology
# Laboratory (EMBL), Heidelberg, Germany, with the assistance of
# Artificial Intelligence (AI) tools for code development,
# optimization, and documentation.
#
# OPEN-SOURCE AVAILABILITY:
# This software is released as open-source software under the MIT License
# to support scientific research, reproducibility, and collaboration.
#
# Users are permitted to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of this software, subject to the terms
# of the MIT License provided below.
#
# ATTRIBUTION AND CITATION:
# When using, modifying, adapting, or redistributing this software,
# users must retain the original copyright and license notices as
# required by the MIT License.
#
# When this software contributes to scientific publications,
# presentations, or other research outputs, users are kindly requested
# to acknowledge the original developer as:
#
# "Linhua Tai, Mahamid Lab, Molecular Systems Biology Unit,
#  European Molecular Biology Laboratory (EMBL), Heidelberg."
#
# Scientific citation and acknowledgment are strongly encouraged
# to recognize the contributions of the original developer.
#
# =============================================================================
# MIT LICENSE
# =============================================================================
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# =============================================================================

import os
import csv
import shutil

# Folder paths
csv_folder = 'warp_tiltseries/selection_csv/' # Replace with the path to the CSV folder
tomostar_folder = 'tomostar/'  # Replace with the path to the TOMOSTAR folder
output_folder = 'tomostar/new' # Replace with the path to the output folder

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Traverse the CSV folder and get all file names
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
tomostar_files = [f for f in os.listdir(tomostar_folder) if f.endswith('.tomostar')]

# Create a file name mapping for matching
csv_files_dict = {os.path.splitext(f)[0]: f for f in csv_files}
tomostar_files_dict = {os.path.splitext(f)[0]: f for f in tomostar_files}

# Process each matching file pair
for filename in csv_files_dict.keys():
    if filename in tomostar_files_dict:
        csv_file_path = os.path.join(csv_folder, csv_files_dict[filename])
        tomostar_file_path = os.path.join(tomostar_folder, tomostar_files_dict[filename])
        output_file_path = os.path.join(output_folder, tomostar_files_dict[filename])

        # Read the lines to skip from the CSV file
        skip_lines = set()
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the first row
            for row in reader:
                skip_lines.add(int(row[0]))

        # Read and process the TOMOSTAR file
        with open(tomostar_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line_number, line in enumerate(infile):
                if line_number < 10:  # Keep the first 1-10 lines
                    outfile.write(line)
                elif (line_number - 10) not in skip_lines:  # From the 11th line, skip specified lines
                    outfile.write(line)

# Copy TOMOSTAR files that do not have a corresponding CSV file
for filename in tomostar_files_dict.keys():
    if filename not in csv_files_dict:
        tomostar_file_path = os.path.join(tomostar_folder, tomostar_files_dict[filename])
        output_file_path = os.path.join(output_folder, tomostar_files_dict[filename])
        shutil.copy(tomostar_file_path, output_file_path)
