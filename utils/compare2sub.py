# {"id": 1, "language": "ar", "origin_query": "%منتخب ايطالي", "category_path": "apparel accessories,hats & caps,baseball caps", "prediction": 1}

import re
import json
import sys

if len(sys.argv) != 3:
    print("Usage: python compare2sub.py <file1> <file2>")
    sys.exit(1)
file1 = sys.argv[1]
file2 = sys.argv[2]

file1_data = {}
file2_data = {}
with open(file1, 'r') as f1, open(file2, 'r') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    for line in lines1:
        json_line = json.loads(line)
        file1_data[json_line['id']] = json_line['prediction']
    for line in lines2:
        json_line = json.loads(line)
        file2_data[json_line['id']] = json_line['prediction']

assert file1_data.keys() == file2_data.keys(), "Mismatch in IDs between files"

# some diff stats
same_count = 0
diff_count = 0

for key in file1_data.keys():
    if file1_data[key] == file2_data[key]:
        same_count += 1
    else:
        diff_count += 1
print(f"Same predictions: {same_count}/{len(file1_data)}")
print(f"Different predictions: {diff_count}/{len(file1_data)} ({(diff_count/len(file1_data))*100:.2f}%)")