import os
# Ensemble voting per ID from multiple input files
import json
import sys
from collections import Counter, defaultdict

def read_predictions(file_path):
	preds = {}
	meta = {}
	with open(file_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				data = json.loads(line)
				id_ = data["id"]
				label = data["prediction"]
				preds[id_] = label
				if id_ not in meta:
					meta[id_] = data
			except (ValueError, KeyError) as e:
				print(f"Error parsing line in {file_path}: {line}")
				continue
	return preds, meta

def main(input_files, output_file, w):
	votes = defaultdict(lambda: defaultdict(float))
	meta = {}
	for idx, file_path in enumerate(input_files):
		preds, meta_part = read_predictions(file_path)
		for id_, label in preds.items():
			votes[id_][label] += w[idx]
		for id_, data in meta_part.items():
			if id_ not in meta:
				meta[id_] = data
	if os.path.exists(output_file):
		print(f"[WARNING] Output file {output_file} already exists. Stopping.")
		return
	with open(output_file, 'w', encoding='utf-8') as out:
		for id_ in sorted(votes.keys()):
			label_weights = votes[id_]
			voted_label = max(label_weights.items(), key=lambda x: x[1])[0]
			data = meta[id_].copy()
			data["prediction"] = voted_label
			out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print("Usage: python voting.py input1.txt input2.txt ... output.txt")
		sys.exit(1)
	*input_files, output_file = sys.argv[1:]
	w = [1,1,1]
	for f in input_files:
		print(f)
	# w = [0.2303407934864819, 0.2541745702223031, 0.2562831629342899, 0.2592014733569251]
	# w = [0.3088400908442315, 0.3436235246238231, 0.3475363845319454]
	if len(w) != len(input_files):
		print(f"[WARNING] Weight vector length {len(w)} does not match number of input files {len(input_files)}. Using uniform weights.")
		w = [1] * len(input_files)
	main(input_files, output_file, w)

