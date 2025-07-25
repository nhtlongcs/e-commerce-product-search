from collections import defaultdict
import json
from pprint import pprint
filename = "train_QC"
filepath = f"../data/{filename}.txt"
paths = []
with open(filepath, 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        paths.append([x.strip() for x in item['category_path'].split(',')])  # Strip whitespace from each value

paths = paths
# Recursive defaultdict for tree construction
def tree(): return defaultdict(tree)

def insert_path(tree_root, path):
    node = tree_root
    for category in map(str.strip, path):
        node = node[category]

# Construct tree
root = tree()
for path in paths:
    insert_path(root, path)

# Convert to regular dict for printing / JSON use
def dictify(t):
    return {k: dictify(v) for k, v in t.items()}


def check_path(tree_root, path):
    node = tree_root
    parts = list(map(str.strip, path.split(',')))

    for i, part in enumerate(parts):
        if part in node:
            node = node[part]
        else:
            if i == 0:
                return "no match"
            else:
                return "partial match"
    return "complete match"

dev_filename = "dev_QC"
dev_filepath = f"../data/{dev_filename}.txt"
dev_paths = []
with open(dev_filepath, 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        dev_paths.append([x.strip() for x in item['category_path'].split(',')])  # Strip whitespace from each value
res = []
for dev_path in dev_paths:
    result = check_path(root, ', '.join(dev_path))
    res.append(result)

print(f"Results for {dev_filename}:")
print(f"Total paths: {len(dev_paths)}")
print(f"Total matches: {sum(1 for r in res if r == 'complete match')}")
print(f"Total partial matches: {sum(1 for r in res if r == 'partial match')}")
print(f"Total no matches: {sum(1 for r in res if r == 'no match')}")

# pprint(dictify(root), width=100)
with open(f"category_tree_{filename}.json", 'w') as f:
    json.dump(dictify(root), f, indent=2, ensure_ascii=False)
print(f"Category tree saved to category_tree_{filename}.json")
