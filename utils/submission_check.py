"""
This script checks the submission file for:
1. All IDs are present.
2. Language matches the original query.
3. Origin query matches the original query.
"""
import sys
if len(sys.argv) != 3:
    print("Usage: python submission_check.py <QC|QI> <submission_file>")
    sys.exit(1)
task = sys.argv[1]  # QC or QI
submission = sys.argv[2]  # submission file path
test_file = f'data/raw/dev_{task}.txt'
import json
QC_id_check = set(list(range(1,100000 + 1)))
QI_id_check = set(list(range(1,65000 + 1)))

id_factory = {
    'QC': QC_id_check,
    'QI': QI_id_check
}

id_check = id_factory[task]

queries = {}
# sanity check, match id, match language, match origin query
with open(test_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        json_line = json.loads(line)
        queries[json_line['id']] = json_line

with open(submission, 'r') as f:
    lines = f.readlines()
    # {"id": 2, "language": "ar", "origin_query": "%منتخب ايطالي", "category_path": "home & garden, home decor, flags,  banners & accessories, flags", "prediction": 1}
    for line in lines:
        json_line = json.loads(line)
        id_check.discard(json_line['id'])

        # sanity check
        if json_line['id'] not in queries:
            print(f"WARNING: ID {json_line['id']} not found in queries")
            continue
        if json_line['language'] != queries[json_line['id']]['language']:
            print(f"WARNING: Language mismatch for ID {json_line['id']}")
        if json_line['origin_query'] != queries[json_line['id']]['origin_query']:
            print(f"WARNING: Origin query mismatch for ID {json_line['id']}")
    else:
        print("All checks passed.")
if len(id_check) == 0:
    print("All IDs are present.")
else:
    print('Missing ids')
    print(id_check)