import json
import os 
from nlp.translate import TranslatorWrapper
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    mt = TranslatorWrapper()

    filenames = ["test_QI", "test_QC"] #"train_QI", "train_QC", "dev_QI", "dev_QC", 
    output_dir = "data/translated"
    os.makedirs(output_dir, exist_ok=True)
    for filename in filenames:
        print(f"Processing {filename}")
        filepath = f"data/raw/{filename}.txt"
        data = []
        save_path = os.path.join(output_dir, f"translated_{filename}_full.csv")
        if os.path.exists(save_path):
            print(f"File {save_path} already exists, skipping...")
            continue
        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item) # list of dicts {{'id': 1, 'task': 'QI','language': 'en', 'origin_query': .. 'item_title': .. ,'label': '0' }
        batch_size = 2048
        for i in tqdm(range(0, len(data), batch_size)):
            chunks = data[i:i + batch_size]
            text_chunks = [x["origin_query"] for x in chunks]
            results = mt.translate(text_chunks, None, method="offline")
            languages = [x["language"] for x in chunks]

            df_chunk = pd.DataFrame(chunks)
            df_chunk["translated_query"] = results
            if i == 0:
                df = df_chunk
            else:
                df = pd.concat([df, df_chunk], ignore_index=True)

        df.to_csv(save_path, index=False)
