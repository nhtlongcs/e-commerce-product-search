import json
import os 
import rich
import pandas as pd

from nlp.lang import split_translate_merge

def translate_process_data(df_dataset, verbose=False):
   
    print("Translating queries")
    df_dataset["translated_query"] = df_dataset.progress_apply(
        lambda x: split_translate_merge(x["origin_query"], x["language"], method="offline", verbose=verbose),
        axis=1,
    )

    return df_dataset

if __name__ == "__main__":
    filenames = ["train_QI", "train_QC", "dev_QI", "dev_QC", "test_QI", "test_QC"]
    output_dir = "data/translated"
    os.makedirs(output_dir, exist_ok=True)
    for filename in filenames:
        print(f"Processing {filename}")
        filepath = f"data/raw/{filename}.txt"
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item) # list of dicts {{'id': 1, 'task': 'QI','language': 'en', 'origin_query': .. 'item_title': .. ,'label': '0' }
        batch_size = 50000
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            df_chunk = pd.DataFrame(chunk)
            df_chunk = translate_process_data(df_chunk, verbose=False)
            if i == 0:
                df = df_chunk
            else:
                df = pd.concat([df, df_chunk], ignore_index=True)
            df_chunk.to_csv(os.path.join(output_dir, 'chunks', f"translated_{filename}_chunk_{i}_{min(i+batch_size, len(data))}.csv"), index=False)
        df.to_csv(os.path.join(output_dir, f"translated_{filename}_full.csv"), index=False)