import json 
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
    # filename = "train_QI"
    filenames = ["train_QI", "train_QC", "dev_QI", "dev_QC"]
    for filename in filenames:
        print(f"Processing {filename}")
        filepath = f"../data/{filename}.txt"
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
            df_chunk.to_csv(f"translated_{filename}_chunk_{i}_{min(i+batch_size, len(data))}.csv", index=False)
        df.to_csv(f"translated_{filename}_full.csv", index=False)