from pathlib import Path
import pandas as pd 
import numpy as np
import random
from sklearn.model_selection import StratifiedGroupKFold
random_seed = 54

np.random.seed(random_seed)
random.seed(random_seed)





def detect_leakage(df: pd.DataFrame, group_col: str, fold_col: str = 'fold'):
    """
    Detects if any group appears in more than one fold, which would indicate
    leakage between training and validation sets.

    Args:
        df (pd.DataFrame): The dataframe containing the data and fold assignments.
        group_col (str): The name of the column to check for leakage (e.g., 'origin_query').
        fold_col (str): The name of the column containing the fold numbers.
    """
    print(f"--- Starting Leakage Detection for column: '{group_col}' ---")
    
    # Get all unique fold numbers, handling cases where folds might not start at 0 or be sequential
    folds = sorted(df[fold_col].unique())
    
    leakage_found = False
    for fold_num in folds:
        print(f"\nChecking Fold {fold_num}...")

        # Define the training and validation sets for the current fold
        train_df = df[df[fold_col] != fold_num]
        val_df = df[df[fold_col] == fold_num]

        # Get the unique group identifiers for each set
        train_groups = set(train_df[group_col].unique())
        val_groups = set(val_df[group_col].unique())

        # Find the intersection, which represents the leaked groups
        intersection = train_groups.intersection(val_groups)

        # some stats here:
        # number labels (train, val)
        # number main_category (train, val)
        # number language (train, val) 

        # print(f"Number of labels in (train, val): {train_df['label'].value_counts()}, {val_df['label'].value_counts()}")  
        print(f"Percentage of labels in (train):\n{train_df['label'].value_counts(normalize=True)}")
        print(f"Percentage of labels in (val):\n{val_df['label'].value_counts(normalize=True)}")
        print(f"Ratio of labels in (val): {val_df['label'].value_counts() / df['label'].value_counts()}")
        print(f"Number of language (train, val): {len(train_df['language'].unique()), len(val_df['language'].unique())}")
        if not intersection:
            print(f"  ✅ SUCCESS: No leakage found. Validation groups are unique to this fold.")
        else:
            print(f"  ❌ FAILED: Leakage detected! {len(intersection)} groups are in both train and val sets.")
            print(f"     Leaked groups: {intersection}")
            leakage_found = True
            
    print("\n--- Leakage Detection Complete ---")
    if not leakage_found:
        print("Overall Result: No leakage was detected across all folds. Your setup is correct!")
    else:
        print("Overall Result: Leakage was found. Review your folding strategy.")


def run_QI():
    file_path = Path('data/translated/translated_train_QI_full.csv')
    out_path = file_path.parent / 'translated_train_QI_full_fold.csv'
    df = pd.read_csv(file_path)

    df['language'] = df.apply(lambda x: 'unk' if ( x['language'] == 'en' and x['origin_query'].lower() != x['translated_query'].lower()) else x['language'], axis=1)

    df['fold'] = -1
    df['stratify_col'] = df['language'].astype(str) + '_' + df['label'].astype(str)

    # Initialize the splitter
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Generate folds

    for i, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df['stratify_col'], groups=df['origin_query'])):
        df.loc[val_idx, 'fold'] = i

    # Verify that a single origin_query only exists in one fold
    query_fold_counts = df.groupby('origin_query')['fold'].nunique()
    print(f"Number of queries split across multiple folds: {(query_fold_counts > 1).sum()}")
    # Expected output: Number of queries split across multiple folds: 0


    detect_leakage(df=df, group_col='origin_query')

    print(df.fold.value_counts())

    df.to_csv(out_path, index=False)

def run_QC():

    file_path = Path('data/translated/translated_train_QC_full.csv')
    out_path = file_path.parent / 'translated_train_QC_full_fold.csv'
    df = pd.read_csv(file_path)

    # ==============================================================================
    # PHẦN CODE ĐÃ SỬA ĐỔI
    # ==============================================================================

    # Số lượng phần tử trong tiền tố path để nhóm (bạn có thể thay đổi số 3 này)
    N_PREFIX = 3

    # Bước 1: Tạo cột group mới dựa trên tiền tố của 'category_path'
    # Hàm này sẽ lấy N_PREFIX phần đầu của path, ví dụ: 'a,b,c,d' -> 'a,b,c'
    def get_path_prefix(path, n):
        parts = path.split(',')
        return ','.join(parts[:n])
    k=1
    df['main_category'] = df['category_path'].str.split(',', n=k).str[:k].str.join(', ').str.strip().str.lower()
    
    df['category_group'] = df['category_path'].apply(lambda x: get_path_prefix(x, N_PREFIX))


    # Bước 2: Chuẩn bị cột để phân tầng (giống như code của bạn)
    df['fold'] = -1
    df['stratify_col'] = df['language'].astype(str) + '_' + df['main_category'].astype(str) + '_' + df['label'].astype(str)


    # Bước 3: Thực hiện chia Fold với Group mới
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Chú ý sự thay đổi ở tham số `groups`
    for i, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df['stratify_col'], groups=df['category_group'])):
        df.loc[val_idx, 'fold'] = i

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    run_QI()
    run_QC()