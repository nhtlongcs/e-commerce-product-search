import pandas as pd
import numpy as np

"""
Preprocessing utilities for text datasets.
"""



"""
Preprocessing utilities for text datasets.
"""

def remove_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with NaN values in 'item_title' column or in 'origin_query' column
    """
    print(f"Original dataset size: {len(df)}")
    nan_items = df[df['item_title'].isna()][['origin_query', 'item_title', 'label']]
    if not nan_items.empty:
        print(f"Found {len(nan_items)} rows with NaN item_title")
    
    df = df.dropna(subset=['item_title', 'origin_query'])
    print(f"Dataset size after removing NaN: {len(df)}")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on 'origin_query' and 'item_title' columns
    """
    original_size = len(df)
    df = df.drop_duplicates(subset=['origin_query', 'item_title'])
    print(f"Removed {original_size - len(df)} duplicate rows")
    return df

def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text fields by removing extra whitespace and handling empty strings
    """

    from neattext.functions import clean_text

    clean_fn = lambda x: clean_text(x, stopwords=False) if isinstance(x, str) else x

    # Clean origin_query
    df['origin_query'] = df['origin_query'].astype(str).str.strip()
    df['origin_query'] = df['origin_query'].replace(['', 'nan', 'None'], np.nan)
    df['origin_query'] = df['origin_query'].apply(lambda x: clean_fn(x) if isinstance(x, str) else x)
    
    # Clean item_title
    if 'item_title' in df.columns:
        df['item_title'] = df['item_title'].astype(str).str.strip()
        df['item_title'] = df['item_title'].replace(['', 'nan', 'None'], np.nan)
        df['item_title'] = df['item_title'].apply(lambda x: clean_fn(x) if isinstance(x, str) else x)

    # Clean category_path if it exists
    if 'category_path' in df.columns:
        df['category_path'] = df['category_path'].astype(str).str.strip()
        df['category_path'] = df['category_path'].replace(['', 'nan', 'None'], np.nan)
        df['category_path'] = df['category_path'].apply(lambda x: clean_fn(x) if isinstance(x, str) else x)

    
    
    return df

