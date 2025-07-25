from langdetect import detect 
from .translate import TranslatorWrapper
from polyfuzz import PolyFuzz
from tqdm import tqdm

tqdm.pandas()

def is_english(text):
    try:
        return detect(text) == "en"
    except Exception as e:
        print(f"Error detecting language: {e}")
        return False


def save_results(df, output_path):
    df.to_csv(output_path, index=False)


def fuzzy_match(from_list, to_list):
    model = PolyFuzz("TF-IDF")
    model.match(from_list, to_list)
    inspect_df = model.get_matches()
    return inspect_df


def split_translate_merge(sentence, src_lang, method="offline", verbose=False):
    try:
        mt = TranslatorWrapper()
        if len(sentence) == 0:
            return ""
        try:
            if src_lang is None or src_lang == "en":
                src_lang = detect(sentence)
                # double check
            if src_lang == "en": 
                return sentence
        except Exception as e:
            if method == "api":
                # print(f"Language detection failed: {e}")
                return sentence
            if method == "offline":
                src_lang = None
                # dont need src_lang for offline translation
        chunk_size = 5000
        chunks = [
            sentence[i : i + chunk_size]
            for i in range(0, len(sentence), chunk_size)
        ]
        translated_chunks = [
            mt.translate(chunk, src_lang, method=method)
            for chunk in chunks
        ]
        merged_sentence = "".join(translated_chunks)
        if verbose:
            print(merged_sentence)
        return merged_sentence
    except Exception as e:
        if verbose:
            print(f"Translation error: {e}")
        return ""



