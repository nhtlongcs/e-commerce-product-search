import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# Directory to save images
IMG_DIR = "eda_figs"
os.makedirs(IMG_DIR, exist_ok=True)
fig_paths = []

def save_fig(fig, name):
    path = os.path.join(IMG_DIR, name)
    # Save as PDF
    fig.write_image(path, format="pdf", scale=2, width=1000, height=600)
    fig_paths.append(path)

# =========================
# Query - Item (QI)
# =========================

df_qi = pd.read_csv('data/translated/translated_train_QI_full_fold.csv')
df_qi = df_qi.dropna(subset=['item_title'])

fig = px.histogram(df_qi, x='label', title='Label Distribution', color='label', text_auto=True)
save_fig(fig, "qi_label_dist.pdf")
fig = px.histogram(df_qi, x='language', title='Language Distribution', color='language', text_auto=True,
                   category_orders={'language': df_qi['language'].value_counts().index.tolist()})
save_fig(fig, "qi_language_dist.pdf")

df_qi['language'] = df_qi.apply(lambda x: 'unk' if (x['language'] == 'en' and x['origin_query'].lower() != x['translated_query'].lower()) else x['language'], axis=1)

fig = px.histogram(df_qi, x='label', title='Label Distribution (After Language Correction)', color='label', text_auto=True)
save_fig(fig, "qi_label_dist_corrected.pdf")
fig = px.histogram(df_qi, x='language', title='Language Distribution (After Correction)', color='language', text_auto=True,
                   category_orders={'language': df_qi['language'].value_counts().index.tolist()})
save_fig(fig, "qi_language_dist_corrected.pdf")

absolute_crosstab = pd.crosstab(df_qi['language'], df_qi['label'])
fig = go.Figure()
for label in absolute_crosstab.columns:
    fig.add_trace(go.Bar(
        x=absolute_crosstab.index,
        y=absolute_crosstab[label],
        name=f'Label {label}'
    ))
fig.update_layout(barmode='stack', title='Label Percentage by Language', yaxis_title='Count', xaxis_title='Language')
save_fig(fig, "qi_label_by_language.pdf")

df_qi['query_length'] = df_qi['origin_query'].str.len()
df_qi['title_length'] = df_qi['item_title'].str.len()
fig = px.histogram(df_qi, x='query_length', nbins=50, title='Query Length Distribution', color_discrete_sequence=['blue'])
save_fig(fig, "qi_query_length.pdf")
fig = px.histogram(df_qi, x='title_length', nbins=50, title='Title Length Distribution', color_discrete_sequence=['red'])
save_fig(fig, "qi_title_length.pdf")

def get_top_ngrams(corpus, n=None, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

top_query_words = get_top_ngrams(df_qi['origin_query'], n=10)
top_title_words = get_top_ngrams(df_qi['item_title'], n=10)
fig = px.bar(x=[w for w, _ in top_query_words], y=[c for _, c in top_query_words], title="Top 10 Words in 'origin_query'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qi_top10_query_words.pdf")
fig = px.bar(x=[w for w, _ in top_title_words], y=[c for _, c in top_title_words], title="Top 10 Words in 'item_title'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qi_top10_title_words.pdf")

df_qi['translated_query_length'] = df_qi['translated_query'].str.len()
fig = px.histogram(df_qi, x='translated_query_length', nbins=50, title='Translated Query Length Distribution', color_discrete_sequence=['blue'])
save_fig(fig, "qi_translated_query_length.pdf")
fig = px.histogram(df_qi, x='title_length', nbins=50, title='Title Length Distribution', color_discrete_sequence=['red'])
save_fig(fig, "qi_title_length2.pdf")

top_query_words = get_top_ngrams(df_qi['translated_query'], n=10)
top_title_words = get_top_ngrams(df_qi['item_title'], n=10)
fig = px.bar(x=[w for w, _ in top_query_words], y=[c for _, c in top_query_words], title="Top 10 Words in 'translated_query'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qi_top10_translated_query_words.pdf")
fig = px.bar(x=[w for w, _ in top_title_words], y=[c for _, c in top_title_words], title="Top 10 Words in 'item_title'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qi_top10_title_words2.pdf")

fig = make_subplots(rows=1, cols=2, subplot_titles=('Query Length vs. Label', 'Title Length vs. Label'))
for label in sorted(df_qi['label'].unique()):
    fig.add_trace(go.Box(y=df_qi[df_qi['label'] == label]['query_length'], name=f'Label {label} (Query)', boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=df_qi[df_qi['label'] == label]['title_length'], name=f'Label {label} (Title)', boxmean=True), row=1, col=2)
fig.update_layout(title_text="Text Length vs. Label")
save_fig(fig, "qi_length_vs_label.pdf")

def jaccard_similarity(query, title):
    query_set = set(str(query).lower().split())
    title_set = set(str(title).lower().split())
    intersection = query_set.intersection(title_set)
    union = query_set.union(title_set)
    if not union:
        return 0.0
    return len(intersection) / len(union)

df_qi['jaccard_similarity'] = df_qi.apply(lambda row: jaccard_similarity(row['origin_query'], row['item_title']), axis=1)
fig = px.box(df_qi, x='label', y='jaccard_similarity', title='Jaccard Similarity between Query and Title by Label')
save_fig(fig, "qi_jaccard_query_title.pdf")

df_qi['jaccard_similarity_en'] = df_qi.apply(lambda row: jaccard_similarity(row['translated_query'], row['item_title']), axis=1)
fig = px.box(df_qi, x='label', y='jaccard_similarity_en', title='Jaccard Similarity between Translated Query and Item Title by Label')
save_fig(fig, "qi_jaccard_translated_query_title.pdf")

# =========================
# Query - Category (QC)
# =========================

df_qc = pd.read_csv('data/translated/translated_train_QC_full_fold_v2.csv')
df_qc = df_qc.dropna(subset=['category_path'])

fig = px.histogram(df_qc, x='label', title='Label Distribution', color='label', text_auto=True)
save_fig(fig, "qc_label_dist.pdf")
fig = px.histogram(df_qc, x='language', title='Language Distribution', color='language', text_auto=True,
                   category_orders={'language': df_qc['language'].value_counts().index.tolist()})
save_fig(fig, "qc_language_dist.pdf")

df_qc['language'] = df_qc.apply(lambda x: 'unk' if (x['language'] == 'en' and x['origin_query'].lower() != x['translated_query'].lower()) else x['language'], axis=1)

fig = px.histogram(df_qc, x='label', title='Label Distribution (After Language Correction)', color='label', text_auto=True)
save_fig(fig, "qc_label_dist_corrected.pdf")
fig = px.histogram(df_qc, x='language', title='Language Distribution (After Correction)', color='language', text_auto=True,
                   category_orders={'language': df_qc['language'].value_counts().index.tolist()})
save_fig(fig, "qc_language_dist_corrected.pdf")

absolute_crosstab = pd.crosstab(df_qc['language'], df_qc['label'])
fig = go.Figure()
for label in absolute_crosstab.columns:
    fig.add_trace(go.Bar(
        x=absolute_crosstab.index,
        y=absolute_crosstab[label],
        name=f'Label {label}'
    ))
fig.update_layout(barmode='stack', title='Label Percentage by Language', yaxis_title='Count', xaxis_title='Language')
save_fig(fig, "qc_label_by_language.pdf")

df_qc['query_length'] = df_qc['origin_query'].str.len()
df_qc['category_path_length'] = df_qc['category_path'].str.len()
fig = px.histogram(df_qc, x='query_length', nbins=50, title='Query Length Distribution', color_discrete_sequence=['blue'])
save_fig(fig, "qc_query_length.pdf")
fig = px.histogram(df_qc, x='category_path_length', nbins=50, title='Category Path Length Distribution', color_discrete_sequence=['red'])
save_fig(fig, "qc_category_path_length.pdf")

top_query_words = get_top_ngrams(df_qc['origin_query'], n=10)
top_title_words = get_top_ngrams(df_qc['category_path'], n=10)
fig = px.bar(x=[w for w, _ in top_query_words], y=[c for _, c in top_query_words], title="Top 10 Words in 'origin_query'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qc_top10_query_words.pdf")
fig = px.bar(x=[w for w, _ in top_title_words], y=[c for _, c in top_title_words], title="Top 10 Words in 'category_path'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qc_top10_category_path_words.pdf")

df_qc['translated_query_length'] = df_qc['translated_query'].str.len()
fig = px.histogram(df_qc, x='translated_query_length', nbins=50, title='Translated Query Length Distribution', color_discrete_sequence=['blue'])
save_fig(fig, "qc_translated_query_length.pdf")
fig = px.histogram(df_qc, x='category_path_length', nbins=50, title='Category Path Length Distribution', color_discrete_sequence=['red'])
save_fig(fig, "qc_category_path_length2.pdf")

top_query_words = get_top_ngrams(df_qc['translated_query'], n=10)
top_title_words = get_top_ngrams(df_qc['category_path'], n=10)
fig = px.bar(x=[w for w, _ in top_query_words], y=[c for _, c in top_query_words], title="Top 10 Words in 'translated_query'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qc_top10_translated_query_words.pdf")
fig = px.bar(x=[w for w, _ in top_title_words], y=[c for _, c in top_title_words], title="Top 10 Words in 'category_path'", labels={'x': 'Word', 'y': 'Count'})
save_fig(fig, "qc_top10_category_path_words2.pdf")

fig = make_subplots(rows=1, cols=2, subplot_titles=('Query Length vs. Label', 'Category Path Length vs. Label'))
for label in sorted(df_qc['label'].unique()):
    fig.add_trace(go.Box(y=df_qc[df_qc['label'] == label]['query_length'], name=f'Label {label} (Query)', boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=df_qc[df_qc['label'] == label]['category_path_length'], name=f'Label {label} (Category Path)', boxmean=True), row=1, col=2)
fig.update_layout(title_text="Text Length vs. Label")
save_fig(fig, "qc_length_vs_label.pdf")

def jaccard_similarity_qc(query, title):
    query_set = set(str(query).lower().split())
    title_set = set(str(title).lower().split())
    intersection = query_set.intersection(title_set)
    union = query_set.union(title_set)
    if not union:
        return 0.0
    return len(intersection) / len(union)

df_qc['jaccard_similarity'] = df_qc.apply(lambda row: jaccard_similarity_qc(row['origin_query'], row['category_path']), axis=1)
fig = px.box(df_qc, x='label', y='jaccard_similarity', title='Jaccard Similarity between Query and Category Path by Label')
save_fig(fig, "qc_jaccard_query_category_path.pdf")

df_qc['jaccard_similarity_en'] = df_qc.apply(lambda row: jaccard_similarity_qc(row['translated_query'], row['category_path']), axis=1)
fig = px.box(df_qc, x='label', y='jaccard_similarity_en', title='Jaccard Similarity between Translated Query and Category Path by Label')
save_fig(fig, "qc_jaccard_translated_query_category_path.pdf")
