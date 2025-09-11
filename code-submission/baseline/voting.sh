#!/bin/bash

# Voting giữa các model 9B folds
python voting.py \
  submission-QC/gemma_2_9b-QC_2lang-2048-fold0-lr-1e-05-bf16-8891.txt \
  submission-QC/gemma_2_9b-QC_2lang-2048-fold1-lr-1e-05-bf16-8870.txt \
  submission-QC/gemma_2_9b-QC_2lang-2048-fold2-lr-1e-05-bf16-8891.txt \
  todo/vote_gemma_9b.txt

# Voting giữa các model 12B fold0 best/last
python voting.py \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold0-lr-1e-05-bf16-best-8936.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold0-lr-1e-05-bf16-last-8915.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold1-lr-1e-05-bf16-best-8912.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold1-lr-1e-05-bf16-last-8893.txt \
  todo/vote_gemma_12b_fold_0_1.txt

# Voting giữa tất cả 12B folds
python voting.py \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold0-lr-1e-05-bf16-best-8936.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold0-lr-1e-05-bf16-last-8915.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold1-lr-1e-05-bf16-best-8912.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold1-lr-1e-05-bf16-last-8893.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold2-lr-1e-05-bf16-8896.txt \
  todo/vote_gemma_12b_all.txt

# Voting kết hợp full model với folds
python voting.py \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold-full-8900.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold0-lr-1e-05-bf16-best-8936.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold1-lr-1e-05-bf16-best-8912.txt \
  todo/vote_gemma_12b_fullplus.txt

# Voting tất cả (9B + 12B)
python voting.py \
  submission-QC/gemma_2_9b-QC_2lang-2048-fold0-lr-1e-05-bf16-8891.txt \
  submission-QC/gemma_2_9b-QC_2lang-2048-fold1-lr-1e-05-bf16-8870.txt \
  submission-QC/gemma_2_9b-QC_2lang-2048-fold2-lr-1e-05-bf16-8891.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold0-lr-1e-05-bf16-best-8936.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold1-lr-1e-05-bf16-best-8912.txt \
  submission-QC/gemma_3_12b_pt-QC_2lang-2048-fold2-lr-1e-05-bf16-8896.txt \
  todo/vote_gemma_all.txt
