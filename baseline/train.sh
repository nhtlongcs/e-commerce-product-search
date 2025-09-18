# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-3-12b-pt config/QC/data_fold2.json config/QC/large_model_full.json config/QC/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-2-9b  config/QC/data_fold1.json config/QC/large_model_full.json config/QC/train_10b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-2-9b  config/QC/data_fold2.json config/QC/large_model_full.json config/QC/train_10b.json

# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-3-12b-pt config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-3-12b-pt config/QI/data_fold1.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-3-12b-pt config/QI/data_fold2.json config/QI/large_model_full.json config/QI/train_14b.json

torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QC-outputs-bf16/gemma_3_12b_pt-QC_2lang-2048-fold-full-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-18750 config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_14b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QC-outputs-bf16/gemma_3_12b_pt-QC_2lang-2048-fold-full-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-18750 config/QI/data_fold1.json config/QI/large_model_full.json config/QI/train_14b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QC-outputs-bf16/gemma_3_12b_pt-QC_2lang-2048-fold-full-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-18750 config/QI/data_fold2.json config/QI/large_model_full.json config/QI/train_14b.json


# transfer / Backbone QI -> train QC 
# transfer / Backbone QC -> train QI

# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-3-12b-it config/QC/data_fold0.json config/QC/large_model_full.json config/QC/train_14b.json


# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-2-9b config/QC/data_fold1.json config/QC/large_model_full.json config/QC/train_10b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-2-9b config/QC/data_fold2.json config/QC/large_model_full.json config/QC/train_10b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-2-9b config/QI/data_full.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/support/llm/Qwen3-14B config/QI/data_fold1.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/support/llm/Qwen3-14B config/QI/data_fold2.json config/QI/large_model_full.json config/QI/train_14b.json

# ko chay dc (TODO: fix)
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/Falcon3-10B-Base config/QI/data_fold0.json config/QI/small_model.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/Qwen2.5-14B-Instruct-1M config/QI/data_fold0.json config/QI/small_model.json config/QI/train_14b.json

# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-S/ config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_7b.json

# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-L/ config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-L/ config/QI/data_fold1.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-L/ config/QI/data_fold2.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/support/llm/Qwen3-14B config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-M/ config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_7b.json
