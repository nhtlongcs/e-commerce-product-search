torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/ModernBERT-large config/QI/data_fold0.json config/QI/small_model.json config/QI/train_1b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/Falcon3-10B-Base config/QI/data_fold0.json config/QI/small_model.json config/QI/train_14b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/Qwen2.5-14B-Instruct-1M config/QI/data_fold0.json config/QI/small_model.json config/QI/train_14b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-2-9b config/QI/data_fold1.json config/QI/large_model_full.json config/QI/train_14b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/gemma-2-9b config/QI/data_fold2.json config/QI/large_model_full.json config/QI/train_14b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/support/llm/Qwen3-14B config/QI/data_fold1.json config/QI/large_model_full.json config/QI/train_14b.json
torchrun --nproc_per_node 4 --nnodes=1 train.py /home/support/llm/Qwen3-14B config/QI/data_fold2.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-S/ config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_7b.json

# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-L/ config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-L/ config/QI/data_fold1.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-L/ config/QI/data_fold2.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/support/llm/Qwen3-14B config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py /home/tnguyenho/spinning-storage/tnguyenho/llms/eCeLLM-M/ config/QI/data_fold0.json config/QI/large_model_full.json config/QI/train_7b.json
