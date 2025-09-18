# QI
accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-bf16/gemma_3_12b_pt-QI_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-8000 config/QI/large_model_full.json config/QI/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-bf16/gemma_3_12b_pt-QI_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-17008 config/QI/data_fold0.json
accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-bf16/gemma_3_12b_pt-QI_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-8500 config/QI/large_model_full.json config/QI/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-bf16/gemma_3_12b_pt-QI_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-16994 config/QI/data_fold0.json
accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-bf16/gemma_3_12b_pt-QI_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-7500 config/QI/large_model_full.json config/QI/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-bf16/gemma_3_12b_pt-QI_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-17014 config/QI/data_fold0.json

# QC -tf -todo

# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QC-outputs-transfer/gemma-3-QC_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-10000 config/QC/large_model_full.json config/QC/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QC-outputs-transfer/gemma-3-QC_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-11500 config/QC/large_model_full.json config/QC/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QC-outputs-transfer/gemma-3-QC_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-15000 config/QC/large_model_full.json config/QC/data_fold0.json

# QI -tf -todo
accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-transfer/gemma-3-QI_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-8000 config/QI/large_model_full.json config/QI/data_fold0.json
accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-transfer/gemma-3-QI_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-7000 config/QI/large_model_full.json config/QI/data_fold0.json
accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-transfer/gemma-3-QI_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-7500 config/QI/large_model_full.json config/QI/data_fold0.json

# gemma_2_9b-QI_2lang-2048-fold-full-8802
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QI-outputs-bf16/gemma_3_12b_pt-QI_2lang-2048-fold-full-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-21250 config/QI/large_model_full.json config/QI/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py /home/tnguyenho/workspace/e-commerce-product-search/baseline/QC-outputs-bf16/gemma_3_12b_pt-QC_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT config/QC/large_model_full.json config/QC/data_fold0.json
# torchrun --nproc_per_node 4 --nnodes=1 eval.py QI-outputs-transfer/gemma-3-QI_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT config/QI/data_fold0.json