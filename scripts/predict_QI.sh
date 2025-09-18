# QI - stage 1
accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QI-stage-01/gemma_3_12b_pt-QI_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-8000 config/QI/large_model_full.json config/QI/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QI-stage-01/gemma_3_12b_pt-QI_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-8500 config/QI/large_model_full.json config/QI/data_fold1.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QI-stage-01/gemma_3_12b_pt-QI_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-7500 config/QI/large_model_full.json config/QI/data_fold2.json

# QI - stage 2
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QI-stage-02/gemma-3-QI_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-8000 config/QI/large_model_full.json config/QI/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QI-stage-02/gemma-3-QI_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-7000 config/QI/large_model_full.json config/QI/data_fold1.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QI-stage-02/gemma-3-QI_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-7500 config/QI/large_model_full.json config/QI/data_fold2.json

