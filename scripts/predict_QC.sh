# QC - stage 1
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QC-stage-01/gemma-3-QC_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-10000 config/QC/large_model_full.json config/QC/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QC-stage-01/gemma-3-QC_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-11500 config/QC/large_model_full.json config/QC/data_fold1.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QC-stage-01/gemma-3-QC_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-15000 config/QC/large_model_full.json config/QC/data_fold2.json

# QC - stage 2
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QC-stage-02/gemma-3-QC_2lang-2048-fold0-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-10000 config/QC/large_model_full.json config/QC/data_fold0.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QC-stage-02/gemma-3-QC_2lang-2048-fold1-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-11500 config/QC/large_model_full.json config/QC/data_fold1.json
# accelerate launch --mixed_precision bf16 --multi_gpu  predict.py models/QC-stage-02/gemma-3-QC_2lang-2048-fold2-lr-1e-05-bf16-SchedulerType.CONSTANT/checkpoint-15000 config/QC/large_model_full.json config/QC/data_fold2.json