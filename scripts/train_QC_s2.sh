# Pretrain QI - Domain Adapt QC
echo "Training QC task - stage 2 - QI pretraining phase"
torchrun --nproc_per_node 4 --nnodes=1 train.py models/gemma-3-12b-pt config/QI/data_full.json config/QI/large_model_full.json config/QI/train_14b.json
# PRETRAINED_CKPT=''
echo "Before running the next commands, please set PRETRAINED_CKPT to the path of the best checkpoint from the previous command"
echo "If you want to train other folds, please uncomment the corresponding lines in the train_QC_s2.sh file"
torchrun --nproc_per_node 4 --nnodes=1 train.py $PRETRAINED_CKPT config/QC/data_fold0.json config/QC/large_model_full.json config/QC/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py $PRETRAINED_CKPT config/QC/data_fold1.json config/QC/large_model_full.json config/QC/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py $PRETRAINED_CKPT config/QC/data_fold2.json config/QC/large_model_full.json config/QC/train_14b.json