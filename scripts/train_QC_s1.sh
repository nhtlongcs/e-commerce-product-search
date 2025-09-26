echo "Training QC task - stage 1 - fold 0"
echo "If you want to train other folds, please uncomment the corresponding lines in the train_QC_s1.sh file"
torchrun --nproc_per_node 4 --nnodes=1 train.py models/gemma-3-12b-pt config/QC/data_fold0.json config/QC/large_model_full.json config/QC/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py models/gemma-3-12b-pt config/QC/data_fold1.json config/QC/large_model_full.json config/QC/train_14b.json
# torchrun --nproc_per_node 4 --nnodes=1 train.py models/gemma-3-12b-pt config/QC/data_fold2.json config/QC/large_model_full.json config/QC/train_14b.json