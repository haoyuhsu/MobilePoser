

### Finetune on DIP-IMU
# python -m mobileposer.finetune_dipimu \
#     --train-data data/processed_datasets/eval/dip_train.pt \
#     --checkpoint-dir checkpoints/all_MotionGV_no_noise \
#     --output-dir checkpoints/finetuned_dip \
#     --epochs 50 --lr 5e-5 --batch-size 32


### Evaluate on DIP-IMU test set
# python -m mobileposer.evaluate_dipimu \
#     --test-data data/processed_datasets/eval/dip_test.pt \
#     --checkpoint-dir checkpoints/finetuned_dip \
#     --base-checkpoint-dir checkpoints/all_MotionGV_no_noise \
#     --combo lw_rp_h \
#     --save-predictions ./predictions/dip


### Finetune on IMUPoser
python -m mobileposer.finetune_dipimu \
    --train-data data/processed_datasets/eval/imuposer_train.pt \
    --checkpoint-dir checkpoints/1_all \
    --output-dir checkpoints/finetuned_imuposer \
    --epochs 50 --lr 5e-5 --batch-size 32


### Evaluate on IMUPoser test set
python -m mobileposer.evaluate_dipimu \
    --test-data data/processed_datasets/eval/imuposer_test.pt \
    --checkpoint-dir checkpoints/finetuned_imuposer \
    --base-checkpoint-dir checkpoints/1_all \
    --combo lw_rp_h \
    --save-predictions ./predictions/imuposer