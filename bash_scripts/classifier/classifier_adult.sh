python -m classifier.train_multi_task_resnet3d_classifier \
--name adult_256_focal_loss_transforms \
--save_dir /data2/braingan/classifier_adult \
--train_age_group adult \
--gpu_ids 2,3 \
--num_epochs 60 \
--lr 1e-5 \
--batch_size 8 \
--num_workers 4 \
--num_slices 32 \
--epochs_per_save 5 \
--epochs_per_eval 5 \
--slice_size 256 \
--focal_loss_alphas 0.86 0.89 0.87 0.87 0.83 0.58