python -m classifier.train_multi_task_resnet3d_classifier \
--name classifier_peds \
--save_dir /data2/braingan/peds_classifier \
--csv_dir /data2/SharonFolder/hikaru/peds_head_ct_numpy \
--gpu_ids 2,3 \
--num_epochs 60 \
--lr 1e-5 \
--batch_size 2 \
--num_workers 4 \
--num_slices 32 \
--epochs_per_save 5 \
--slice_size 256 \
--features Bleed Fracture Tumor Vent/EVD Craniotomy Normal \
--focal_loss_alphas 0.69 0.95 0.94 0.81 0.84 0.64

