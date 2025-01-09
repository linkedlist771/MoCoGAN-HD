## 1. Finetune

**deposition_data**

1. generate npy
```bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/ucf_101 \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_weights/ucf-256-fid41.6761-snapshot-006935-generator.pt \
  --style_gan_size 256 \
  --gpu 0
```

2. train

```bash
CUDA_VISIBLE_DEVICES=3 python -W ignore train.py --name ucf_101 \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/ucf_101 \
  --latent_dimension 512 \
  --dataroot  data/deposition_data \
  --checkpoints_dir checkpoints \
  --img_g_weights pretrained_weights/ucf-256-fid41.6761-snapshot-006935-generator.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 8 \
  --workers 1 \
  --style_gan_size 256 \
  --total_epoch 100 
#   --load_pretrain_path /path/to/checkpoints \
#   --load_pretrain_epoch 0
  ```
- nohup type:
```bash

CUDA_VISIBLE_DEVICES=3  nohup python -W ignore train.py --name ucf_101 \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/ucf_101 \
  --latent_dimension 512 \
  --dataroot  data/deposition_data \
  --checkpoints_dir checkpoints \
  --img_g_weights pretrained_weights/ucf-256-fid41.6761-snapshot-006935-generator.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 8 \
  --workers 1 \
  --style_gan_size 256 \
  --total_epoch 100 > $(date +%m_%d).log 2>&1 &
```

3. inference:
```bash
python -W ignore evaluate.py \
  --save_pca_path pca_stats/ucf_101 \
  --latent_dimension 512 \
  --style_gan_size 256 \
  --img_g_weights checkpoints/ucf_101_20250103_090743/modelG_epoch_36.pth \
  --load_pretrain_path checkpoints/ucf_101_20250103_090743 \
  --load_pretrain_epoch 36 \
  --results results/ucf_101 \
  --num_test_videos 10
```