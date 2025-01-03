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
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --name ucf_101 \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/ucf_101 \
  --latent_dimension 512 \
  --dataroot  data/deposition_data \
  --checkpoints_dir checkpoints \
  --img_g_weights pretrained_weights/ucf-256-fid41.6761-snapshot-006935-generator.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 16 \
  --workers 1 \
  --style_gan_size 256 \
  --total_epoch 100 
#   --load_pretrain_path /path/to/checkpoints \
#   --load_pretrain_epoch 0
  ```