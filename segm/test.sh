python -m torch.distributed.launch
--nproc_per_node=1
-m segm.test
--log-dir segm/vit-large
--local_rank 0
--only_test True
--epochs=1