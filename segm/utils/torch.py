import os
import torch
import torch.distributed
"""
GPU wrappers
"""
use_gpu = False
gpu_id = 0
device = None
distributed = False
dist_rank = 0
world_size = 1
def set_gpu_mode(mode, local_rank, need_gpus):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    # gpu_id = torch.cuda.device_count()
    # gpu_id = int(os.environ("LOCAL_RANK"))
    gpu_numbers = torch.cuda.device_count()
    if gpu_numbers >= need_gpus and need_gpus:
        gpu_id = ','.join(str(i) for i in range(need_gpus))
    else:
        print(
            'get this message because '
            'number of actual gpus are less than required or '
            'requiring 0 gpu')
        print('max gpu numbers are:', gpu_numbers)
        gpu_id = ','.join(str(i) for i in range(gpu_numbers))
    dist_rank = 0
    # world_size = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))
    world_size = torch.cuda.device_count()
    distributed = world_size > 1
    use_gpu = mode
    # torch.cuda.set_device('cuda:' + gpu_id[0])
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
