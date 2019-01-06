import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from benchmark import benchmarking
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import os
import sys
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

#if args.data_test == 'video':
#    from videotester import VideoTester
#    model = model.Model(args, checkpoint)
#    t = VideoTester(args, model, checkpoint)
#    t.test()
#else:
#    if checkpoint.ok:
#        loader = data.Data(args)
#        model = model.Model(args, checkpoint)
#        loss = loss.Loss(args, checkpoint) if not args.test_only else None
#        t = Trainer(args, loader, model, loss, checkpoint)
#        for i in range(1):
 #           #t.train()
 #           psnr = t.test()
 #           print("main", psnr)

  #      checkpoint.done()
model2 = model.Model(args, checkpoint)
loss = loss.Loss(args, checkpoint) if not args.test_only else None
@benchmarking(team=8, task=5, model=model2, preprocess_fn=None)
def inference(net, **kwargs):
    dev = kwargs['device']
    if(dev == 'cpu'):
        metric = do_cpu_inference()
    elif(dev == 'cuda'):
        metric = do_gpu_inference()
    return metric
def do_cpu_inference():
    loader = data.Data(args)
    args.cpu = False
    print("cpu inf")
    devi = 'cuda'
    model2.to(devi)
    #loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model2, loss, checkpoint)
    for i in range(1):
        psnr = t.test()
        print("main", psnr)
        checkpoint.done()
    return psnr

def do_gpu_inference():
    loader = data.Data(args)
    args.cpu = False
    devi = 'cuda'
    model2.to(devi)
    t = Trainer(args, loader, model2, loss, checkpoint)
    for i in range(1):
        psnr = t.test()
        print("main", psnr)
    checkpoint.done()
    return psnr

if __name__ == '__main__':
    print("args", args.cpu)
    inference(model)
