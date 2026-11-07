"""Short synthetic CUDA-graph activity; does not load or benchmark the model."""
import multiprocessing
import sys
import time
import torch


def work(device, barrier, end, reserve_gib):
    torch.cuda.set_device(device)
    reserve=None
    if reserve_gib:
        reserve=torch.empty((reserve_gib*(2**30)//2,),device='cuda',dtype=torch.float16)
        reserve.fill_(1)
        torch.cuda.synchronize(device)
        print('CONTROL_RESERVE device='+str(device)+' bytes='+str(reserve.numel()*reserve.element_size()),flush=True)
    x=torch.ones((1024,1024),device='cuda',dtype=torch.float16)
    for _ in range(3): y=x@x
    torch.cuda.synchronize(device)
    first=True
    while time.monotonic()<end:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y=x@x
        graph.replay();torch.cuda.synchronize(device)
        if first:
            first=False
            if barrier.wait()==0: print('GRAPH_CONTROL_READY',flush=True)
        del graph,y


def main():
    count=int(sys.argv[2]) if len(sys.argv)==3 else 8
    if count not in (2,8) or torch.cuda.device_count() != count: raise RuntimeError('declared visible GPU count required')
    if len(sys.argv) not in (2,3) or sys.argv[1] not in ('0','64','128'): raise ValueError('unsupported control reservation')
    reserve_gib=int(sys.argv[1])
    context=multiprocessing.get_context('spawn');barrier=context.Barrier(count,timeout=120);end=time.monotonic()+300
    processes=[context.Process(target=work,args=(device,barrier,end,reserve_gib)) for device in range(count)]
    for process in processes: process.start()
    for process in processes: process.join()
    if any(process.exitcode!=0 for process in processes): raise RuntimeError('GPU control worker failed')


if __name__=='__main__': main()
