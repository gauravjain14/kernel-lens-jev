import { readFileSync } from 'node:fs';
export const releaseCases = [
  {name:'rms-norm-release',file:'rms.cu',source:readFileSync('test/fixtures/rms-norm-release.cu','utf8'),cursor:0,
    issues:['cuda.serial_reduction','review.cuda-'], absent:[] as string[]},
  {name:'sgemm-coalesced-release',file:'gemm.cu',source:readFileSync('test/fixtures/sgemm-coalesced-release.cu','utf8'),cursor:2,
    issues:['review.cuda-reuse'],absent:['cuda.global_accumulation']},
  {name:'warp-improved',file:'warp.cu',cursor:0,source:`__global__ void reduce(float* out, const float* input) {
    float sum = input[threadIdx.x];
    for (int offset=16; offset>0; offset/=2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x == 0) out[0] = sum;
}`,issues:[],absent:['cuda.serial_reduction','review.cuda-thread-work','review.cuda-accumulator']},
  {name:'training-risk',file:'training.py',cursor:3,source:`import torch
from torch.utils.data import DataLoader

def train(model, dataset, optimizer):
    loader = DataLoader(dataset, batch_size=32)
    history = []
    for step in range(100):
        x, y = next(iter(loader))
        x = x.to('cuda')
        y = y.to('cuda')
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        history.append(loss)
        print(loss.item())
    return history
`,issues:['training.iterator_recreated','review.train-host-sync'],absent:[]},
  {name:'training-improved',file:'training.py',cursor:3,source:`import torch
from torch.utils.data import DataLoader

def train(model, dataset, optimizer):
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    total = torch.zeros((), device='cuda')
    for x, y in loader:
        x = x.to('cuda', non_blocking=True)
        y = y.to('cuda', non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        total += loss.detach()
    return total
`,issues:[],absent:['review.train-loader','review.train-host-sync','review.train-retention']},
  {name:'inference-risk',file:'predict.py',cursor:2,source:`import torch

def predict(model, requests):
    model.eval()
    outputs = []
    for request in requests:
        x = torch.tensor(request, device='cuda', dtype=torch.float32)
        prediction = model(x)
        outputs.append(prediction.cpu())
    return outputs
`,issues:['review.infer-'],absent:[]},
  {name:'inference-improved',file:'predict.py',cursor:2,source:`import torch

@torch.inference_mode()
def predict(model, batch):
    model.eval()
    x = batch.to('cuda', non_blocking=True)
    predictions = model(x)
    return predictions
`,issues:[],absent:['review.infer-grad','review.infer-transfers','review.infer-retention','review.tensor-batching']},
];
