import type { PredictionCase } from './predictions';

const train = (body: string, setup = '', before = '') => `import torch
from torch import nn
import torch.nn.functional as F
model = nn.Linear(8, 2).cuda()
optimizer = torch.optim.AdamW(model.parameters())
${setup}
def train(loader):
    model.train()
${before}    for x, y in loader:
        x, y = x.cuda(), y.cuda()
${body}
`;
const update = `        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()`;
const predict = (body: string, decorator = '@torch.inference_mode()', setup = '') => `import torch
from torch import nn
import torch.nn.functional as F
model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5)).cuda().eval()
${setup}
${decorator}
def predict(batches):
${body}
`;
const predictLoop = `    outputs = []
    for x in batches:
        outputs.append(model(x.cuda()))
    return outputs`;
const at = (source: string, focus?: string) => Math.max(0, source.split('\n').findIndex(l => l.includes(focus ?? 'def ')));
const cases: PredictionCase[] = [];
function pair(name: string, pack: PredictionCase['pack'], id: string, bad: string, good: string, focus?: string, goodOutcomes = ['supported']) {
  cases.push({ name: `audit_${name}_concern`, file: `${name}.py`, source: bad, cursorLine: at(bad, focus), pack, expect: { [id]: ['concern'] } });
  cases.push({ name: `audit_${name}_supported`, file: `${name}.py`, source: good, cursorLine: at(good, focus), pack, expect: { [id]: goodOutcomes } });
}

pair('reset_order', 'training', 'train-grad-reset', train(update.replace('        optimizer.zero_grad(set_to_none=True)\n', '').replace('        optimizer.step()', '        optimizer.zero_grad()\n        optimizer.step()')), train(update));
pair('detached_loss', 'training', 'train-graph', train(update.replace('loss.backward()', 'loss.detach().requires_grad_().backward()')), train(update));
pair('optimizer_lifetime', 'training', 'train-optimizer', train('        optimizer = torch.optim.AdamW(model.parameters())\n' + update), train(update));
pair('update_order', 'training', 'train-step-order', train(update.replace('        loss.backward()\n        optimizer.step()', '        optimizer.step()\n        loss.backward()')), train(update));
pair('loader_lifecycle', 'training', 'train-loader', train(update).replace('    for x, y in loader:', '    for _ in range(100):\n        x, y = next(iter(loader))'), train(update));
pair('loss_contract', 'training', 'train-loss', train(update.replace('F.cross_entropy(logits, y)', 'F.cross_entropy(logits.softmax(dim=-1), y)')), train(update));
pair('scheduler_order', 'training', 'train-scheduler', train(update.replace('        optimizer.step()', '        scheduler.step()\n        optimizer.step()'), 'scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)'), train(update + '\n        scheduler.step()', 'scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)'));
const accumulated = `import torch
model = torch.nn.Linear(8, 2).cuda()
optimizer = torch.optim.AdamW(model.parameters())
def train(microbatches):
    model.train()
    optimizer.zero_grad()
    for i, (x, y) in enumerate(microbatches):
        x, y = x.cuda(), y.cuda()
        loss = torch.nn.functional.cross_entropy(model(x), y) / 4
        loss.backward()
        if (i + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
`;
pair('accumulation_window', 'training', 'train-accum-boundary', accumulated.replace('        x, y =', '        optimizer.zero_grad()\n        x, y ='), accumulated);
const meanAccumulation = accumulated.replace('    model.train()', '    """Optimize mean loss over groups of four equal-sized microbatches."""\n    model.train()');
pair('accumulation_scaling', 'training', 'train-accumulation', meanAccumulation.replace(' / 4\n', '\n'), meanAccumulation);
pair('training_grad_mode', 'training', 'train-mode', train(update.replace('        logits = model(x)', '        with torch.no_grad():\n            logits = model(x)')), train(update));
const validation = train(update + `
        model.eval()
        with torch.no_grad():
            for vx, vy in validation_loader:
                validation_loss = F.cross_entropy(model(vx.cuda()), vy.cuda())
        model.train()`);
pair('validation_phase', 'training', 'train-validation', validation.replace('        model.train()\n', ''), validation);
const ddp = `import torch
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(torch.nn.Linear(8, 2).cuda())
optimizer = torch.optim.AdamW(model.parameters())
def train(loader):
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        final = (i + 1) % 4 == 0
        with model.no_sync() if not final else nullcontext():
            loss = torch.nn.functional.cross_entropy(model(x.cuda()), y.cuda()) / 4
            loss.backward()
        if final:
            optimizer.step()
            optimizer.zero_grad()
`;
pair('ddp_accumulation', 'training', 'train-ddp-accum', ddp.replace('with model.no_sync() if not final else nullcontext():', 'with nullcontext():'), ddp);
const distributed = train(update, 'model = torch.nn.parallel.DistributedDataParallel(model)\nrank = torch.distributed.get_rank()');
pair('distributed_participation', 'training', 'train-distributed', distributed.replace('        loss.backward()', '        if rank == 0:\n            loss.backward()'), distributed);
pair('host_sync', 'training', 'train-host-sync', train(update + '\n        print(loss.item())'), train(update + '\n        total_loss += loss.detach()', '', '    total_loss = torch.zeros((), device="cuda")\n') + '    return total_loss\n');
pair('training_retention', 'training', 'train-retention', train(update + '\n        history.append(loss)', 'history = []'), train(update + '\n        history.append(loss.detach())', 'history = []'));
const amp = train(`        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()`, 'scaler = torch.amp.GradScaler("cuda")');
pair('amp_clipping', 'training', 'train-amp', amp.replace('        scaler.unscale_(optimizer)\n', ''), amp);

// Focusing an inner loop must still include the inference decorator and globals.
pair('inference_grad', 'inference', 'infer-grad', predict(predictLoop, ''), predict(predictLoop), 'for x in batches');
pair('inference_mode', 'inference', 'infer-mode', predict(predictLoop).replace('.cuda().eval()', '.cuda().train()'), predict(predictLoop));
pair('inference_device', 'inference', 'infer-device', predict(predictLoop).replace('.cuda().eval()', '.cpu().eval()'), predict(predictLoop));
pair('inference_transfers', 'inference', 'infer-transfers', predict(`    outputs = []
    for x in batches:
        y = model(x.cuda()).cpu().numpy()
        outputs.append(model(torch.from_numpy(y).cuda()))
    return outputs`), predict(`    outputs = []
    for x in batches:
        y = model(x.cuda())
        outputs.append(model(y))
    return outputs`));
const growingOutput = predict(`    x, steps = batches
    outputs = torch.empty((steps, *x.shape), device=x.device)
    for step in range(steps):
        outputs[step] = model(x)
    return outputs`);
pair('inference_storage', 'inference', 'infer-cache', growingOutput.replace('outputs = torch.empty((steps, *x.shape), device=x.device)', 'outputs = torch.empty((0, *x.shape), device=x.device)').replace('outputs[step] = model(x)', 'outputs = torch.cat([outputs, model(x).unsqueeze(0)], dim=0)'), growingOutput);
const decoding = `import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
@torch.inference_mode()
def generate(input_ids, steps):
    past = None
    tokens = input_ids
    outputs = []
    for _ in range(steps):
        result = model(input_ids=tokens, past_key_values=past, use_cache=True)
        past = result.past_key_values
        tokens = result.logits[:, -1:].argmax(dim=-1)
        outputs.append(tokens)
    return torch.cat(outputs, dim=-1)
`;
const recomputing = decoding.replace('    past = None\n', '').replace('input_ids=tokens, past_key_values=past, use_cache=True', 'input_ids=tokens, use_cache=False').replace('        past = result.past_key_values\n', '').replace('        tokens = result.logits[:, -1:].argmax(dim=-1)', '        next_token = result.logits[:, -1:].argmax(dim=-1)\n        tokens = torch.cat([tokens, next_token], dim=-1)').replace('outputs.append(tokens)', 'outputs.append(next_token)');
pair('decode_cache', 'inference', 'infer-kv-cache', recomputing, decoding);
const padding = `import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()
@torch.inference_mode()
def predict(texts):
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    return model(**batch)
`;
pair('padding_mask', 'inference', 'infer-padding', padding.replace('model(**batch)', 'model(input_ids=batch["input_ids"])'), padding);
const causal = `import torch
import torch.nn.functional as F
@torch.inference_mode()
def predict_causal_prefill(q, k, v):
    """Multi-token prefill of an autoregressive causal decoder."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
`;
pair('causal_attention', 'inference', 'infer-causal-mask', causal.replace('is_causal=True', 'is_causal=False'), causal);
pair('attention_dropout', 'inference', 'infer-attention-dropout', causal.replace('dropout_p=0.0', 'dropout_p=0.2'), causal);
const attention = `import torch
import torch.nn.functional as F
@torch.inference_mode()
def predict_encoder_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
`;
pair('attention_implementation', 'inference', 'infer-attention', attention.replace('    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)', '    scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)\n    probabilities = torch.softmax(scores, dim=-1)\n    return probabilities @ v'), attention);
const setup = `import torch
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased").eval()
@torch.inference_mode()
def predict(batch):
    return model(**batch)
`;
pair('serving_setup', 'inference', 'infer-setup', setup.replace('model = AutoModel.from_pretrained("bert-base-uncased").eval()\n', '').replace('    return model', '    model = AutoModel.from_pretrained("bert-base-uncased").eval()\n    return model'), setup);
const service = `import torch
from collections import deque
class Service:
    def __init__(self):
        self.model = torch.nn.Linear(8, 2).cuda().eval()
        self.history = deque(maxlen=32)
    @torch.inference_mode()
    def predict(self, x):
        result = self.model(x.cuda())
        self.history.append(result)
        return result
`;
pair('serving_retention', 'inference', 'infer-retention', service.replace('deque(maxlen=32)', '[]'), service, 'def predict');
pair('allocator_flush', 'training', 'tensor-allocator', train(update + '\n        torch.cuda.empty_cache()'), train(update) + '    torch.cuda.empty_cache()\n');
pair('tensor_batching', 'inference', 'tensor-batching', predict(`    x = batches.cuda()
    outputs = []
    for row in x:
        outputs.append(model(row.unsqueeze(0)))
    return torch.cat(outputs, dim=0)`), predict('    return model(batches.cuda())'));
const compiled = `import torch
model = torch.nn.Linear(8, 2).cuda().eval()
@torch.compile
@torch.inference_mode()
def predict(x):
    y = model(x)
    return torch.where(y.sum() > 0, y * 2, y)
`;
pair('compiled_control', 'inference', 'tensor-compile', compiled.replace('    return torch.where(y.sum() > 0, y * 2, y)', '    if y.sum().item() > 0:\n        return y * 2\n    return y'), compiled);
const normalizing = `import torch
def normalize(x):
    return torch.softmax(x, dim=-1)
`;
pair('normalization', 'pytorch', 'tensor-numerics', normalizing.replace('    return torch.softmax(x, dim=-1)', '    values = torch.exp(x)\n    return values / values.sum(dim=-1, keepdim=True)'), normalizing);
const layout = `import torch
def transform():
    x = torch.randn(4, 8, device="cuda")
    return x.transpose(0, 1).reshape(32)
`;
pair('tensor_layout', 'pytorch', 'tensor-layout', layout.replace('.reshape(32)', '.view(32)'), layout);
const transfer = train(update, `dataset = torch.utils.data.TensorDataset(torch.randn(64, 8), torch.randint(2, (64,)))
loader = torch.utils.data.DataLoader(dataset, batch_size=8, pin_memory=True)`).replace('def train(loader):', 'def train():').replace('x.cuda(), y.cuda()', 'x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)');
pair('batch_transfer', 'training', 'tensor-transfer', transfer.replace('non_blocking=True', 'non_blocking=False').replace('non_blocking=True', 'non_blocking=False'), transfer);

const preprocessing = `import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").cuda().eval()
def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
@torch.inference_mode()
def predict(texts):
    loader = DataLoader(texts, batch_size=32, collate_fn=tokenize, num_workers=4, pin_memory=True)
    for batch in loader:
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        yield model(**batch)
`;
pair('cpu_preprocessing', 'inference', 'data-preprocess', preprocessing
  .replace('    loader = DataLoader(texts, batch_size=32, collate_fn=tokenize, num_workers=4, pin_memory=True)\n    for batch in loader:',
    '    for start in range(0, len(texts), 32):\n        batch = tokenizer(texts[start:start + 32], padding=True, truncation=True, return_tensors="pt")'),
  preprocessing, 'def predict');

const callerUnknown = `import torch
def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x.cuda()), y.cuda())
    loss.backward()
    optimizer.step()
`;
cases.push({ name: 'audit_unknown_model_device', file: 'step.py', source: callerUnknown, cursorLine: 1, pack: 'training', expect: { 'train-device': ['unknown'] } });
const teacher = train(update.replace('loss = F.cross_entropy(logits, y)', 'loss = F.mse_loss(logits, teacher(x).detach())'), 'teacher = nn.Linear(8, 2).cuda().eval()');
cases.push({ name: 'audit_frozen_teacher_control', file: 'distill.py', source: teacher, cursorLine: at(teacher), pack: 'training', expect: { 'train-graph': ['supported'] } });
const softTargets = train(update.replace('F.cross_entropy(logits, y)', 'F.cross_entropy(logits, torch.softmax(torch.randn_like(logits), dim=-1))'));
cases.push({ name: 'audit_soft_target_control', file: 'train.py', source: softTargets, cursorLine: at(softTargets), pack: 'training', expect: { 'train-loss': ['supported'] } });
const below = `import torch
@torch.inference_mode()
def predict(batches):
    for x in batches:
        yield model(x.cuda())
model = torch.nn.Sequential(torch.nn.Linear(8, 2), torch.nn.Dropout(0.5)).cuda().eval()
`;
cases.push({ name: 'audit_setup_below_function', file: 'predict.py', source: below, cursorLine: 4, pack: 'inference', expect: { 'infer-grad': ['supported'], 'infer-device': ['supported'], 'infer-mode': ['supported'] } });
export const workflowCases = cases;

const cleanTrainer = `import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
model = nn.Linear(8, 2).cuda()
optimizer = torch.optim.AdamW(model.parameters())
dataset = TensorDataset(torch.randn(64, 8), torch.randint(2, (64,)))
loader = DataLoader(dataset, batch_size=8, pin_memory=True)
def train():
    model.train()
    for x, y in loader:
        x = x.to("cuda", non_blocking=True)
        y = y.to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
`;
cases.push({ name: 'audit_clean_training_control', file: 'trainer.py', source: cleanTrainer, cursorLine: at(cleanTrainer), pack: 'training',
  expect: { 'train-grad-reset': ['supported'], 'train-graph': ['supported'], 'train-loss': ['supported'], 'tensor-transfer': ['supported'] }, noStrongConcerns: true });
const cleanInference = `import torch
model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.5)).cuda().eval()
@torch.inference_mode()
def predict(x):
    return model(x)
def caller():
    x = torch.randn(16, 8, device="cuda")
    return predict(x)
`;
cases.push({ name: 'audit_clean_inference_control', file: 'service.py', source: cleanInference, cursorLine: at(cleanInference), pack: 'inference',
  expect: { 'infer-grad': ['supported'], 'infer-mode': ['supported'], 'infer-setup': ['supported'] }, noStrongConcerns: true });
const aliases = train(update.replace('loss.backward()', 'loss.detach().requires_grad_().backward()'))
  .replaceAll('import torch\n', 'import torch as pt\n').replaceAll('torch.', 'pt.').replaceAll('optimizer', 'opt').replaceAll('model', 'network').replaceAll('loss', 'objective').replace('def train(', 'def fit(');
cases.push({ name: 'audit_renamed_graph_break', file: 'fit.py', source: aliases, cursorLine: at(aliases), pack: 'training', expect: { 'train-graph': ['concern'] } });
const parentDecorator = `import torch
model = torch.nn.Linear(8, 2).cuda().eval()
def inference_impl(x):
    return model(x.cuda())
@torch.inference_mode()
def predict(x):
    return inference_impl(x)
`;
cases.push({ name: 'audit_decorated_caller_control', file: 'helpers.py', source: parentDecorator, cursorLine: 2, pack: 'inference', expect: { 'infer-grad': ['supported'] } });
