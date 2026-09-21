import type { SystemsOp } from './types';
export interface ApiSemantics { kind: SystemsOp['kind']; op: SystemsOp['op']; tensor?: boolean; outputArg?: number; inputArg?: number }
const table: Record<string, ApiSemantics> = {};
const put = (names: string[], kind: SystemsOp['kind'], op: SystemsOp['op'], extra: Partial<ApiSemantics> = {}) => names.forEach(n => { table[n] = { kind, op, ...extra }; });
put(['jax.device_get'], 'memory', 'device_to_host');
put(['jax.device_put'], 'memory', 'copy', { tensor: true });
put(['torch.cuda.synchronize'], 'synchronization', 'host_sync');
put(['torch.cuda.current_stream.wait_stream', 'torch.cuda.Stream.wait_stream'], 'synchronization', 'stream_wait');
for (const [suffix, op] of Object.entries({ all_reduce: 'all_reduce', all_gather: 'all_gather', all_gather_into_tensor: 'all_gather', reduce_scatter: 'reduce_scatter', reduce_scatter_tensor: 'reduce_scatter', all_to_all: 'all_to_all', all_to_all_single: 'all_to_all', broadcast: 'broadcast', send: 'send', recv: 'recv' }) as [string, SystemsOp['op']][]) {
  put([`torch.distributed.${suffix}`], 'collective', op, { outputArg: 0, inputArg: ['all_gather', 'reduce_scatter', 'all_to_all'].includes(op) ? 1 : 0 });
}
for (const [suffix, op] of Object.entries({ psum: 'all_reduce', pmean: 'all_reduce', all_gather: 'all_gather', psum_scatter: 'reduce_scatter', all_to_all: 'all_to_all', ppermute: 'send', with_sharding_constraint: 'reshard' }) as [string, SystemsOp['op']][]) put([`jax.lax.${suffix}`], 'collective', op, { tensor: true });
put(['jax.jit', 'torch.compile'], 'compilation', 'jit_boundary');
put(['jax.pjit', 'jax.experimental.pjit.pjit'], 'compilation', 'jit_boundary');
put(['torch.compiler.disable', 'torch._dynamo.disable'], 'compilation', 'graph_boundary');
put(['torch.cuda.graph'], 'compilation', 'graph_boundary');
for (const prefix of ['torch', 'jax.numpy']) {
  put(['matmul', 'mm', 'bmm', 'einsum', 'dot'].map(n => `${prefix}.${n}`), 'compute', 'gemm', { tensor: true });
  put(['sum', 'mean', 'max', 'min', 'amax', 'softmax', 'logsumexp'].map(n => `${prefix}.${n}`), 'compute', 'reduction', { tensor: true });
  put(['sin', 'cos', 'exp', 'log', 'sqrt', 'add', 'mul', 'relu', 'tanh', 'sigmoid'].map(n => `${prefix}.${n}`), 'compute', 'elementwise', { tensor: true });
  put(['zeros', 'ones', 'empty', 'randn', 'arange', 'full'].map(n => `${prefix}.${n}`), 'memory', 'allocate', { tensor: true });
  put(['cat', 'concat', 'concatenate', 'stack', 'clone', 'array', 'tensor'].map(n => `${prefix}.${n}`), 'memory', 'materialize', { tensor: true });
}
put(['torch.nn.functional.linear'], 'compute', 'gemm', { tensor: true });
put(['torch.nn.functional.scaled_dot_product_attention'], 'compute', 'attention', { tensor: true });
put(['torch.nn.functional.layer_norm', 'torch.nn.functional.batch_norm', 'jax.nn.standardize'], 'compute', 'normalization', { tensor: true });
put(['torch.nn.functional.conv1d', 'torch.nn.functional.conv2d', 'jax.lax.conv_general_dilated'], 'compute', 'convolution', { tensor: true });
put(['torch.utils.data.DataLoader'], 'scheduling', 'batch');
put(['torch.distributed.init_process_group'], 'scheduling', 'configuration');
put(['vllm.LLM', 'vllm.EngineArgs', 'vllm.AsyncEngineArgs', 'vllm.engine.arg_utils.EngineArgs', 'vllm.engine.arg_utils.AsyncEngineArgs', 'sglang.Engine', 'sglang.srt.server_args.ServerArgs'], 'scheduling', 'configuration');
put(['triton.language.load'], 'memory', 'load', { tensor: true });
put(['triton.language.store'], 'memory', 'store');
put(['triton.language.dot'], 'compute', 'gemm', { tensor: true });
put(['triton.language.sum', 'triton.language.max'], 'compute', 'reduction', { tensor: true });
put(['triton.language.arange', 'triton.language.full'], 'memory', 'allocate', { tensor: true });
put(['triton.jit'], 'compilation', 'jit_boundary');
export const apiSemantics: Readonly<Record<string, ApiSemantics>> = table;
export const lensDimensions = {
  kernel: ['parallelism', 'serialization', 'synchronization', 'global_memory', 'shared_memory', 'register_pressure', 'control_flow_divergence', 'reduction_collectives', 'compute_efficiency', 'occupancy_risk', 'launch_shape_assumptions'],
  graph: ['execution_granularity', 'kernel_launch_count_risk', 'fusion_opportunities', 'host_device_synchronization', 'graph_breaks', 'jit_boundaries', 'recompilation_risk', 'intermediate_materialization', 'memory_lifetime', 'activation_memory', 'sharding', 'collective_placement', 'communication_overlap', 'input_pipeline', 'dtype_and_precision'],
  distributed: ['collective_type', 'collective_placement', 'communication_volume', 'sharding_transitions', 'replication', 'critical_path_communication', 'communication_compute_overlap', 'parallelism_strategy', 'pipeline_bubbles'],
  serving: ['batching', 'continuous_batching', 'prefill_decode_interference', 'chunked_prefill', 'kv_cache_capacity', 'kv_cache_movement', 'prefix_caching', 'scheduler_policy', 'request_queueing', 'tensor_parallelism', 'data_parallelism', 'expert_parallelism', 'collective_critical_path', 'cuda_graph_compatibility', 'kernel_granularity', 'cpu_orchestration', 'pd_disaggregation', 'network_transfer'],
} as const;
