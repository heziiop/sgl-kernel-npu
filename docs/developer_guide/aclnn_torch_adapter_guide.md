# ACLNN Torch adapter stream ownership

The native ACLNN adapters allocate their work tensors on the current NPU stream
and pass that same stream to the ACLNN execution function. The generated ACLNN
entry forwards it to `NnopbaseRunWithWorkspace`; choosing ACLNN does not itself
introduce a different execution stream.

In eager execution, `RunOpApi` can queue the launch callback on a host thread.
That does not introduce a new device stream: device launches and later work on
the same stream must retain FIFO submission order. The caching allocator can
reuse released storage on its allocation stream under that ordering, even
while earlier device work is pending. Internal workspace and conversion
buffers allocated and used on this stream therefore need neither callback
Tensor captures nor `record_stream` calls. Retaining those Tensors in queued
callbacks unnecessarily extends their lifetime and increases peak memory.

The callback captures workspace as a raw pointer and retains ACL descriptors
and host-side attribute owners needed by the queued execution. These objects
do not retain Tensor storage. This rule relies on FIFO host submission as well
as device-stream order; it does not apply to arbitrary host-side access to
released memory or to storage allocated on another stream.

During capture, `OpApiContext::run` drains prior host submissions for the current
stream and launches directly. This preserves order relative to earlier queued
work. It does not wait for device completion. Tensor storage is not captured by
the callback. Recording every temporary during capture can defer its reuse until
capture ends and make graph memory grow with operator calls.

Callers own dependencies and storage lifetimes when passing a Tensor to another
stream, including caller-provided outputs and mutable state. For example:

```python
consumer.wait_stream(producer)  # Input writes must precede the consumer's reads.
with torch.npu.stream(consumer):
    batch_matmul_transpose(x, b, c, impl="ascendc")
x.record_stream(consumer)  # Prevent reuse on x's allocation stream while in use.
```

The caller must provide the same protection for any other argument crossing
streams, or keep its storage alive until the consuming work completes. Waiting
for input readiness and protecting storage lifetime are distinct requirements.
External graph inputs/outputs also need stable storage for future replays;
`record_stream` during capture alone does not provide that replay lifetime.

If an adapter later introduces an internal side stream, it must manage the
dependencies and lifetimes it creates, for example by joining that stream before
the storage can be reused or by recording its use. The current single-stream
argument would no longer cover that implementation. This guide does not add
capture support to operators with other capture restrictions.
