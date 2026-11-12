# ACLNN Torch adapter stream ownership

The native ACLNN adapters allocate their work tensors on the current NPU stream
and pass that same stream to the ACLNN execution function. The generated ACLNN
entry forwards it to `NnopbaseRunWithWorkspace`; choosing ACLNN does not itself
introduce a different execution stream.

The launch callback must capture the owning Tensors and ACL descriptors by value
until the device work has been submitted. Keeping only a `data_ptr()` or an
`aclTensor` descriptor does not retain Tensor storage. In eager execution,
`RunOpApi` can queue that callback on a host thread; the captured device stream
still determines where the work executes. After submission, the allocator's
ordering on the allocation stream protects internal workspace, conversion
buffers and outputs. These adapters therefore need no `record_stream` calls in
either eager execution or graph capture.

During capture, `OpApiContext::run` drains prior host submissions for the current
stream and launches directly. This keeps submission order and releases temporary
Tensor references on the capture thread, allowing graph-pool reuse. It does not
wait for device completion. Recording every temporary during capture would defer
its reuse until capture ends and can make graph memory grow with operator calls.

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
