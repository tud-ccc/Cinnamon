


# Lowering strategy from BTFL to upmem

## Dpu-side lowering

Preparation:
```mlir
  transform.btfl.lower.create_accelerator_scopes %block
  transform.btfl.lower.prepare_thread_assignment %block
  transform.btfl.lower.do_threads_assignment %block
  transform.btfl.lower.create_scoped_kernels %block
```
This creates scopes and assigns threads.
The rest of the lowering is performed top-down.
- Convert `btfl.block` into a `func.func`
- Walk the block and call the lowerTopDown on each operation.
These add value bindings to the scope. 

There is an ordering problem though.
- We need the buffer symbols to lower the transfers.
- The symbols are assigned by the `btfl.scope` -> `upmem.dpu_program` part.

Maybe what we should do is:
- Lower the accelerators into the func to map the `%upmem: !tilefirst.accelerator_ref` to an `!upmem.workgroup` value
- For the (unique) `btfl.scope` of this accelerator, outline the `upmem.dpu_program`. 
Map the shared `!tilefirst.buffer`s to SymbolRefAttr in the TopDownLoweringContext.

Let the top-down processing continue:
- empty_buf on host are mapped to with memref.alloc
- empty_buf on mram/wram are mapped to nothing, but we check they have a SymbolRefAttr in the map?
- transfer host->mram/wram is mapped to scatter using the symbolrefattr
- scope is mapped to wait_for
- transfer mram/wram->host is mapped to gather using the symbolrefattr















