include "UPMEMBase.td"

def UPMEM_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for UPMEM operations that execute asynchronously on the device.

    UPMEM operations implementing this interface take a list of dependencies
    as `upmem.async.token` arguments and optionally return a `upmem.async.token`.

    The op doesn't start executing until all depent ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op merely schedules the execution on the
    device and returns immediately, without waiting for the execution to
    complete. On the hand, if the op does not return a token, the op will wait
    for the execution to complete.
  }];
  let cppNamespace = "::mlir::upmem";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAsyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies if it is not already there.
      }],
      "void", "addAsyncDependency", (ins "Value":$token),
      [{}], [{
        if (!::llvm::is_contained(this->getAsyncDependencies(), token))
          ::mlir::upmem::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "Value", "getAsyncToken"
    >
  ];
}

