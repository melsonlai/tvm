import tvm.relay
from ....r2n_assert import *
from .convert_scalar_to_tensor_for_broadcast_operators import ConvertScalarToTensorForBroadcastOperators

class FixIllegalPatternForNnapi: 
    def __call__(self, func): 
        R2N_CHECK(isinstance(func, tvm.relay.Function))
        passes = [ ConvertScalarToTensorForBroadcastOperators() ]
        func = tvm.relay.transform.InferType()(tvm.IRModule({ "main": func }))["main"]
        for p in passes: 
            func = p(func)
            func = tvm.relay.transform.InferType()(tvm.IRModule({ "main": func }))["main"]
        return func

