import tvm.relay


class ConvertScalarToTensorForBroadcastOperators(tvm.relay.ExprMutator): 
    """Convert scalar arguments to a broadcasting operator to its tensor equivalent for Android NNAPI conversion
    """
    def __init__(self): 
        super().__init__()
        self._call_op_stack = []

    def __call__(self, expr): 
        return self.visit(expr)

    def visit_call(self, call): 
        self._call_op_stack.append(call)
        if (self._parent_is_transform_target() and self._is_scalar(call)): 
            assert isinstance(call.op, tvm.ir.Op) and call.op == tvm.relay.op.get("zeros"), "Only tvm.relay.zeros are supported for tvm.relay.Call scalar to tensor transformation"
            self._call_op_stack.pop()
            return tvm.relay.zeros(shape=(1, ), dtype=call.checked_type.dtype)

        ret = super().visit_call(call)
        self._call_op_stack.pop()
        return ret

    def visit_constant(self, const): 
        if (self._parent_is_transform_target() and self._is_scalar(const)): 
            return tvm.relay.Constant(tvm.nd.array(const.data.asnumpy().reshape([1, ])))
        return super().visit_constant(const)

    def visit_var(self, var): 
        assert not self._parent_is_transform_target() or not self._is_scalar(var), "Transforming variable scalar is not supported" # due to the need to also transform the parameter dict
        return super().visit_var(var)

    def _parent_is_transform_target(self): 
        if (len(self._call_op_stack) == 0): 
            return False

        last_call = self._call_op_stack[-1]
        if (not isinstance(last_call, tvm.ir.Op)): 
            return False

        return last_call.op in [ tvm.relay.op.get(name) for name in "add", "subtract", "multiply", "divide" ] # only these ops are supported for the fix for now

    def _is_scalar(self, node): 
        return len(node.checked_type.shape) == 0

