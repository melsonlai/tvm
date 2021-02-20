import re
import functools
import copy
import tvm
import tvm.relay
from .error import *
from .operation_utils import relay_op
from .export_object import ExportObject


class FunctionToJsonConverter(tvm.relay.ExprVisitor): 
    def __init__(self, options): 
        super().__init__()
        self._options = options
        self._export_obj = ExportObject(self._options)

    def __call__(self, func): 
        assert isinstance(func, tvm.relay.Function)
        self.visit(func.body)
        self._export_obj.helper.node_to_operand_idxs_map[func] = copy.deepcopy(self._export_obj.helper.node_to_operand_idxs_map[func.body])

        # identify Android NNAPI model inputs
        for p in func.params: 
            for i in self._export_obj.helper.node_to_operand_idxs_map[p]: # param may be a tuple, which results in multiple indices
                if (i not in self._export_obj["inputs"]): 
                    self._export_obj["inputs"].append(i)

        # identify Android NNAPI model outputs
        for i in self._export_obj.helper.node_to_operand_idxs_map[func]: # again, the output may be a tuple, which results in multiple indices
            if (i not in self._export_obj["outputs"]): 
                self._export_obj["outputs"].append(i)
        assert len(self._export_obj["outputs"]) == 1 # for now, let's force the function to return a single value, i.e. denying tuple as return type

        # set resulting memory for outputs
        for i, op_i in enumerate(self._export_obj["outputs"]): 
            op = self._export_obj["operands"][op_i]
            assert "value" not in op
            op["value"] = {
                    "type": "memory_ptr", 
                    "value": "out".format(i), # no real formatting since len(outs) == 1
                    }

        return self._export_obj

    @property
    def export_obj(self): 
        return self._export_obj

    @property
    def options(self): 
        return self._options

    def visit_function(self, func): 
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.Function not supported")

    def visit_let(self, let): 
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.Let not supported")

    def visit_call(self, call): 
        if (isinstance(call.op, tvm.ir.Op)): 
            op_handler_module = relay_op
            for ns in call.op.name.split("."): # lookup the handler dynamically
                op_handler_module = getattr(op_handler_module, ns, None)
                ANC_COMPATIBILITY_CHECK(op_handler_module != None, f"Relay IR Op { call.op } not implemented")
            op_handler_module.handler(self, call)
        else: 
            raise AndroidNNAPICompilerIncompatibleError(f"Conversion of { call.op.type_key } not supported")

    def visit_var(self, var): 
        self._export_obj.add_operand(
                type_idx=self._export_obj.get_type_idx((var.checked_type.shape, var.checked_type.dtype)), 
                node=var, 
                value={
                    "type": "memory_ptr", 
                    "value": var.name_hint, 
                    }, 
                )

    def visit_type(self, typ):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.Type not supported")

    def visit_if(self, ifs):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.If not supported")

    def visit_tuple(self, tup):
        field_idxs = []
        for f in tup.fields: 
            self.visit(f)
            field_idxs += self._export_obj.helper.node_to_operand_idxs_map[f]
        self._export_obj.helper.node_to_operand_idxs_map[tup] = copy.deepcopy(field_idxs)

    def visit_tuple_getitem(self, tgi):
        self.visit(tgi.tuple_value)
        self._export_obj.helper.node_to_operand_idxs_map[tgi] = [ self._export_obj.helper.node_to_operand_idxs_map[tgi.tuple_value][tgi.index] ]

    def visit_global_var(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.GlobalVar not supported")

    def visit_op(self, _):
        assert False, "Unreachable"

    def visit_constant(self, const):
        ANC_COMPATIBILITY_CHECK(isinstance(const.checked_type, tvm.relay.TensorType), f"Unsupported type { const.checked_type.type_key }")
        shape, dtype = const.data.shape, const.data.dtype
        type_idx = self._export_obj.get_type_idx((shape, dtype))

        if (shape == ()): 
            const_idx = self._export_obj.add_scalar_constant(const.data.asnumpy().item(), dtype)
        elif (type(shape) is tuple): 
            ANC_COMPATIBILITY_CHECK(len(shape) == 1, "Only flat array constants are supported")
            constants = list(map(lambda i: i.item(), const.data.asnumpy()))
            const_idx = self._export_obj.add_array_constant(constants, dtype)
        else: 
            assert False, "Unreachable"

        self._export_obj.add_operand(
                type_idx=type_idx, 
                value={
                    "type": "constant_idx", 
                    "value": const_idx, 
                    }, 
                node=const)

    def visit_ref_create(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of Relay IR reference not supported")

    def visit_ref_write(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of Relay IR reference not supported")

    def visit_ref_read(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of Relay IR reference not supported")

    def visit_constructor(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of Relay IR ADT not supported")

    def visit_match(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of Relay IR ADT not supported")

