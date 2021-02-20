from .error import *


def add_operation(converter, inputs, outputs): 
    """Add an ANEURALNETWORKS_CAST operation with checking

    Parameters
    ----------
    converter: FunctionToJsonConverter
        the converter object holding export_obj

    inputs: list of int
        inputs to the operation

    outputs: list of int
        outputs of the operation

    """
    API_LEVEL = converter.options["target"]["api_level"]
    ANC_COMPATIBILITY_CHECK(API_LEVEL >= 29, f"Target Android API level { API_LEVEL } is too low to support the operation")
    
    # check inputs
    ANC_NNAPI_OP_CHECK(len(inputs) == 1)
    ins = [ {} ]

    # check inputs[0]
    ins[0] = {}
    ins[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[0])
    ANC_NNAPI_OP_CHECK(ins[0]["dtype"] == "TENSOR_FLOAT16" \
            or ins[0]["dtype"] == "TENSOR_FLOAT32" \
            or ins[0]["dtype"] == "TENSOR_INT32")
    ins[0]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[0])

    # check outputs
    ANC_NNAPI_OP_CHECK(len(outputs) == 1)
    outs = [ {} ]

    # check outputs[0]
    outs[0] = {}
    outs[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(outputs[0])
    ANC_NNAPI_OP_CHECK(outs[0]["dtype"] == "TENSOR_FLOAT16" \
            or outs[0]["dtype"] == "TENSOR_FLOAT32" \
            or outs[0]["dtype"] == "TENSOR_INT32")
    outs[0]["shape"] = converter.export_obj.helper.operand.get_shape(outputs[0])
    ANC_NNAPI_OP_CHECK(outs[0]["shape"] == ins[0]["shape"])

    converter.export_obj.add_operation("CAST", inputs, outputs)



