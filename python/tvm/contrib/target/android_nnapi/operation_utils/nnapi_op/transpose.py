from .error import *


def add_operation(converter, inputs, outputs): 
    """ Add an ANEURALNETWORKS_TRANSPOSE operation with checking

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
    ANC_COMPATIBILITY_CHECK(API_LEVEL >= 28, f"Target Android API level { API_LEVEL } is too low to support the operation")

    # check inputs
    ANC_NNAPI_OP_CHECK(len(inputs) == 2)
    ins = [ {}, {} ]

    # check inputs[0]
    ins[0] = {}
    ins[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[0])
    if (ins[0]["dtype"] == "TENSOR_FLOAT16"): 
        ANC_NNAPI_OP_CHECK(API_LEVEL >= 29)
    else: 
        ANC_NNAPI_OP_CHECK(ins[0]["dtype"] == "TENSOR_FLOAT32")
    ins[0]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[0])
    ins[0]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[0])
    ANC_NNAPI_OP_CHECK(ins[0]["rank"] <= 4)

    # check inputs[1]
    ins[1] = {}
    ins[1]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[1])
    ANC_NNAPI_OP_CHECK(ins[1]["dtype"] == "TENSOR_INT32")
    ins[1]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[1])
    ANC_NNAPI_OP_CHECK(ins[1]["rank"] == 1)
    ins[1]["constant"] = converter.export_obj.helper.operand.get_constant(inputs[1])
    ANC_NNAPI_OP_CHECK(ins[1]["constant"]["type"] == "array" and len(ins[1]["constant"]["value"]) == ins[0]["rank"])
    ins[1]["value"] = converter.export_obj.helper.operand.get_value(inputs[1])

    # check outputs
    ANC_NNAPI_OP_CHECK(len(outputs) == 1)
    outs = [ {} ]

    # check outputs[0]
    outs[0] = {}
    outs[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(outputs[0])
    ANC_NNAPI_OP_CHECK(outs[0]["dtype"] == ins[0]["dtype"])
    outs[0]["shape"] = converter.export_obj.helper.operand.get_shape(outputs[0])
    ANC_NNAPI_OP_CHECK(outs[0]["shape"] == [ ins[0]["shape"][i] for i in ins[1]["value"] ])

    converter.export_obj.add_operation("TRANSPOSE", inputs, outputs)



