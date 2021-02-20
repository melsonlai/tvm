from .error import *


def add_operation(converter, inputs, outputs): 
    """Add an ANEURALNETWORKS_GROUPED_CONV_2D operation with checking

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
    ANC_NNAPI_OP_CHECK(len(inputs) == 12)
    ins = [ {} for i in range(len(inputs)) ]

    # check inputs[0]
    ins[0] = {}
    ins[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[0])
    ANC_NNAPI_OP_CHECK(ins[0]["dtype"] == "TENSOR_FLOAT32" or ins[0]["dtype"] == "TENSOR_FLOAT16")
    ins[0]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[0])
    ANC_NNAPI_OP_CHECK(ins[0]["rank"] == 4)
    ins[0]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[0])

    # check inputs[1]
    ins[1] = {}
    ins[1]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[1])
    ANC_NNAPI_OP_CHECK(ins[1]["dtype"] == "TENSOR_FLOAT32" or ins[1]["dtype"] == "TENSOR_FLOAT16")
    ANC_NNAPI_OP_CHECK(ins[1]["dtype"] == ins[0]["dtype"])
    ins[1]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[1])
    ANC_NNAPI_OP_CHECK(ins[1]["rank"] == 4)
    ins[1]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[1])
    felter = { k: v for (k, v) in zip([ "do", "fh", "fw", "dg" ], ins[1]["shape"]) }
    
    # check inputs[2]
    ins[2] = {}
    ins[2]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[2])
    ANC_NNAPI_OP_CHECK(ins[2]["dtype"] == ins[1]["dtype"] and ins[2]["dtype"] == ins[0]["dtype"])
    ins[2]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[2])
    ANC_NNAPI_OP_CHECK(ins[2]["rank"] == 1)
    ins[2]["constant"] = converter.export_obj.helper.operand.get_constant(inputs[2])
    ANC_NNAPI_OP_CHECK(ins[2]["constant"]["type"] == "array" and len(ins[2]["constant"]["value"]) == felter["do"])

    # check inputs[3]
    ins[3] = {}
    ins[3]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[3])
    ANC_NNAPI_OP_CHECK(ins[3]["dtype"] == "INT32")
    ins[3]["value"] = converter.export_obj.helper.operand.get_value(inputs[3])
    ANC_NNAPI_OP_CHECK(ins[3]["value"] >= 0)
    padding = {}
    padding["l"] = ins[3]["value"]

    # check inputs[4]
    ins[4] = {}
    ins[4]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[4])
    ANC_NNAPI_OP_CHECK(ins[4]["dtype"] == "INT32")
    ins[4]["value"] = converter.export_obj.helper.operand.get_value(inputs[4])
    ANC_NNAPI_OP_CHECK(ins[4]["value"] >= 0)
    padding["r"] = ins[4]["value"]

    # check inputs[5]
    ins[5] = {}
    ins[5]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[5])
    ANC_NNAPI_OP_CHECK(ins[5]["dtype"] == "INT32")
    ins[5]["value"] = converter.export_obj.helper.operand.get_value(inputs[5])
    ANC_NNAPI_OP_CHECK(ins[5]["value"] >= 0)
    padding["t"] = ins[5]["value"]

    # check inputs[6]
    ins[6] = {}
    ins[6]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[6])
    ANC_NNAPI_OP_CHECK(ins[6]["dtype"] == "INT32")
    ins[6]["value"] = converter.export_obj.helper.operand.get_value(inputs[6])
    ANC_NNAPI_OP_CHECK(ins[6]["value"] >= 0)
    padding["b"] = ins[6]["value"]

    # check inputs[7]
    ins[7] = {}
    ins[7]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[7])
    ANC_NNAPI_OP_CHECK(ins[7]["dtype"] == "INT32")
    ins[7]["value"] = converter.export_obj.helper.operand.get_value(inputs[7])
    ANC_NNAPI_OP_CHECK(ins[7]["value"] >= 0)
    stride = {}
    stride["w"] = ins[7]["value"]

    # check inputs[8]
    ins[8] = {}
    ins[8]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[8])
    ANC_NNAPI_OP_CHECK(ins[8]["dtype"] == "INT32")
    ins[8]["value"] = converter.export_obj.helper.operand.get_value(inputs[8])
    ANC_NNAPI_OP_CHECK(ins[8]["value"] >= 0)
    stride["h"] = ins[8]["value"]

    # check inputs[9]
    ins[9] = {}
    ins[9]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[9])
    ANC_NNAPI_OP_CHECK(ins[9]["dtype"] == "INT32")
    ins[9]["value"] = converter.export_obj.helper.operand.get_value(inputs[9])
    num_groups = ins[9]["value"]
    ANC_NNAPI_OP_CHECK(num_groups >= 0)
    ANC_NNAPI_OP_CHECK(felter["do"] % num_groups == 0)

    # check inputs[10]
    ANC_NNAPI_OP_CHECK(converter.export_obj.helper.operand.is_FuseCode(inputs[10]))

    # check inputs[11]
    ins[11] = {}
    ins[11]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[11])
    ANC_NNAPI_OP_CHECK(ins[11]["dtype"] == "BOOL")
    ins[11]["value"] = converter.export_obj.helper.operand.get_value(inputs[11])
    ANC_NNAPI_OP_CHECK(ins[11]["value"] == "false" or ins[11]["value"] == "true")

    # check shapes
    if (API_LEVEL >= 29 and ins[11]["value"] == "true"): 
        data_shape = {
                "n": ins[0]["shape"][0], 
                "c": ins[0]["shape"][1], 
                "h": ins[0]["shape"][2], 
                "w": ins[0]["shape"][3], 
                }
    else: 
        data_shape = {
                "n": ins[0]["shape"][0], 
                "h": ins[0]["shape"][1], 
                "w": ins[0]["shape"][2], 
                "c": ins[0]["shape"][3], 
                }

    ANC_NNAPI_OP_CHECK(data_shape["c"] == num_groups * felter["dg"])

    # check outputs
    ANC_NNAPI_OP_CHECK(len(outputs) == 1)
    outs = [ {} ]

    # check outputs[0]
    outs[0] = {}
    outs[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(outputs[0])
    ANC_NNAPI_OP_CHECK(outs[0]["dtype"] == ins[0]["dtype"] and outs[0]["dtype"] == ins[1]["dtype"])
    outs[0]["shape"] = converter.export_obj.helper.operand.get_shape(outputs[0])

    if (API_LEVEL >= 29 and ins[11]["value"] == "true"): 
        out_data_shape = {
                "n": outs[0]["shape"][0], 
                "c": outs[0]["shape"][1], 
                "h": outs[0]["shape"][2], 
                "w": outs[0]["shape"][3], 
                }
    else: 
        out_data_shape = {
                "n": outs[0]["shape"][0], 
                "h": outs[0]["shape"][1], 
                "w": outs[0]["shape"][2], 
                "c": outs[0]["shape"][3], 
                }
    total_h = data_shape["h"] + padding["t"] + padding["b"]
    total_w = data_shape["w"] + padding["l"] + padding["r"]
    ANC_NNAPI_OP_CHECK(out_data_shape["n"] == data_shape["n"])
    ANC_NNAPI_OP_CHECK(out_data_shape["h"] == ((total_h - felter["fh"]) // stride["h"] + 1))
    ANC_NNAPI_OP_CHECK(out_data_shape["w"] == ((total_w - felter["fw"]) // stride["w"] + 1))
    ANC_NNAPI_OP_CHECK(out_data_shape["c"] == felter["do"])

    converter.export_obj.add_operation("GROUPED_CONV_2D", inputs, outputs)



