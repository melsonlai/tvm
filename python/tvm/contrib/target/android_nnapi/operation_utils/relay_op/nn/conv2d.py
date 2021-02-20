from ....error import *
from ... import utils
from ... import nnapi_op


def handler(converter, node): 
    """Handler for tvm.relay.nn.conv2d

    Parameters
    ----------
    converter: FunctionToJsonConverter
        the converter object holding export_obj

    node: relay.Call
        operation call node

    """
    args = utils.name_args(node.args, [ "data", "weight" ])
    attrs = node.attrs
    ngroups = int(attrs.groups)
    C_dim = int(args["data"].checked_type.shape[attrs.data_layout.index("C")])
    O_dim = int(args["weight"].checked_type.shape[attrs.kernel_layout.index("O")])
    I_dim = int(args["weight"].checked_type.shape[attrs.kernel_layout.index("I")])
    if (ngroups == 1): # classic convolution
        _1_group_handler(converter, node)
    elif (ngroups == C_dim and C_dim == O_dim and I_dim == 1): 
        _depthwise_handler(converter, node)
    else: 
        _grouped_handler(converter, node)


def _1_group_handler(converter, node): 
    API_LEVEL = converter.options["target"]["api_level"]
    args = utils.name_args(node.args, [ "data", "weight" ])
    attrs = node.attrs
    nnapi = {}

    # START: handle inputs
    # use explicit padding of ANEURALNETWORKS_CONV_2D
    nnapi["inputs"] = []

    # START: handle input[0]
    # check compatibility
    ANC_COMPATIBILITY_CHECK(args["data"].checked_type.dtype == "float32" or args["data"].checked_type.dtype == "float16")

    # generate nnapi node of "data"
    converter.visit(args["data"])

    # change layout of "data" to NNAPI's NHWC
    ANC_COMPATIBILITY_CHECK(len(attrs.data_layout) == 4, f"Unrecognized layout { attrs.data_layout }")
    if (attrs.data_layout == "NHWC" or (API_LEVEL >= 29 and attrs.data_layout == "NCHW")): 
        nnapi["inputs"] += converter.export_obj.helper.node_to_operand_idxs_map[args["data"]]
    else: 
        # START: add TRANSPOSE
        transpose_idxs = list(map(lambda ele: attrs.data_layout.index(ele), [ "N", "H", "W", "C" ]))
        inputs = []
        inputs += converter.export_obj.helper.node_to_operand_idxs_map[args["data"]]
        inputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_array_constant(
                        vals=transpose_idxs, 
                        dtype="int32", 
                        ), 
                    }, 
                )
        outputs = []
        outputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx((
                    tuple(map(lambda ele: args["data"].checked_type.shape[ele], transpose_idxs)), 
                    args["data"].checked_type.dtype))
                )
        nnapi_op.transpose.add_operation(converter, inputs, outputs)
        nnapi["inputs"] += outputs
        # END: add TRANSPOSE
    # END: handle input[0]

    # START: handle input[1]
    # check compatibility
    ANC_COMPATIBILITY_CHECK(args["weight"].checked_type.dtype == args["data"].checked_type.dtype)

    # generate nnapi node for weight
    converter.visit(args["weight"])

    # change layout of "weight" to NNAPI's OHWI
    ANC_COMPATIBILITY_CHECK(len(attrs.kernel_layout) == 4, f"Unrecognized layout { attrs.kernel_layout }")
    if (attrs.kernel_layout == "OHWI"): 
        nnapi["inputs"] += converter.export_obj.helper.node_to_operand_idxs_map[args["weight"]]
    else: 
        # START: add TRANSPOSE
        transpose_idxs = list(map(lambda ele: attrs.kernel_layout.index(ele), [ "O", "H", "W", "I" ]))
        inputs = []
        inputs += converter.export_obj.helper.node_to_operand_idxs_map[args["weight"]]
        inputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_array_constant(
                        vals=transpose_idxs, 
                        dtype="int32", 
                        ), 
                    }, 
                )
        outputs = []
        outputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx((
                    tuple(map(lambda ele: args["weight"].checked_type.shape[ele], transpose_idxs)), 
                    args["weight"].checked_type.dtype))
                )
        nnapi_op.transpose.add_operation(converter, inputs, outputs)
        nnapi["inputs"] += outputs
        # END: add TRANSPOSE
    # END: handle input[1]

    # START: handle input[2]
    # add empty bias since CONV_2D needs it
    bias_shape = (converter.export_obj.helper.operand.get_shape(nnapi["inputs"][1])[0], )
    if (args["data"].checked_type.dtype == "float32" or args["data"].checked_type.dtype == "float16"): 
        bias_dtype = args["data"].checked_type.dtype
    else: 
        raise AndroidNNAPICompilerIncompatibleError(f"Unable to determine bias data type for CONV_2D. args['data'].dtype was { args['data'].checked_type.dtype }")
    bias_type = (bias_shape, bias_dtype)
    nnapi["inputs"] += converter.export_obj.add_operand(
            type_idx=converter.export_obj.get_type_idx(bias_type), 
            value={
                "type": "constant_idx", 
                "value": converter.export_obj.add_array_constant(
                    vals=[ 0.0 ] * bias_shape[0], 
                    dtype=bias_dtype, 
                    ), 
                }, 
            )
    # END: handle input[2]

    # START: handle input[3:7]
    def _add_int32_scalar_constant(e): 
        return converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_scalar_constant(
                        val=int(e), 
                        dtype="int32", 
                        ), 
                    }, 
                )[0]
    relay_paddings = list(map(_add_int32_scalar_constant, attrs.padding))
    if (len(relay_paddings) == 2): 
        nnapi["inputs"] += [ relay_paddings[1], relay_paddings[1], relay_paddings[0], relay_paddings[0] ]
    elif (len(relay_paddings) == 4): 
        nnapi["inputs"] += [ relay_paddings[1], relay_paddings[3], relay_paddings[0], relay_paddings[2] ]
    else: 
        raise AndroidNNAPICompilerIncompatibleError(f"Unexpected padding format { attrs.padding }")
    # END: handle input[3:7]

    # START: handle input[7:9]
    relay_strides = list(map(_add_int32_scalar_constant, attrs.strides))
    nnapi["inputs"] += [ relay_strides[1], relay_strides[0] ]
    # END: handle input[7:9]

    # START: handle input[9]
    # add ANEURALNETWORKS_FUSED_NONE activation since CONV_2D needs it
    nnapi["inputs"] += converter.export_obj.add_operand(
            type_idx=converter.export_obj.get_type_idx(((), "int32")), 
            value={
                "type": "constant_idx", 
                "value": converter.export_obj.add_scalar_constant(
                    val="ANEURALNETWORKS_FUSED_NONE", 
                    dtype="int32", 
                    ), 
                }, 
            )
    # END: handle input[9]

    nnapi_output_layout = "NHWC"
    if (API_LEVEL >= 29): 
        # START: handle input[10]
        if (attrs.data_layout == "NCHW"): 
            nnapi["inputs"] += converter.export_obj.add_operand(
                    type_idx=converter.export_obj.get_type_idx(((), "bool")), 
                    value={
                        "type": "constant_idx", 
                        "value": converter.export_obj.add_scalar_constant(
                            val="true", 
                            dtype="bool", 
                            ), 
                        }, 
                    )
            nnapi_output_layout = "NCHW"
        else: 
            nnapi["inputs"] += converter.export_obj.add_operand(
                    type_idx=converter.export_obj.get_type_idx(((), "bool")), 
                    value={
                        "type": "constant_idx", 
                        "value": converter.export_obj.add_scalar_constant(
                            val="false", 
                            dtype="bool", 
                            ), 
                        }, 
                    )
        # END: handle input[10]

        # START: handle input[11:]
        # unpack dilation
        relay_dilations = list(map(_add_int32_scalar_constant, attrs.dilation))
        nnapi["inputs"] += [ relay_dilations[1], relay_dilations[0] ]
        # END: handle input[11:]
    # END: handle inputs

    # START: handle outputs
    nnapi["outputs"] = []

    # START: handle output[0]
    attrs_out_layout = attrs.data_layout if attrs.out_layout == "" else attrs.out_layout
    attrs_out_dtype = args["data"].checked_type.dtype if attrs.out_dtype == "" else attrs.out_dtype
    if (attrs_out_dtype == args["data"].checked_type.dtype and attrs_out_layout == nnapi_output_layout): 
        nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, node.checked_type.dtype)))
        node_operands = nnapi["outputs"]
    else: 
        if (attrs_out_layout == nnapi_output_layout): 
            nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, args["data"].checked_type.dtype)))
            last_outputs = nnapi["outputs"]
        else: 
            transpose_idxs = list(map(lambda ele: attrs_out_layout.index(ele), [ "N", "H", "W", "C" ]))
            NHWC_shape = tuple(map(lambda ele: node.checked_type.shape[ele], transpose_idxs))
            nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((NHWC_shape, args["data"].checked_type.dtype)))

            # START: add TRANSPOSE
            rev_transpose_idxs = list(map(lambda ele: "NHWC".index(ele), attrs_out_layout))
            inputs = []
            inputs += nnapi["outputs"]
            inputs += converter.export_obj.add_operand(
                    type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                    value={
                        "type": "constant_idx", 
                        "value": converter.export_obj.add_array_constant(
                            vals=rev_transpose_idxs, 
                            dtype="int32", 
                            ), 
                        }, 
                    )
            outputs = []
            outputs += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, args["data"].checked_type.dtype)))
            nnapi_op.transpose.add_operation(converter, inputs, outputs)
            # END: add TRANSPOSE

            last_outputs = outputs

        if (attrs_out_dtype == args["data"].checked_type.dtype): 
            node_operands = last_outputs
        else: 
            # START: add CAST
            inputs = []
            inputs += last_outputs
            outputs = []
            outputs += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, node.checked_type.dtype)))
            nnapi_op.cast.add_operation(converter, inputs, outputs)
            # END: add CAST
        
            node_operands = outputs

    # register operands to node
    converter.export_obj.helper.node_to_operand_idxs_map[node] = node_operands
    # END: handle output[0]
    # END: handle outputs

    nnapi_op.conv_2d.add_operation(converter, nnapi["inputs"], nnapi["outputs"])
    

def _depthwise_handler(converter, node): 
    API_LEVEL = converter.options["target"]["api_level"]
    args = utils.name_args(node.args, [ "data", "weight" ])
    attrs = node.attrs
    nnapi = {}

    # START: handle inputs
    # use explicit padding
    nnapi["inputs"] = []

    # START: handle input[0]
    # generate nnapi node of "data"
    converter.visit(args["data"])

    # change layout of "data" to NNAPI's NHWC
    ANC_COMPATIBILITY_CHECK(len(attrs.data_layout) == 4, f"Unrecognized layout { attrs.data_layout }")
    if (attrs.data_layout == "NHWC" or (API_LEVEL >= 29 and attrs.data_layout == "NCHW")): 
        nnapi["inputs"] += converter.export_obj.helper.node_to_operand_idxs_map[args["data"]]
    else: 
        # START: add TRANSPOSE
        transpose_idxs = list(map(lambda ele: attrs.data_layout.index(ele), [ "N", "H", "W", "C" ]))
        inputs = []
        inputs += converter.export_obj.helper.node_to_operand_idxs_map[args["data"]]
        inputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_array_constant(
                        vals=transpose_idxs, 
                        dtype="int32", 
                        ), 
                    }, 
                )
        outputs = []
        outputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx((
                    tuple(map(lambda ele: args["data"].checked_type.shape[ele], transpose_idxs)), 
                    args["data"].checked_type.dtype))
                )
        nnapi_op.transpose.add_operation(converter, inputs, outputs)
        nnapi["inputs"] += outputs
        # END: add TRANSPOSE
    # END: handle input[0]

    # START: handle input[1]
    # check compatibility
    ANC_COMPATIBILITY_CHECK(args["weight"].checked_type.dtype == args["data"].checked_type.dtype)

    # generate nnapi node for weight
    converter.visit(args["weight"])

    # change layout of "weight" to NNAPI's IHWO
    ANC_COMPATIBILITY_CHECK(len(attrs.kernel_layout) == 4, f"Unrecognized layout { attrs.kernel_layout }")
    if (attrs.kernel_layout == "IHWO"): 
        nnapi["inputs"] += converter.export_obj.helper.node_to_operand_idxs_map[args["weight"]]
    else: 
        # START: add TRANSPOSE
        transpose_idxs = list(map(lambda ele: attrs.kernel_layout.index(ele), [ "I", "H", "W", "O" ]))
        inputs = []
        inputs += converter.export_obj.helper.node_to_operand_idxs_map[args["weight"]]
        inputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_array_constant(
                        vals=transpose_idxs, 
                        dtype="int32", 
                        ), 
                    }, 
                )
        outputs = []
        outputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx((
                    tuple(map(lambda ele: args["weight"].checked_type.shape[ele], transpose_idxs)), 
                    args["weight"].checked_type.dtype))
                )
        nnapi_op.transpose.add_operation(converter, inputs, outputs)
        nnapi["inputs"] += outputs
        # END: add TRANSPOSE
    # END: handle input[1]

    # START: handle input[2]
    # add empty bias
    bias_shape = (converter.export_obj.helper.operand.get_shape(nnapi["inputs"][1])[3], )
    if (args["data"].checked_type.dtype == "float32" or args["data"].checked_type.dtype == "float16"): 
        bias_dtype = args["data"].checked_type.dtype
    else: 
        raise AndroidNNAPICompilerIncompatibleError(f"Unable to determine bias data type for DEPTHWISE_CONV_2D. args['data'].dtype was { args['data'].checked_type.dtype }")
    bias_type = (bias_shape, bias_dtype)
    nnapi["inputs"] += converter.export_obj.add_operand(
            type_idx=converter.export_obj.get_type_idx(bias_type), 
            value={
                "type": "constant_idx", 
                "value": converter.export_obj.add_array_constant(
                    vals=[ 0.0 ] * bias_shape[0], 
                    dtype=bias_dtype, 
                    ), 
                }, 
            )
    # END: handle input[2]

    # START: handle input[3:7]
    def _add_int32_scalar_constant(e): 
        return converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_scalar_constant(
                        val=int(e), 
                        dtype="int32", 
                        ), 
                    }, 
                )[0]
    relay_paddings = list(map(_add_int32_scalar_constant, attrs.padding))
    if (len(relay_paddings) == 2): 
        nnapi["inputs"] += [ relay_paddings[1], relay_paddings[1], relay_paddings[0], relay_paddings[0] ]
    elif (len(relay_paddings) == 4): 
        nnapi["inputs"] += [ relay_paddings[1], relay_paddings[3], relay_paddings[0], relay_paddings[2] ]
    else: 
        raise AndroidNNAPICompilerIncompatibleError(f"Unexpected padding format { attrs.padding }")
    # END: handle input[3:7]

    # START: handle input[7:9]
    relay_strides = list(map(_add_int32_scalar_constant, attrs.strides))
    nnapi["inputs"] += [ relay_strides[1], relay_strides[0] ]
    # END: handle input[7:9]

    # START: handle input[9]
    def _s(): 
        if (API_LEVEL >= 29 and attrs.data_layout == "NCHW"): 
            depth_in = converter.export_obj.helper.operand.get_shape(nnapi["inputs"][0])[1]
        else: 
            depth_in = converter.export_obj.helper.operand.get_shape(nnapi["inputs"][0])[3]
        depth_out = converter.export_obj.helper.operand.get_shape(nnapi["inputs"][1])[3]
        assert depth_out % depth_in == 0
        depth_multiplier = int(depth_out // depth_in)
        nnapi["inputs"] += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_scalar_constant(
                        val=depth_multiplier, 
                        dtype="int32", 
                        ), 
                    }, 
                )
    _s()
    # END: handle input[9]

    # START: handle input[10]
    # add ANEURALNETWORKS_FUSED_NONE activation since CONV_2D needs it
    nnapi["inputs"] += converter.export_obj.add_operand(
            type_idx=converter.export_obj.get_type_idx(((), "int32")), 
            value={
                "type": "constant_idx", 
                "value": converter.export_obj.add_scalar_constant(
                    val="ANEURALNETWORKS_FUSED_NONE", 
                    dtype="int32", 
                    ), 
                }, 
            )
    # END: handle input[10]

    nnapi_output_layout = "NHWC"
    if (API_LEVEL >= 29): 
        # START: handle input[11]
        if (attrs.data_layout == "NCHW"): 
            nnapi["inputs"] += converter.export_obj.add_operand(
                    type_idx=converter.export_obj.get_type_idx(((), "bool")), 
                    value={
                        "type": "constant_idx", 
                        "value": converter.export_obj.add_scalar_constant(
                            val="true", 
                            dtype="bool", 
                            ), 
                        }, 
                    )
            nnapi_output_layout = "NCHW"
        else: 
            nnapi["inputs"] += converter.export_obj.add_operand(
                    type_idx=converter.export_obj.get_type_idx(((), "bool")), 
                    value={
                        "type": "constant_idx", 
                        "value": converter.export_obj.add_scalar_constant(
                            val="false", 
                            dtype="bool", 
                            ), 
                        }, 
                    )
        # END: handle input[11]

        # START: handle input[12:]
        # unpack dilation
        relay_dilations = list(map(_add_int32_scalar_constant, attrs.dilation))
        nnapi["inputs"] += [ relay_dilations[1], relay_dilations[0] ]
        # END: handle input[12:]
    # END: handle inputs

    # START: handle outputs
    nnapi["outputs"] = []

    # START: handle output[0]
    attrs_out_layout = attrs.data_layout if attrs.out_layout == "" else attrs.out_layout
    attrs_out_dtype = args["data"].checked_type.dtype if attrs.out_dtype == "" else attrs.out_dtype
    if (attrs_out_dtype == args["data"].checked_type.dtype and attrs_out_layout == nnapi_output_layout): 
        nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, node.checked_type.dtype)))
        node_operands = nnapi["outputs"]
    else: 
        if (attrs_out_layout == nnapi_output_layout): 
            nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, args["data"].checked_type.dtype)))
            last_outputs = nnapi["outputs"]
        else: 
            transpose_idxs = list(map(lambda ele: attrs_out_layout.index(ele), [ "N", "H", "W", "C" ]))
            NHWC_shape = tuple(map(lambda ele: node.checked_type.shape[ele], transpose_idxs))
            nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((NHWC_shape, args["data"].checked_type.dtype)))

            # START: add TRANSPOSE
            rev_transpose_idxs = list(map(lambda ele: "NHWC".index(ele), attrs_out_layout))
            inputs = []
            inputs += nnapi["outputs"]
            inputs += converter.export_obj.add_operand(
                    type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                    value={
                        "type": "constant_idx", 
                        "value": converter.export_obj.add_array_constant(
                            vals=rev_transpose_idxs, 
                            dtype="int32", 
                            ), 
                        }, 
                    )
            outputs = []
            outputs += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, args["data"].checked_type.dtype)))
            nnapi_op.transpose.add_operation(converter, inputs, outputs)
            # END: add TRANSPOSE

            last_outputs = outputs

        if (attrs_out_dtype == args["data"].checked_type.dtype): 
            node_operands = last_outputs
        else: 
            # START: add CAST
            inputs = []
            inputs += last_outputs
            outputs = []
            outputs += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, node.checked_type.dtype)))
            nnapi_op.cast.add_operation(converter, inputs, outputs)
            # END: add CAST
        
            node_operands = outputs

    # register operands to node
    converter.export_obj.helper.node_to_operand_idxs_map[node] = node_operands
    # END: handle output[0]
    # END: handle outputs

    nnapi_op.depthwise_conv_2d.add_operation(converter, nnapi["inputs"], nnapi["outputs"])
    

def _grouped_handler(converter, node): 
    API_LEVEL = converter.options["target"]["api_level"]
    args = utils.name_args(node.args, [ "data", "weight" ])
    attrs = node.attrs
    nnapi = {}

    # START: handle inputs
    # use explicit padding
    nnapi["inputs"] = []

    # START: handle input[0]
    # generate nnapi node of "data"
    converter.visit(args["data"])

    # change layout of "data" to NNAPI's NHWC
    ANC_COMPATIBILITY_CHECK(len(attrs.data_layout) == 4, f"Unrecognized layout { attrs.data_layout }")
    if (attrs.data_layout == "NHWC" or (API_LEVEL >= 29 and attrs.data_layout == "NCHW")): 
        nnapi["inputs"] += converter.export_obj.helper.node_to_operand_idxs_map[args["data"]]
    else: 
        # START: add TRANSPOSE
        transpose_idxs = list(map(lambda ele: attrs.data_layout.index(ele), [ "N", "H", "W", "C" ]))
        inputs = []
        inputs += converter.export_obj.helper.node_to_operand_idxs_map[args["data"]]
        inputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_array_constant(
                        vals=transpose_idxs, 
                        dtype="int32", 
                        ), 
                    }, 
                )
        outputs = []
        outputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx((
                    tuple(map(lambda ele: args["data"].checked_type.shape[ele], transpose_idxs)), 
                    args["data"].checked_type.dtype))
                )
        nnapi_op.transpose.add_operation(converter, inputs, outputs)
        nnapi["inputs"] += outputs
        # END: add TRANSPOSE
    # END: handle input[0]

    # START: handle input[1]
    # check compatibility
    ANC_COMPATIBILITY_CHECK(args["weight"].checked_type.dtype == args["data"].checked_type.dtype)

    # generate nnapi node for weight
    converter.visit(args["weight"])

    # change layout of "weight" to NNAPI's OHWI
    ANC_COMPATIBILITY_CHECK(len(attrs.kernel_layout) == 4, f"Unrecognized layout { attrs.kernel_layout }")
    if (attrs.kernel_layout == "OHWI"): 
        nnapi["inputs"] += converter.export_obj.helper.node_to_operand_idxs_map[args["weight"]]
    else: 
        # START: add TRANSPOSE
        transpose_idxs = list(map(lambda ele: attrs.kernel_layout.index(ele), [ "O", "H", "W", "I" ]))
        inputs = []
        inputs += converter.export_obj.helper.node_to_operand_idxs_map[args["weight"]]
        inputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_array_constant(
                        vals=transpose_idxs, 
                        dtype="int32", 
                        ), 
                    }, 
                )
        outputs = []
        outputs += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx((
                    tuple(map(lambda ele: args["weight"].checked_type.shape[ele], transpose_idxs)), 
                    args["weight"].checked_type.dtype))
                )
        nnapi_op.transpose.add_operation(converter, inputs, outputs)
        nnapi["inputs"] += outputs
        # END: add TRANSPOSE
    # END: handle input[1]

    # START: handle input[2]
    # add empty bias
    bias_shape = (converter.export_obj.helper.operand.get_shape(nnapi["inputs"][1])[0], )
    if (args["data"].checked_type.dtype == "float32" or args["data"].checked_type.dtype == "float16"): 
        bias_dtype = args["data"].checked_type.dtype
    else: 
        raise AndroidNNAPICompilerIncompatibleError(f"Unable to determine bias type for GROUPED_CONV_2D. args['data'].dtype was { args['data'].checked_type.dtype }")
    bias_type = (bias_shape, bias_dtype)
    nnapi["inputs"] += converter.export_obj.add_operand(
            type_idx=converter.export_obj.get_type_idx(bias_type), 
            value={
                "type": "constant_idx", 
                "value": converter.export_obj.add_array_constant(
                    vals=[ 0.0 ] * bias_shape[0], 
                    dtype=bias_dtype, 
                    ), 
                }, 
            )
    # END: handle input[2]

    # START: handle input[3:7]
    def _add_int32_scalar_constant(e): 
        return converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((), "int32")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_scalar_constant(
                        val=int(e), 
                        dtype="int32", 
                        ), 
                    }, 
                )[0]
    relay_paddings = list(map(_add_int32_scalar_constant, attrs.padding))
    if (len(relay_paddings) == 2): 
        nnapi["inputs"] += [ relay_paddings[1], relay_paddings[1], relay_paddings[0], relay_paddings[0] ]
    elif (len(relay_paddings) == 4): 
        nnapi["inputs"] += [ relay_paddings[1], relay_paddings[3], relay_paddings[0], relay_paddings[2] ]
    else: 
        raise AndroidNNAPICompilerIncompatibleError(f"Unexpected padding format { attrs.padding }")
    # END: handle input[3:7]

    # START: handle input[7:9]
    relay_strides = list(map(_add_int32_scalar_constant, attrs.strides))
    nnapi["inputs"] += [ relay_strides[1], relay_strides[0] ]
    # END: handle input[7:9]

    # START: handle input[9]
    nnapi["inputs"] += converter.export_obj.add_operand(
            type_idx=converter.export_obj.get_type_idx(((), "int32")), 
            value={
                "type": "constant_idx", 
                "value": converter.export_obj.add_scalar_constant(
                    val=int(attrs.groups), 
                    dtype="int32", 
                    ), 
                }, 
            )
    # END: handle input[9]

    # START: handle input[10]
    # add ANEURALNETWORKS_FUSED_NONE activation since CONV_2D needs it
    nnapi["inputs"] += converter.export_obj.add_operand(
            type_idx=converter.export_obj.get_type_idx(((), "int32")), 
            value={
                "type": "constant_idx", 
                "value": converter.export_obj.add_scalar_constant(
                    val="ANEURALNETWORKS_FUSED_NONE", 
                    dtype="int32", 
                    ), 
                }, 
            )
    # END: handle input[10]

    # START: handle input[11]
    nnapi_output_layout = "NHWC"
    if (attrs.data_layout == "NCHW"): 
        nnapi["inputs"] += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((), "bool")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_scalar_constant(
                        val="true", 
                        dtype="bool", 
                        ), 
                    }, 
                )
        nnapi_output_layout = "NCHW"
    else: 
        nnapi["inputs"] += converter.export_obj.add_operand(
                type_idx=converter.export_obj.get_type_idx(((), "bool")), 
                value={
                    "type": "constant_idx", 
                    "value": converter.export_obj.add_scalar_constant(
                        val="false", 
                        dtype="bool", 
                        ), 
                    }, 
                )
    # END: handle input[11]
    # END: handle inputs

    # START: handle outputs
    nnapi["outputs"] = []

    # START: handle output[0]
    attrs_out_layout = attrs.data_layout if attrs.out_layout == "" else attrs.out_layout
    attrs_out_dtype = args["data"].checked_type.dtype if attrs.out_dtype == "" else attrs.out_dtype
    if (attrs_out_dtype == args["data"].checked_type.dtype and attrs_out_layout == nnapi_output_layout): 
        nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, node.checked_type.dtype)))
        node_operands = nnapi["outputs"]
    else: 
        if (attrs_out_layout == nnapi_output_layout): 
            nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, args["data"].checked_type.dtype)))
            last_outputs = nnapi["outputs"]
        else: 
            transpose_idxs = list(map(lambda ele: attrs_out_layout.index(ele), [ "N", "H", "W", "C" ]))
            NHWC_shape = tuple(map(lambda ele: node.checked_type.shape[ele], transpose_idxs))
            nnapi["outputs"] += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((NHWC_shape, args["data"].checked_type.dtype)))

            # START: add TRANSPOSE
            rev_transpose_idxs = list(map(lambda ele: "NHWC".index(ele), attrs_out_layout))
            inputs = []
            inputs += nnapi["outputs"]
            inputs += converter.export_obj.add_operand(
                    type_idx=converter.export_obj.get_type_idx(((4, ), "int32")), 
                    value={
                        "type": "constant_idx", 
                        "value": converter.export_obj.add_array_constant(
                            vals=rev_transpose_idxs, 
                            dtype="int32", 
                            ), 
                        }, 
                    )
            outputs = []
            outputs += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, args["data"].checked_type.dtype)))
            nnapi_op.transpose.add_operation(converter, inputs, outputs)
            # END: add TRANSPOSE

            last_outputs = outputs

        if (attrs_out_dtype == args["data"].checked_type.dtype): 
            node_operands = last_outputs
        else: 
            # START: add CAST
            inputs = []
            inputs += last_outputs
            outputs = []
            outputs += converter.export_obj.add_operand(type_idx=converter.export_obj.get_type_idx((node.checked_type.shape, node.checked_type.dtype)))
            nnapi_op.cast.add_operation(converter, inputs, outputs)
            # END: add CAST
        
            node_operands = outputs

    # register operands to node
    converter.export_obj.helper.node_to_operand_idxs_map[node] = node_operands
    # END: handle output[0]
    # END: handle outputs

    nnapi_op.grouped_conv_2d.add_operation(converter, nnapi["inputs"], nnapi["outputs"])
    
