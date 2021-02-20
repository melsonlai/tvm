from .. import templates


# NOTICE: make sure TVM maps type A to type B before modifying this table!!
C_TYPES_MAP = {
        "BOOL": "bool", 
        "FLOAT32": "float", 
        "INT32": "int", 
        "TENSOR_BOOL8": "bool", 
        "TENSOR_FLOAT16": "uint16_t", 
        "TENSOR_FLOAT32": "float", 
        "TENSOR_INT32": "int", 
        }


def declare_wrapper_class(lines, export_obj, options): 
    data = {
            "class": {
                "self": {
                    "name": options["class"]["name"], 
                    }, 
                "model": {
                    "name": options["model"]["name"], 
                    }, 
                "compilation": {
                    "name": options["compilation"]["name"], 
                    }, 
                "execution": {
                    "name": options["execution"]["name"], 
                    "end_event_name": options["execution"]["end_event_name"], 
                    }
                }, 
            "codes": {
                "model_creation": "\n".join([ "    " + s for s in "\n".join(lines["tmp"]["model_creation"]).split("\n") ]), 
                "set_execution_io": "\n".join([ "    " + s for s in "\n".join(lines["tmp"]["set_execution_io"]).split("\n") ]), 
                }, 
            }
    def _s(): 
        var_decls = []
        for i in export_obj["inputs"]: 
            op = export_obj["operands"][i]
            assert (op["value"]["type"] == "memory_ptr")
            tipe = export_obj["types"][op["type"]]
            var_decls.append("{}* {}".format(C_TYPES_MAP[tipe["type"]], op["value"]["value"]))
        for o in export_obj["outputs"]: 
            op = export_obj["operands"][o]
            assert (op["value"]["type"] == "memory_ptr")
            tipe = export_obj["types"][op["type"]]
            var_decls.append("{}* {}".format(C_TYPES_MAP[tipe["type"]], op["value"]["value"]))
        data["class"]["execution"]["func_params_decl_str"] = ", ".join(var_decls)
    _s()
    lines["tmp"]["wrapper_class"].append(templates.declare_wrapper_class.substitute(**data))
    return lines, export_obj


