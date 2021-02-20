from .. import templates


C_TYPES_MAP = {
        "int32": "int32_t",
        "uint32": "uint32_t",
        "float16": "uint16_t", 
        "float32": "float", 
        "bool": "bool", 
        }


def declare_constants(lines, export_obj, options): 
    for c in export_obj["constants"]: 
        tipe = c["type"]
        c_dtype = C_TYPES_MAP[c["dtype"]]
        if (tipe == "scalar"): 
            data = {
                    "dtype": c_dtype, 
                    "name": c["name"], 
                    "value": c["value"], 
                    }
        elif (tipe == "array"): 
            data = {
                    "dtype": c_dtype, 
                    "name": c["name"], 
                    "length": len(c["value"]), 
                    "value": "{" + ", ".join([ str(v) for v in c["value"] ]) + "}", 
                    }
        else: 
            raise RuntimeError("Unknown constant type {}".format(tipe))
        lines["tmp"]["model_creation"].append(templates.declare_constant[tipe].substitute(**data))
    return lines, export_obj


