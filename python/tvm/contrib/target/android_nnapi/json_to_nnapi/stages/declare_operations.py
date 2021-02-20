from .. import templates


def declare_operations(lines, export_obj, options): 
    for op in export_obj["operations"]: 
        data = {
                "inputs": {
                    "length": len(op["input"]), 
                    "str": "{" + ", ".join([ str(i) for i in op["input"] ]) + "}", 
                    }, 
                "outputs": {
                    "length": len(op["output"]), 
                    "str": "{" + ", ".join([ str(i) for i in op["output"] ]) + "}", 
                    }, 
                "model": options["model"]["name"], 
                "op_code": templates.ANN_PREFIX + op["op"], 
                }
        lines["tmp"]["model_creation"].append(templates.declare_operation.substitute(**data))
    return lines, export_obj


