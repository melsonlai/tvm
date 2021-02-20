from .. import templates


def declare_operands(lines, export_obj, options): 
    for i, op in enumerate(export_obj["operands"]): 
        op_type = export_obj["types"][op["type"]]
        data = {
                "model": options["model"]["name"], 
                "type": op_type["name"], 
                "index": i, 
                }
        lines["tmp"]["model_creation"].append(templates.declare_operand.substitute(**data))
    return lines, export_obj


