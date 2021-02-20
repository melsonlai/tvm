from .. import templates


def initialize_operands(lines, export_obj, options): 
    for i, op in enumerate(export_obj["operands"]): 
        value = op.get("value", None)
        if (value == None): 
            continue

        data = {
                "model": options["model"]["name"], 
                "op_idx": i, 
                }
        if (value["type"] == "constant_idx"): 
            const = export_obj["constants"][value["value"]]
            data["memory_size"] = "sizeof({})".format(const["name"])
            if (const["type"] == "scalar"): 
                data["memory_ptr"] = "&" + const["name"]
            elif (const["type"] == "array"): 
                data["memory_ptr"] = const["name"]
            else: 
                raise RuntimeError("Unknown const type ({}) for operand {}".format(const["type"], i))
            lines["tmp"]["model_creation"].append(templates.initialize_operand["memory_ptr"].substitute(**data))
        elif (value["type"] == "memory_ptr"): 
            pass
        elif (value["type"] == "ann_memory"): 
            memory = export_obj["memories"][value["value"]]
            data["memory_idx"] = value["value"]
            data["length"] = memory["size"]
            lines["tmp"]["model_creation"].append(templates.initialize_operand["ann_memory"].substitute(**data))
        else: 
            raise RuntimeError("Unknown value type ({}) for operand {}".format(value["type"], i))
    return lines, export_obj


