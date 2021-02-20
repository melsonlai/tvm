from .. import templates


def declare_memories(lines, export_obj, options): 
    for m in export_obj["memories"]: 
        data = {
                "file_path": "{}/{}".format(options["class"]["base_path"], m["file_name"]), 
                "mem_size": m["size"], 
                }
        lines["tmp"]["model_creation"].append(templates.declare_memory.substitute(**data))
    return lines, export_obj


