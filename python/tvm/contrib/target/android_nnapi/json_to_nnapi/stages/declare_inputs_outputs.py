from .. import templates


def declare_inputs_outputs(lines, export_obj, options): 
    inputs = export_obj["inputs"]
    outputs = export_obj["outputs"]
    data = {
            "inputs": {
                "length": len(inputs), 
                "str": "{" + ", ".join([ str(i) for i in inputs ]) + "}", 
                }, 
            "outputs": {
                "length": len(outputs), 
                "str": "{" + ", ".join([ str(i) for i in outputs ]) + "}", 
                }, 
            "model": options["model"]["name"], 
            }
    lines["tmp"]["model_creation"].append(templates.declare_inputs_outputs.substitute(**data))
    return lines, export_obj


