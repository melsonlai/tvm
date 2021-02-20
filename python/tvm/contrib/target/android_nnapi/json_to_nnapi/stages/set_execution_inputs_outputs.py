import re
from functools import reduce
from .. import templates


def set_execution_inputs_outputs(lines, export_obj, options): 
    for i, op_i in enumerate(export_obj["inputs"]): 
        op = export_obj["operands"][op_i]
        value = op["value"]
        assert (value["type"] == "memory_ptr")

        data = {
                "execution": options["execution"]["name"], 
                "input_idx": i, 
                }
        tipe = export_obj["types"][op["type"]]
        nnapi_dtype = tipe["type"]
        nbits = int((lambda s: s if s != "" else "8")(re.sub(r"^[^0-9]+", "", nnapi_dtype)))
        assert ((nbits != 0) and (nbits % 8 == 0))
        data["memory_ptr"] = value["value"]
        if (nnapi_dtype.startswith("TENSOR")): 
            data["memory_size"] = reduce(lambda a, b: a * b, tipe["shape"], 1) * nbits // 8
        else: 
            data["memory_size"] = nbits // 8
        lines["tmp"]["set_execution_io"].append(templates.set_execution_input.substitute(**data))
    def _outputs(): 
        assert (len(export_obj["outputs"]) == 1)
        op = export_obj["operands"][export_obj["outputs"][0]]
        value = op["value"]
        assert (value["type"] == "memory_ptr")

        data = {
                "execution": options["execution"]["name"], 
                "output_idx": 0, 
                }
        tipe = export_obj["types"][op["type"]]
        nnapi_dtype = tipe["type"]
        nbits = int((lambda s: s if s != "" else "8")(re.sub(r"^[^0-9]+", "", nnapi_dtype)))
        assert ((nbits != 0) and (nbits % 8 == 0))
        data["memory_ptr"] = value["value"]
        if (nnapi_dtype.startswith("TENSOR")): 
            data["memory_size"] = reduce(lambda a, b: a * b, tipe["shape"], 1) * nbits // 8
        else: 
            data["memory_size"] = nbits // 8
        lines["tmp"]["set_execution_io"].append(templates.set_execution_output.substitute(**data))
    _outputs()
    return lines, export_obj


