def symbolize(lines, export_obj, options): 
    def _symbolize_types(types): 
        cnts = {
                "tensor": 0, 
                "scalar": 0, 
                }
        for t in types: 
            if (t["type"].startswith("TENSOR_")): 
                t["name"] = "tensor" + str(cnts["tensor"])
                cnts["tensor"] += 1
            else: 
                t["name"] = "scalar" + str(cnts["scalar"])
                cnts["scalar"] += 1
    _symbolize_types(export_obj["types"])

    def _symbolize_consts(consts): 
        cnt = 0
        for c in consts: 
            c["name"] = "const_val" + str(cnt)
            cnt += 1
    if ("constants" in export_obj): 
        _symbolize_consts(export_obj["constants"])

    return lines, export_obj


