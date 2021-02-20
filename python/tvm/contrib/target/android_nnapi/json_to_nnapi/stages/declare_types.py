from .. import templates

def declare_types(lines, export_obj, options): 
    for t in export_obj["types"]: 
        tipe = {
                "name": t["name"], 
                "type": templates.ANN_PREFIX + t["type"], 
                }
        if ("shape" in t): 
            tipe["dim_name"] = tipe["name"] + "_dims"
            tipe["shape"] = {
                    "rank": len(t["shape"]), 
                    "str": "{" + ", ".join([ str(i) for i in t["shape"] ]) + "}", 
                    }
        lines["tmp"]["model_creation"].append(templates.declare_type.substitute(tipe=tipe))
    return lines, export_obj


