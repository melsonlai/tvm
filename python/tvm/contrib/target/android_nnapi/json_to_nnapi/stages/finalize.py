import re


def finalize(lines, export_obj, options): 
    lines["result"] = "\n".join(lines["tmp"]["wrapper_class"])
    lines["result"] = "\n".join([ s for s in lines["result"].split("\n") if s.strip() ])
    return lines, export_obj
