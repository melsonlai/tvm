import copy
from .stages import stages


DEFAULT_OPTIONS = {
        "class": {
            "base_path": "/sdcard/nnapi_result", 
            "name": "AnnGraph", 
            }, 
        "model": {
            "name": "model", 
            }, 
        "compilation": {
            "name": "compilation", 
            }, 
        "execution": {
            "name": "run", 
            "end_event_name": "run_end", 
            }, 
        }


def convert(export_obj, options={}): 
    """Convert export_obj to NNAPI codes

    Parameters
    ----------
    export_obj: dict
        The json representation of a NNAPI model. 

    options["class"]["base_path"]: str
        The base path of file accesses. Defaults to "/sdcard/nnapi_result". 

    options["class"]["name"]: str
        The name of the generated C++ class wrapping around NNAPI codes. Defaults to "AnnGraph". 

    options["model"]["name"]: str
        The name of the `ANeuralNetworksModel*` created. Defaults to "model". 

    options["compilation"]["name"]: str
        The name of the `ANeuralNetworksCompilation*` created. Defaults to "compilation". 

    options["execution"]["name"]: str
        The name of the `ANeuralNetworksExecution*` created. Defaults to "run". 

    options["execution"]["end_event_name"]: str
        The name of the `ANeuralNetworksEvent*` used to wait for execution completion. Defaults to "run_end". 

    Returns
    -------
    code: str
        The generated code
    """
    lines = {
            "tmp": {
                "model_creation": [], 
                "set_execution_io": [], 
                "wrapper_class": [], 
                }, 
            "result": "", 
            }
    options = _set_options(options)
    _export_obj = copy.deepcopy(export_obj)

    for s in stages: 
        lines, _export_obj = s(lines, _export_obj, options)

    return lines["result"]


def _set_options(options): 
    """Set options

    Parameters
    ----------
    options: dict
        The options to be set. 

    Returns
    -------
    options: dict
        The updated options. 
    """
    def _recursive_merge(cur_opts, def_opts): 
        for k, v in def_opts.items(): 
            if (k in cur_opts): 
                if (type(v) == dict): 
                    assert (type(v) == type(cur_opts[k]))
                    _recursive_merge(cur_opts[k], v)
                else: 
                    assert (any([ type(cur_opts[k]) == t for t in [ float, int, str ] ]))
            else: 
                cur_opts[k] = copy.deepcopy(v)
    _recursive_merge(options, DEFAULT_OPTIONS)

    return options
