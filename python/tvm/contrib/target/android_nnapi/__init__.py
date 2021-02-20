import copy
import tvm
from .converter import Converter


@tvm.register_func("relay.ext.android_nnapi_compiler.relayir_to_nnapi_converter")
def relayir_to_nnapi_converter(func): 
    """Converts a Relay IR Function to Android NNAPI C++ source code

    Parameters
    ----------
    func: tvm.relay.Function
        The function to be converted to NNAPI. Required. 

    Returns
    -------
    code: str
        The resulting Android NNAPI code. 

    Note
    ----
    Certain function attributes can be configured: 

    * func.attrs.NnapiClassName: (str) The name of the generated class wrapped around ANN model. Defaults to "AnnGraph". 
    * func.attrs.NnapiTargetVersion: (int) The targeting API level of Android. Defaults to 29. 

    """
    assert isinstance(func, tvm.relay.Function)

    options = {
            "class": {
                "self": {
                    "name": str(func.attrs.NnapiClassName)[1:-1], 
                    }, 
                }, 
            "target": {
                "api_level": int(func.attrs.NnapiTargetVersion), 
                }, 
            }
    options = _expand_options(options)

    converter = Converter(options)
    return converter.convert(node)


def _expand_options(options): 
    """Expand options

    Parameters
    ----------
    options: dict
        The options to be expanded. 

    Returns
    -------
    options: dict
        The updated options. 
    """
    DEFAULT_OPTIONS = {
            "class": {
                "base_path": "/sdcard/r2n/AnnGraph/", # This option is here for loading weights from external storage directly. However, the feature is disabled for now due to its complexity to setup
                "self": {
                    "name": "AnnGraph", 
                    }, 
                }, 
            "target": {
                "api_level": 29, 
                }, 
            }
    ret = copy.deepcopy(options)
    def _recursive_merge(cur_opts, def_opts): 
        for k, v in def_opts.items(): 
            if (k in cur_opts): 
                if (type(v) == dict): 
                    assert type(v) == type(cur_opts[k])
                    _recursive_merge(cur_opts[k], v)
                else: 
                    assert any([ type(cur_opts[k]) == t for t in [ float, int, str ] ]) # type(cur_opts[k]) should be a basic type
            else: # option k does not exist in current options, so copy from default options
                cur_opts[k] = copy.deepcopy(v)
    _recursive_merge(ret, DEFAULT_OPTIONS)

    return ret


