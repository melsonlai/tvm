import tvm
from . import transform
from . import json_to_nnapi
from .function_to_json_converter import FunctionToJsonConverter

class Converter: 
    def __init__(self, options): 
        self._options = options

    def convert(self, func): 
        assert isinstance(func, tvm.relay.Function)
        func = transform.FixIllegalPatternForNnapi()(func)

        mod = tvm.IRModule({ "main": func })
        export_obj = FunctionToJsonConverter(self._options)(mod["main"])

        ret = json_to_nnapi.convert(
                export_obj=export_obj.asjson(), 
                options={
                    "class": {
                        "base_path": self._options["class"]["base_path"], 
                        "name": self._options["class"]["self"]["name"], 
                        }, 
                    }, 
                )
        return ret


