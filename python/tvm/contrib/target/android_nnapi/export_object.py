import struct
import copy
from .error import *
from ._export_object import Helper as _Helper


# ExportObject is a dict-like class that acts like a JSON object with methods for Android NNAPI model construction
class ExportObject: 
    _SCALAR_RELAY_NNAPI_TYPE_MAP = {
        "bool": "BOOL", 
        "float16": "FLOAT16", 
        "float32": "FLOAT32", 
        "int32": "INT32", 
        "uint32": "UINT32", 
    }

    _TENSOR_RELAY_NNAPI_TYPE_MAP = {
        "bool": "TENSOR_BOOL", 
        "float16": "TENSOR_FLOAT16", 
        "float32": "TENSOR_FLOAT32", 
        "int32": "TENSOR_INT32", 
        "uint32": "TENSOR_UINT32", 
    }

    def __init__(self, options): 
        self.helper = _Helper(self)
        self._json = {
                "constants": [], 
                "inputs": [], 
                "memories": [], 
                "operands": [], 
                "operations": [], 
                "outputs": [], 
                "types": [], 
                }
        self._options = options

    def __getitem__(self, key): 
        return self._json[key]

    def __setitem__(self, key, value): 
        self._json[key] = value

    def asjson(self): 
        return copy.deepcopy(self._json)

    def get_type_idx(self, tipe): 
        """Register and lookup type index in export_obj["types"]

        Parameters
        ----------
        tipe: ((int, ...), str)
            type (shape, dtype) to look up

        Returns
        -------
        index: int
            type index in export object
        """
        tipe = (tuple(map(lambda e: int(e), tipe[0])), str(tipe[1])) # canonicalize
        shape, dtype = tipe
        ANC_COMPATIBILITY_CHECK(str(dtype) in [ "bool", "float16", "float32", "int32", "uint32" ], "Unsupported data type { dtype }")

        if (self.helper.type_to_idx_map.get(tipe, None) == None): # create new type
            shape, dtype = tipe

            if (dtype == "bool"): 
                ANC_COMPATIBILITY_CHECK(self._options["target"]["api_level"] >= 29, f"Boolean is not supported for Android API{ self._options['target']['api_level'] }")

            new_type = {}
            if (len(shape) == 0): 
                new_type["type"] = self._SCALAR_RELAY_NNAPI_TYPE_MAP[dtype]
            else: 
                new_type["shape"] = list(shape)
                new_type["type"] = self._TENSOR_RELAY_NNAPI_TYPE_MAP[dtype]

            self["types"].append(new_type)
            self.helper.type_to_idx_map[tipe] = len(self["types"]) - 1
        return self.helper.type_to_idx_map[tipe]

    @staticmethod
    def _canonicalize_scalar_constant(dtype, val): 
        if (not isinstance(val, str)): # skip canonicalizing strings as they may carry specific meanings (e.g. macro-defined values)
            if (dtype == "float16"): 
                if (isinstance(e, float)): 
                    val = hex(struct.unpack("H", struct.pack("e", val))[0]) # for float16 we use uint16_t in C, hence the conversion
            elif (dtype == "float32"): 
                val = float(val)
            elif (dtype == "int32"): 
                val = int(val)
            elif (dtype == "uint32"): 
                val = int(val)
            else: 
                assert False, "Unreachable"
        return val

    def add_scalar_constant(self, val, dtype): 
        """Add scalar constant to export object

        Parameters
        ----------
        val: numerical or str
            value of the constant. Can be defined constant in the NNAPI framework. 

        dtype: str
            data type of the constant

        Returns
        -------
        index: int
            index of the constant in export object constants array
        """
        # canonicalize
        dtype = str(dtype)
        ANC_COMPATIBILITY_CHECK(dtype in [ "float16", "float32", "int32", "uint32" ], "Unsupported data type { dtype }")
        val = self._canonicalize_scalar_constant(dtype, val)

        new_const = {
                "type": "scalar", 
                "dtype": dtype, 
                "value": val, 
                }
        if (new_const in self["constants"]): 
            return self["constants"].index(new_const)
        else: 
            self["constants"].append(new_const)
            return len(self["constants"]) - 1

    def add_array_constant(self, vals, dtype): 
        """Add array constant to export object

        Parameters
        ----------
        vals: array of values in dtype
            values of array

        dtype: string
            data type of array

        Returns
        -------
        index: int
            index of added constant in export_obj["constants"]
        """
        # canonicalize
        dtype = str(dtype)
        ANC_COMPATIBILITY_CHECK(dtype in [ "float16", "float32", "int32", "uint32" ], "Unsupported data type { dtype }")
        assert len(vals) > 0, "Array constant should not be empty"
        vals = list(map(self._canonicalize_scalar_constant, vals))

        new_const = {
                "type": "array", 
                "dtype": dtype, 
                "value": vals, 
                }
        if (new_const in self["constants"]): 
            return self["constants"].index(new_const)
        else: 
            self["constants"].append(new_const)
            return len(self["constants"]) - 1

    def add_operand(self, type_idx, **kwargs): 
        """Add node to export_obj["operands"] and return its index

        Parameters
        ----------
        type_idx: int
            index of node type in export_obj["types"]

        kwargs["value"]: dict
            dict representing node value. See below for more info

        kwargs["value"]["type"]: str
            type of value. Can be "constant_idx", "memory_ptr"

        kwargs["value"]["value"]: 
            value of initialized value. Should correspond to `kwargs["value"]["type"]`

        kwargs["node"]: relay.Node
            node to add. Use `None` to prevent operand being added to `node_to_operand_idxs_map`

        Returns
        -------
        indices: array of int
            indices of node in export_obj["operands"]
        """
        node = kwargs.get("node", None)
        value = kwargs.get("value", None)

        new_op = {
                "type": type_idx, 
                }

        if (value != None): 
            new_op["value"] = value

        if (node != None and self.helper.node_to_operand_idxs_map.get(node, None) != None): 
            assert self["operands"][self.helper.node_to_operand_idxs_map[node][0]] == new_op
            return self.helper.node_to_operand_idxs_map[node]

        self["operands"].append(new_op)
        ret = [ len(self["operands"]) - 1 ]
        if (node != None): 
            self.helper.node_to_operand_idxs_map[node] = ret
        return ret

    def add_operation(self, nnapi_op_name, inputs, outputs): 
        """Add operation to export_obj["operations"]
    
        Parameters
        ----------
        nnapi_op_name: str
            name of operator to be added in NNAPI
    
        inputs: array of int
            indices of input operands
    
        outputs: array of int
            indices of output operands
        """
        new_op = {
                "input": inputs, 
                "op": nnapi_op_name, 
                "output": outputs, 
                }
        self["operations"].append(new_op)

    def add_ann_memory(self, file_name, size): 
        """Add memory to export_obj["memories"]
    
        Parameters
        ----------
        file_name: str
            file name or relative path to the underlying file of memory
    
        size: int
            size in bytes of the underlying file

        Returns
        -------
        idx: int
            the index of the new memory
        """
        new_mem = {
                "file_name": file_name, 
                "size": size, 
                }
        if (new_mem not in self["memories"]): 
            self["memories"].append(new_mem)

        return self["memories"].index(new_mem)


