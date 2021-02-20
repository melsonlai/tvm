from .operand import Operand as _Operand

class Helper: 
    def __init__(self, export_obj): 
        self._export_obj = export_obj
        self.node_to_operand_idxs_map = {}
        self.type_to_idx_map = {}
        self.operand = _Operand(self._export_obj)


