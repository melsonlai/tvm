class Operand: 
    def __init__(self, export_obj): 
        self._export_obj = export_obj

    def get_dtype(self, idx): 
        """Get operand dtype

        Parameters
        ----------
        idx: int
            operand to be queried

        Returns
        -------
        dtype: str
            dtype of the queried operand

        """
        return self._export_obj["types"][self._export_obj["operands"][idx]["type"]]["type"]

    def get_shape(self, idx): 
        """Get operand shape
    
        Parameters
        ----------
        idx: int
            operand to be queried
    
        Returns
        -------
        shape: tuple of int or None
            shape of the queried operand. None if operand has no shape. 
    
        """
        return self._export_obj["types"][self._export_obj["operands"][idx]["type"]].get("shape", None)
    
    def get_rank(self, idx): 
        """Get operand rank
    
        Parameters
        ----------
        idx: int
            operand to be queried
    
        Returns
        -------
        rank: int
            rank of the queried operand
    
        """
        shape = self.get_shape(idx)
        if (shape == None): 
            return 0
        else: 
            return len(shape)
    
    def get_value(self, idx): 
        """Get operand value
    
        Parameters
        ----------
        idx: int
            operand to be queried
    
        Returns
        -------
        value: 
            value of the queried operand. None if there's no value. 
    
        """
        value_dict = self._export_obj["operands"][idx].get("value", None)
        if (value_dict == None): 
            return None
    
        if (value_dict["type"] == "constant_idx"): 
            return self._export_obj["constants"][value_dict["value"]]["value"]
        elif (value_dict["type"] == "memory_ptr"): 
            return value_dict["value"]
        else: 
            assert False, "Unreachable"
    
    def get_constant(self, idx): 
        """Get operand constant
    
        Parameters
        ----------
        idx: int
            operand to be queried
    
        Returns
        -------
        obj: 
            constant object of the queried operand. None if there's no value. 
    
        """
        value_dict = self._export_obj["operands"][idx].get("value", None)
        if (value_dict == None or value_dict["type"] != "constant_idx"): 
            return None
        return self._export_obj["constants"][value_dict["value"]]

    def is_FuseCode(self, idx): 
        """Check whether the operand pointed by idx is a FuseCode
    
        Parameters
        ----------
        idx: int
            the index of the queried operand
    
        Returns
        -------
        b: bool
            the queried operand is a FuseCode or not
    
        """
        dtype = self.get_dtype(idx)
        if (dtype != "INT32"): 
            return False
        shape = self.get_shape(idx)
        if (shape != None): 
            return False
        value = self.get_value(idx)
        if (not (value == "ANEURALNETWORKS_FUSED_NONE" \
                or value == "ANEURALNETWORKS_FUSED_RELU" \
                or value == "ANEURALNETWORKS_FUSED_RELU1" \
                or value == "ANEURALNETWORKS_FUSED_RELU6")): 
            return False
        return True

