from .symbolize import symbolize
from .declare_types import declare_types
from .declare_operands import declare_operands
from .declare_constants import declare_constants
from .declare_memories import declare_memories
from .initialize_operands import initialize_operands
from .declare_operations import declare_operations
from .declare_inputs_outputs import declare_inputs_outputs
from .declare_wrapper_class import declare_wrapper_class
from .set_execution_inputs_outputs import set_execution_inputs_outputs
from .finalize import finalize


stages = [
        # model creation
        symbolize, 
        declare_types, 
        declare_operands, 
        declare_constants, 
        declare_memories, 
        initialize_operands, 
        declare_operations, 
        declare_inputs_outputs, 

        # set execution io
        set_execution_inputs_outputs, 

        # finalize
        declare_wrapper_class, 
        finalize, 
        ]


