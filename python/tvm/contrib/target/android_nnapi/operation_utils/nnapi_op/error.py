from ...error import *


class AndroidNNAPICompilerBadNNAPIOperationError(AndroidNNAPICompilerError): 
    """Error caused by unexpected parse result of the Relay AST
    """
    pass


def ANC_NNAPI_OP_CHECK(boolean, *msg): 
    if (not boolean): 
        raise AndroidNNAPICompilerBadNNAPIOperationError(*msg)

