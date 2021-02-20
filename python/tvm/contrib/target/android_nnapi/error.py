class AndroidNNAPICompilerError(RuntimeError): 
    """Android NNAPI compiler error base class
    """
    pass


class AndroidNNAPICompilerIncompatibleError(AndroidNNAPICompilerIncompatibleError): 
    """Error caused by parsing unsupported Relay AST
    """
    pass


def ANC_COMPATIBILITY_CHECK(boolean, *msg): 
    if (not boolean): 
        raise AndroidNNAPICompilerIncompatibleError(*msg)

