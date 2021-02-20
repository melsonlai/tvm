def name_args(args, arg_names): 
    """Put arguments into dict

    Parameters
    ----------
    args: array of relay.Expr
        args of relay.Call

    arg_names: array of string
        names of args

    Returns
    -------
    args_map: dict of string to relay.Expr
        named args dict
    """
    assert len(args) == len(arg_names)
    return { k: v for (k, v) in zip(arg_names, args) }


