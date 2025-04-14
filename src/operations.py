from cgp.node import OperatorNode

class Constant255(OperatorNode):
    """A node with a constant output."""

    _arity = 0
    _def_output = "255"
    _def_numpy_output = "np.ones(len(x[0])) * 255"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * 255"
    

class Identity(OperatorNode):
    """Returns its input as output."""

    _arity = 1
    _def_output = "x_0"
    
class Inversion(OperatorNode):
    """Returns the inverted value of its input."""

    _arity = 1
    _def_output = "255 - x_0"

class Max(OperatorNode):
    """Returns the maximum value of its two inputs."""

    _arity = 2
    _def_output = "max(x_0, x_1)"
    _def_numpy_output = "np.maximum(x_0, x_1)"
    _def_torch_output = "torch.max(x_0, x_1)"
    # sympy, because it is needed for some reason
    _def_sympy_output = "x_0"

class Min(OperatorNode):
    """Returns the minimum value of its two inputs."""

    _arity = 2
    _def_output = "min(x_0, x_1)"
    _def_numpy_output = "np.minimum(x_0, x_1)"
    _def_torch_output = "torch.min(x_0, x_1)"
    # sympy, because it is needed for some reason
    _def_sympy_output = "x_0"

class UINTDivBy2(OperatorNode):
    """Returns the value of its input divided by 2."""

    _arity = 1
    _def_output = "x_0 // 2"  # Integer division to stay in uint8 range

class UINTDivBy4(OperatorNode):
    """Returns the value of its input divided by 4."""

    _arity = 1
    _def_output = "x_0 // 4"  # Integer division to stay in uint8 range

class UINT8Add(OperatorNode):
    """A node that adds its two inputs with uint8 overflow."""

    _arity = 2
    _def_output = "x_0 + x_1 % 256"  # Modulo to stay in uint8 range
    _def_numpy_output = "np.mod(np.add(x_0, x_1), 256)"  # Modulo to stay in uint8 range

class UINT8AddSat(OperatorNode):
    """A node that adds its two inputs without uint8 overflow."""

    _arity = 2
    _def_output = "x_0 + min(255 - x_0, x_1)"  # Saturation to stay in uint8 range
    _def_numpy_output = "x_0 + np.minimum(255 - x_0, x_1)"  # Modulo to stay in uint8 range
    _def_torch_output = "torch.clamp(x_0 + x_1, 0, 255)"  # Modulo to stay in uint8 range

class Average(OperatorNode):
    """A node that averages its two inputs."""

    _arity = 2
    _def_output = "(x_0 + x_1) // 2"  # Integer division to stay in uint8 range

class ConditionalAssign(OperatorNode):
    """A node that assigns (x > 127) ? y : x """

    _arity = 2
    _def_output = "x_1 if x_0 > 127 else x_0"
    _def_numpy_output = "np.where(x_0 > 127, x_1, x_0)"
    _def_torch_output = "torch.where(x_0 > 127, x_1, x_0)"
    # sympy, because it is needed for some reason
    _def_sympy_output = "x_0"

class AbsoluteDiff(OperatorNode):
    """A node that calculates the absolute difference between two inputs."""

    _arity = 2
    _def_output = "abs(x_0 - x_1)"
    _def_numpy_output = "np.abs(x_0 - x_1)"

