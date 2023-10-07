from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)
    tmp = f(*vals)
    vals[arg] += epsilon
    return (f(*vals) - tmp) / epsilon 
     


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: need to implement by self
    PermanentMarked = []
    TemporaryMarked = []
    result = []
    def visit (n: Variable) :
    # Don ’ t do anything with constants
        if n.is_constant() :
            return
        if n.unique_id in PermanentMarked:
            return 
        elif n.unique_id in TemporaryMarked:
            raise(RuntimeError("not a Dag"))
        TemporaryMarked.append(n.unique_id)
        if n.is_leaf():
            pass
        else:
            for inp in n.history.inputs:
                visit(inp)
        TemporaryMarked.remove(n.unique_id)
        PermanentMarked.append(n.unique_id)
        result.insert(0, n)
    visit(variable)
    print(len(result))
    return result



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # fn = variable.history.last_fn
    # variable.derivative += fn(variable.history.ctx, deriv)
    # for parent in variable.parents:
    #     backpropagate(parent, variable.derivative)

    order = topological_sort(variable)
    derivs = {variable.unique_id:deriv}
    for node in order:
        d_output = derivs[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d_output)
        else :
            for inp, d in node.chain_rule(d_output):
                if inp.unique_id not in derivs:
                    derivs[inp.unique_id] = 0.0
                derivs[inp.unique_id] += d
    return 

    


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
