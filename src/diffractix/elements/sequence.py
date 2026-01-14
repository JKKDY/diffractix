from typing import Dict, List, Tuple
from .base import OpticalElement

class ElementSequence(list):
    """
    Base class for composite optical elements (like ThickLens, Slab).
    It behaves like a list of elements but supports fluent configuration via the aliases parameter.
    """
    def __init__(self, elements: list, aliases: Dict[str, List[Tuple[int, str]]] = None):
        super().__init__(elements)
        self.aliases = aliases or {}

    def variable(self, *args):
        # set everything variable if no args
        if not args:
            for el in self:
                if isinstance(el, OpticalElement): el.variable()
            return self

        for name in args:
            # see what name maps to
            if name in self.aliases:
                for idx, internal_name in self.aliases[name]:
                    self[idx].variable(internal_name)
            
            # otherwise, try to set on all elements that have it
            else:
                for el in self:
                    if name in el.param_names:
                        el.variable(name)
        return self

    def fixed(self):
        for el in self:
            if isinstance(el, OpticalElement): el.fixed()
        return self
    
    def __repr__(self):
        # Nice string representation: ThickLens(3 elements)
        return f"{self.__class__.__name__}(len={len(self)})"