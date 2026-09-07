from enum import Enum



class Relation(Enum):
    EQ = "=="
    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"


class Comparison:
    """Symbolic relation between two graph expressions."""

    def __init__(self, relation: Relation, left: "Node", right: "Node"):
        self.relation = relation
        self.left = left
        self.right = right

    def __bool__(self):
        raise TypeError("Symbolic comparisons cannot be used as booleans.")

    def __repr__(self):
        return (
            f"{self.left} {self.relation.value} {self.right}"
        )