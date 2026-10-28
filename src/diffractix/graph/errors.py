class ASTError(Exception):
    """Base exception for AST traversal and compilation errors."""


class ASTCycleError(ASTError):
    """Raised when a cycle is found in the AST."""

class UnresolvedInputError(ASTError):
    """Raised when an empty InputNode is encountered."""


class UnsupportedNodeError(ASTError):
    """Raised when an unknown Node subclass is encountered."""