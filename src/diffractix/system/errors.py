


class SystemValidationError(ValueError):
    """Raised by `System.build` when the system is invalid, listing every problem found.

    Attributes
    ----------
    errors : tuple of str
        One message per problem.
    """

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        message = "System validation failed:\n" + "\n".join(
            f"  {i}. {error}" for i, error in enumerate(errors, start=1)
        )
        super().__init__(message)
