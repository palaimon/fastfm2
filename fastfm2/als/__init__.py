try:
    from .regression import FMRegression  # noqa: F401
except ImportError:
    print("NO REGRESSION MODULE EXISTS!")
    pass

try:
    from .classification import FMClassification  # noqa: F401
except ImportError:
    print("NO CLASSIFICATION MODULE EXISTS!")
    pass
