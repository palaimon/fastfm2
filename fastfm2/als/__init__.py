try:
    from . import regression
    from .regression import FMRegression
except ImportError:
    print("NO REGRESSION MODULE EXISTS!")
    pass

try:
    from . import classification
    from .classification import FMClassification
except ImportError:
    print("NO CLASSIFICATION MODULE EXISTS!")
    pass
