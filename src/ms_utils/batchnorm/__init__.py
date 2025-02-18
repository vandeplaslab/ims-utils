"""Init."""

try:
    from ms_utils.batchnorm.combat import ComBat, combat
    from ms_utils.batchnorm.utilities import combine_arrays, create_batches

    HAS_COMBAT = True
except ImportError:
    HAS_COMBAT = False

    def ComBat(*args, **kwargs):
        return print("ComBat not available - please install pandas or ms-utils[batch]")

    def combat(*args, **kwargs):
        return print("ComBat not available - please install pandas or ms-utils[batch]")

    def combine_arrays(*args, **kwargs):
        return print("ComBat not available - please install pandas or ms-utils[batch]")

    def create_batches(*args, **kwargs):
        return print("ComBat not available - please install pandas or ms-utils[batch]")


try:
    from ms_utils.batchnorm.recombat import ReComBat

    HAS_RECOMBAT = True
except ImportError:
    HAS_RECOMBAT = False

    def ReComBat(*args, **kwargs):
        return print("ReComBat not available - please install patsy, scikit-learn or  or ms-utils[batch]")
