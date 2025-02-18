"""Init."""

try:
    from ims_utils.batchnorm.combat import ComBat, combat
    from ims_utils.batchnorm.utilities import combine_arrays, create_batches

    HAS_COMBAT = True
except ImportError:
    HAS_COMBAT = False

    def ComBat(*args, **kwargs):
        return print("ComBat not available - please install pandas or ims-utils[batch]")

    def combat(*args, **kwargs):
        return print("ComBat not available - please install pandas or ims-utils[batch]")

    def combine_arrays(*args, **kwargs):
        return print("ComBat not available - please install pandas or ims-utils[batch]")

    def create_batches(*args, **kwargs):
        return print("ComBat not available - please install pandas or ims-utils[batch]")


try:
    from ims_utils.batchnorm.recombat import ReComBat

    HAS_RECOMBAT = True
except ImportError:
    HAS_RECOMBAT = False

    def ReComBat(*args, **kwargs):
        return print("ReComBat not available - please install patsy, scikit-learn or  or ims-utils[batch]")
