"""Init."""

from ms_utils.batchnorm.combat import ComBat, combat
from ms_utils.batchnorm.utilities import combine_arrays, create_batches

try:
    from ms_utils.batchnorm.recombat import ReComBat

    HAS_RECOMBAT = True
except ImportError:
    HAS_RECOMBAT = False

    def ReComBat(*args, **kwargs):
        return print("ReComBat not available - please install scikit-learn")
