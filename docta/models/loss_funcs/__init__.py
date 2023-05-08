from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .accuracy import Accuracy, accuracy
from .peer_loss import PeerLoss
from .loss_utils import reduce_loss, weight_reduce_loss, weighted_loss
from .loss_correction import ForwardLoss, BackwardLoss