import torch.nn as nn
import torch

from .sleepyco import SleePyCoBackbone
from .classifiers import get_classifier

class MainModel(nn.Module):
    """
    Single-modality model (S2S).
    This model uses a single feature extraction path and a classifier.
    It's a simplified version of the M2M model's single branch.
    """
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.cfg = config
        
        self.feature_extractor = SleePyCoBackbone(self.cfg)
        self.classifier = get_classifier(config)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        """
        Forward pass for the S2S model.
        Args:
            x (torch.Tensor): Input tensor for the single modality (time-domain data).
            labels (torch.Tensor, optional): Ground truth labels. Defaults to None.

        Returns:
            - if labels are provided: (logits, loss)
            - if labels are not provided: (logits, None)
        """
        features = self.feature_extractor(x)
        transposed_features = features[0].transpose(1, 2)
        logits, _, _ = self.classifier(transposed_features)
        loss = self.criterion(logits, labels)
        return logits, loss
        
