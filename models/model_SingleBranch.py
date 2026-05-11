import torch.nn as nn
import torch.nn.functional as F
import torch
from .sleepyco import SleePyCoBackbone
from .classifiers import get_classifier

class SingleBranchModel(nn.Module):
    """
    Single Branch Model for evaluating individual branch performance.
    Can evaluate either 'single' or 'multi' branch from M2S model.
    """
    
    def __init__(self, config, branch_type='single'):
        """
        Args:
            config: configuration dictionary
            branch_type: 'single' or 'multi' to specify which branch to use
        """
        super(SingleBranchModel, self).__init__()

        self.cfg = config
        self.training_mode = config['training_params']['mode']
        self.branch_type = branch_type  # 'single' or 'multi'
        
        # 获取单一模态（列表中的第一个）
        self.modalities = config.get('modalities', [])
        assert len(self.modalities) == 1, "SingleBranch model only supports single modality"
        self.modality = self.modalities[0]
        
        self.num_classes = config['classifier']['num_classes']

        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            # Feature extractor
            self.feature_extractor = SleePyCoBackbone(self.cfg)
            # Classifier
            self.classifier = get_classifier(config)
            # 新增：简单线性分类头
            self.linear_head = nn.Linear(
                self.cfg['classifier']['model_dim'],
                self.num_classes
            )

    def forward(self, x, labels=None):
        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            # Input for the specific modality
            modal_input = x[f'{self.modality}_time']
            
            # Feature extraction
            features = self.feature_extractor(modal_input)
            
            # Classification
            _, _, pooled_feature = self.classifier(features[0].transpose(1, 2))
            
            # 新增：线性层分类头
            cls_output = self.linear_head(pooled_feature)
            
            # Calculate loss
            loss = F.cross_entropy(cls_output, labels)
            
            if self.training:
                return cls_output, loss
            else:
                return cls_output, loss
        else:
            raise NotImplementedError(f"Training mode '{self.training_mode}' is not implemented.")