import torch.nn as nn
import torch.nn.functional as F
import torch
from .sleepyco import SleePyCoBackbone
from .classifiers import get_classifier
from .VIB import VIB

class M2SModel(nn.Module):
    """
    Multi-modal to Single-modal (M2S) Model.
    Extracts a single modality's weights from a jointly trained multi-modal model 
    for single-modal fine-tuning.
    """
    
    def __init__(self, config):
        super(M2SModel, self).__init__()

        self.cfg = config
        self.training_mode = config['training_params']['mode']
        
        # The single modality to be used for this model, e.g., 'eeg_fpzcz'
        self.modality = config['modality']
        self.num_classes = config['classifier']['num_classes']

        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            # Single-path network for the modality
            self.feature_single = SleePyCoBackbone(self.cfg)
            self.classifier_single = get_classifier(config)
            
            # Multi-path network for the modality (weights loaded from joint training)
            self.feature_multi = SleePyCoBackbone(self.cfg)
            self.classifier_multi = get_classifier(config)
            
            # ===== 新增：为 single 与 multi 前置各自的 VIB（与 M2M 保持一致） =====
            model_dim = config['classifier']['model_dim']
            self.vib_single_branch = VIB(d_x=model_dim, d_z=model_dim, n_classes=self.num_classes)
            self.vib_multi_branch = VIB(d_x=model_dim, d_z=model_dim, n_classes=self.num_classes)
            # ==================================================================

            # VIB after concatenation (same as in M2M model)
            # Input: model_dim (single) + model_dim (multi) = 2 * model_dim
            self.combined_vib = VIB(d_x=model_dim * 2, 
                                   d_z=model_dim, 
                                   n_classes=self.num_classes)
            
            # Final classifier after VIB
            self.final_classifier = nn.Linear(model_dim, self.num_classes)

    def forward(self, x, labels=None):
        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            # Input for the specific modality, e.g., x['eeg_fpzcz_time']
            modal_input = x[f'{self.modality}_time']
            
            # ===== Single-path processing =====
            single_features = self.feature_single(modal_input)
            # Transpose to [Batch, SeqLen, Channels] for the classifier
            _, _, single_pooled = self.classifier_single(single_features[0].transpose(1, 2))
            
            # ===== Multi-path processing =====
            multi_features = self.feature_multi(modal_input)
            _, _, multi_pooled = self.classifier_multi(multi_features[0].transpose(1, 2))
            
            # ===== 新增：对 single / multi pooled 各自做 VIB（前置） =====
            single_token, vib_loss_single, _ = self.vib_single_branch(single_pooled, labels)
            multi_token, vib_loss_multi, _ = self.vib_multi_branch(multi_pooled, labels)
            # ================================================================
            
            # ===== Feature combination =====
            # Concatenate features from both paths (use VIB tokens)
            combined_feature = torch.cat([single_token, multi_token], dim=1)
            
            # ===== VIB processing (after concat) =====
            final_token, vib_loss_after_concat, _ = self.combined_vib(combined_feature, labels)
            
            # ===== Final classification =====
            cls_output = self.final_classifier(final_token)
            classification_loss = F.cross_entropy(cls_output, labels)
            
            # Total loss includes classification and all VIB losses
            total_loss = classification_loss + vib_loss_single.mean() + vib_loss_multi.mean() + vib_loss_after_concat.mean()
            
            if self.training:
                return cls_output, total_loss
            else:
                return cls_output, classification_loss
        else:
            raise NotImplementedError(f"Training mode '{self.training_mode}' is not implemented for M2S model.")
