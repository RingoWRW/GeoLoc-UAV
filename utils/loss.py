import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SupervisedContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.device = device
    
    def forward(self, image_feature, labels):

        dot_product = torch.mm(image_feature, image_feature.T) / self.temperature
        exp_dot_product = torch.exp(dot_product - torch.max(dot_product, dim=1, keepdim=True)[0]) + 1e-5

        mask_similar_class = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels).to(self.device)
        mask_anchor_out = (1 - torch.eye(exp_dot_product.shape[0])).to(self.device)
        mask_combined = mask_similar_class * mask_anchor_out
        per_sample = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_product / (torch.sum(exp_dot_product * mask_anchor_out, dim=1, keepdim=True)))
        supervised_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / per_sample
        supervised_loss = torch.mean(supervised_loss_per_sample)

        return supervised_loss

class WeightedInfoNCE(nn.Module):
    def __init__(self, label_smoothing, k=-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
        self.k = k

    def loss(self, similarity_matrix, eps_all):
        n = similarity_matrix.shape[0]
        total_loss = 0.0
        for i in range(n):
            eps = eps_all[i]
            total_loss += (1 - eps) * (-1. * similarity_matrix[i, i] + torch.logsumexp(similarity_matrix[i, :], dim=0))
            total_loss += eps * (-1. / n * similarity_matrix[i, :].sum() + torch.logsumexp(similarity_matrix[i, :], dim=0))
        total_loss /= n
        return total_loss

    def forward(self, image_features1, image_features2, logit_scale, positive_weights=None):
        # Normalize the image features
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # Compute similarity logits
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        # Apply positive weights if provided
        if positive_weights is not None:
            eps = 1. - (1. - self.label_smoothing) / (1 + torch.exp(-self.k * positive_weights))
        else:
            eps = [self.label_smoothing for _ in range(image_features1.shape[0])]
        
        logits_per_image2 = logits_per_image1.T
        
        # Generate labels
        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss1 = self.loss(logits_per_image1, eps)
        loss2 = self.loss(logits_per_image2, eps)
        # # Compute loss
        # loss1 = self.loss_function(logits_per_image1, labels)
        # loss2 = self.loss_function(logits_per_image2, labels)
        loss = (loss1 + loss2) / 2

        return loss