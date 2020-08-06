
#loss functions for classifier
from torch import nn
import torch.nn.functional as F
ALPHA = 2.0
BETA = 10000.5
GAMMA = 10

class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        print(CE_loss)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
#         inputs=inputs.argmax(dim=1)
        #flatten label and prediction tensors
        y_pred=inputs
        y_true=targets
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        TP=tp
        FP=fp
        TN=tn
        FN=fn
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky.mean()
    
class F1_loss_soft_0(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(fbeta_loss_soft_0, self).__init__()

    def forward(self, inputs, targets, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True,**kwarg):
        
        beta2 = beta ** 2
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
#         inputs=(inputs[:,0]).float()
#         targets=targets.float()
        y_pred=inputs
        y_true=targets
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        y_true=y_true[:,0]
        y_pred=y_pred[:,0]
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2* (precision*recall) / (precision + recall + eps)
        f1 = f1.clamp(min=eps, max=1-eps)
       

        return 1 - (f1.mean())
class F1_loss_soft_all(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(fbeta_loss_soft_all, self).__init__()

    def forward(self, inputs, targets, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True,**kwarg):
        
        beta2 = beta ** 2
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
#         inputs=(inputs[:,0]).float()
#         targets=targets.float()
        y_pred=inputs
        y_true=targets
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
       
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2* (precision*recall) / (precision + recall + eps)
        f1 = f1.clamp(min=eps, max=1-eps)


        return 1 - (f1.mean())    
# a variation of the F1 loss function that tries to focus more on the postive samples, in practice it ended up biasing the model to predicting postively on everything 
class F1_loss_sig_0(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(fbeta_loss_sig_0, self).__init__()

    def forward(self, inputs, targets, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True,**kwarg):
        
        beta2 = beta ** 2
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
#         inputs=(inputs[:,0]).float()
#         targets=targets.float()
        y_pred=inputs
        y_true=targets
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.sigmoid(y_pred)
        y_true=y_true[:,0]
        y_pred=y_pred[:,0]
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2* (precision*recall) / (precision + recall + eps)
        f1 = f1.clamp(min=eps, max=1-eps)
       

        return 1 - (f1.mean())        
class F1_loss_sig_all(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(fbeta_loss_sig_all, self).__init__()

    def forward(self, inputs, targets, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True,**kwarg):
        
        beta2 = beta ** 2
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
#         inputs=(inputs[:,0]).float()
#         targets=targets.float()
        y_pred=inputs
        y_true=targets
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.sigmoid(y_pred)
      
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2* (precision*recall) / (precision + recall + eps)
        f1 = f1.clamp(min=eps, max=1-eps)       

        return 1 - (f1.mean())  
