
import os,random,numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def reset_random(seed=32):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

class HTDLELCD(nn.Module):
    def __init__(self,num_classes=1,pretrained=True,hidden_dim=256,num_heads=8,transformer_layers=2,gru_layers=1,dropout=0.3):
        super().__init__()
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone=models.resnet18(weights=weights)
        self.feature_extractor=nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim=512
        self.projection=nn.Conv2d(self.feature_dim,hidden_dim,kernel_size=1)
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=num_heads,dim_feedforward=hidden_dim*4,dropout=dropout,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=transformer_layers)
        self.bigru=nn.GRU(input_size=hidden_dim,hidden_size=hidden_dim//2,num_layers=gru_layers,batch_first=True,bidirectional=True,dropout=dropout if gru_layers>1 else 0.0)
        self.attention=nn.Sequential(nn.Linear(hidden_dim,hidden_dim//2),nn.Tanh(),nn.Linear(hidden_dim//2,1))
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden_dim,num_classes)
        self.gradcam_layer=self.feature_extractor[-1][-1].conv2
    def forward(self,x,return_attention=False):
        features=self.feature_extractor(x)
        projected=self.projection(features)
        sequence=projected.flatten(2).transpose(1,2)
        encoded=self.transformer(sequence)
        contextual,_=self.bigru(encoded)
        attention_weights=torch.softmax(self.attention(contextual).squeeze(-1),dim=1)
        pooled=torch.sum(contextual*attention_weights.unsqueeze(-1),dim=1)
        logits=self.classifier(self.dropout(pooled))
        if return_attention:
            return logits,attention_weights
        return logits
    def predict_proba(self,x):
        return torch.sigmoid(self.forward(x))
    def predict(self,x,threshold=0.5):
        return (self.predict_proba(x)>=threshold).long()

class GradCAM:
    def __init__(self,model,target_layer=None):
        self.model=model
        self.target_layer=target_layer if target_layer is not None else model.gradcam_layer
        self.activations=None
        self.gradients=None
        self.forward_handle=self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle=self.target_layer.register_full_backward_hook(self._save_gradient)
    def _save_activation(self,module,inputs,output):
        self.activations=output
    def _save_gradient(self,module,grad_input,grad_output):
        self.gradients=grad_output[0]
    def generate(self,images,class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits=self.model(images)
        if class_idx is None:
            score=logits[:,0].sum()
        else:
            score=logits[:,class_idx].sum()
        score.backward()
        weights=self.gradients.mean(dim=(2,3),keepdim=True)
        cam=(weights*self.activations).sum(dim=1,keepdim=True)
        cam=F.relu(cam)
        cam=F.interpolate(cam,size=images.shape[-2:],mode="bilinear",align_corners=False)
        minimum=cam.amin(dim=(2,3),keepdim=True)
        maximum=cam.amax(dim=(2,3),keepdim=True)
        return (cam-minimum)/(maximum-minimum+1e-8)
    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

def build_model(pretrained=True):
    return HTDLELCD(num_classes=1,pretrained=pretrained)

if __name__=="__main__":
    reset_random(32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(pretrained=False).to(device)
    sample=torch.randn(2,3,224,224,device=device)
    print("Output shape:",model(sample).shape)
    print("Device:",device)
