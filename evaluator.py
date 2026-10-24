
import os,json,numpy as np,torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from model import HTDLELCD,reset_random

DATA_DIR="Data/preprocessed"
MODEL_DIR="model"
BATCH_SIZE=32
SEED=32
NUM_WORKERS=0

class LungCTDataset(Dataset):
    def __init__(self,files,labels):
        self.files=files
        self.labels=labels
        self.transform=transforms.ToTensor()
    def __len__(self):
        return len(self.files)
    def __getitem__(self,index):
        image=Image.open(self.files[index]).convert("RGB")
        return self.transform(image),torch.tensor([self.labels[index]],dtype=torch.float32)

def collect_images(data_dir,classes):
    files=[]
    labels=[]
    extensions=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    for label,class_name in enumerate(classes):
        class_dir=os.path.join(data_dir,class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(extensions):
                files.append(os.path.join(class_dir,filename))
                labels.append(label)
    if not files:
        raise ValueError(f"No images found in {data_dir}")
    return np.asarray(files),np.asarray(labels)

def predict_dataset(model,loader,device):
    model.eval()
    y_true=[]
    y_prob=[]
    with torch.no_grad():
        for images,labels in loader:
            logits=model(images.to(device))
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            y_true.extend(labels.numpy().ravel().astype(int))
    y_true=np.asarray(y_true)
    y_prob=np.asarray(y_prob)
    y_pred=(y_prob>=0.5).astype(int)
    return y_true,y_pred,y_prob

def calculate_metrics(y_true,y_pred,y_prob):
    metrics={"accuracy":accuracy_score(y_true,y_pred),"precision":precision_score(y_true,y_pred,zero_division=0),"recall":recall_score(y_true,y_pred,zero_division=0),"f1_score":f1_score(y_true,y_pred,zero_division=0),"mcc":matthews_corrcoef(y_true,y_pred)}
    if len(np.unique(y_true))==2:
        metrics["roc_auc"]=roc_auc_score(y_true,y_prob)
    else:
        metrics["roc_auc"]=None
    return metrics

def plot_confusion_matrix(y_true,y_pred,classes,filename):
    matrix=confusion_matrix(y_true,y_pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(6,5))
    image=ax.imshow(matrix,cmap="Blues")
    ax.set_xticks([0,1],labels=classes)
    ax.set_yticks([0,1],labels=classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center")
    fig.colorbar(image,ax=ax)
    fig.tight_layout()
    fig.savefig(filename,dpi=300)
    plt.close(fig)

def plot_roc(y_true,y_prob,filename):
    if len(np.unique(y_true))<2:
        print("ROC curve skipped: target contains only one class")
        return
    fpr,tpr,_=roc_curve(y_true,y_prob)
    auc=roc_auc_score(y_true,y_prob)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr,label=f"ROC AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],"--",label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename,dpi=300)
    plt.close(fig)

def plot_history(history):
    epochs=range(1,len(history["train_loss"])+1)
    for key,title,ylabel,filename in [("loss","Training and Validation Loss","Loss","loss_curve.png"),("accuracy","Training and Validation Accuracy","Accuracy","accuracy_curve.png")]:
        fig,ax=plt.subplots(figsize=(7,5))
        if key=="loss":
            ax.plot(epochs,history["train_loss"],label="Train Loss")
            ax.plot(epochs,history["val_loss"],label="Validation Loss")
        else:
            ax.plot(epochs,history["train_accuracy"],label="Train Accuracy")
            ax.plot(epochs,history["val_accuracy"],label="Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(MODEL_DIR,filename),dpi=300)
        plt.close(fig)

def main():
    reset_random(SEED)
    os.makedirs(MODEL_DIR,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path=os.path.join(MODEL_DIR,"best_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    classes=checkpoint["classes"]
    files,labels=collect_images(DATA_DIR,classes)
    indices=np.arange(len(files))
    trainval_idx,test_idx=train_test_split(indices,test_size=0.30,random_state=SEED,stratify=labels)
    train_idx,val_idx=train_test_split(trainval_idx,test_size=0.10,random_state=SEED,stratify=labels[trainval_idx])
    train_loader=DataLoader(LungCTDataset(files[train_idx],labels[train_idx]),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    test_loader=DataLoader(LungCTDataset(files[test_idx],labels[test_idx]),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    model=HTDLELCD(num_classes=1,pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    results={}
    predictions={}
    for name,loader in [("train",train_loader),("test",test_loader)]:
        y_true,y_pred,y_prob=predict_dataset(model,loader,device)
        metrics=calculate_metrics(y_true,y_pred,y_prob)
        results[name]=metrics
        predictions[name]=(y_true,y_pred,y_prob)
        plot_confusion_matrix(y_true,y_pred,classes,os.path.join(MODEL_DIR,f"confusion_matrix_{name}.png"))
        plot_roc(y_true,y_prob,os.path.join(MODEL_DIR,f"roc_curve_{name}.png"))
        print(f"\n{name.upper()} METRICS")
        for metric,value in metrics.items():
            print(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")
    with open(os.path.join(MODEL_DIR,"history.json"),"r",encoding="utf-8") as file:
        history=json.load(file)
    plot_history(history)
    with open(os.path.join(MODEL_DIR,"metrics.json"),"w",encoding="utf-8") as file:
        json.dump(results,file,indent=2)
    print(f"\nEvaluation results and plots saved in: {MODEL_DIR}")

if __name__=="__main__":
    main()
