
import os,json,random,numpy as np,torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from model import HTDLELCD,reset_random

DATA_DIR="Data/preprocessed"
MODEL_DIR="model"
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
SEED=32
EARLY_STOPPING_PATIENCE=8
MIN_DELTA=1e-4  
NUM_WORKERS=0

class LungCTDataset(Dataset):
    def __init__(self,root,files,labels,training=False):
        self.root=root
        self.files=files
        self.labels=labels
        self.transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomAffine(degrees=15,translate=(0.08,0.08),scale=(0.9,1.1)),transforms.ToTensor()]) if training else transforms.ToTensor()
    def __len__(self):
        return len(self.files)
    def __getitem__(self,index):
        image=Image.open(self.files[index]).convert("RGB")
        image=self.transform(image)
        label=torch.tensor([self.labels[index]],dtype=torch.float32)
        return image,label

def collect_images(data_dir):
    classes=sorted(name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,name)))
    if len(classes)!=2:
        raise ValueError(f"Expected exactly 2 class folders, found {classes}")
    files=[]
    labels=[]
    extensions=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    for label,class_name in enumerate(classes):
        class_dir=os.path.join(data_dir,class_name)
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(extensions):
                files.append(os.path.join(class_dir,filename))
                labels.append(label)
    if not files:
        raise ValueError(f"No images found in {data_dir}")
    if len(set(labels))!=2:
        raise ValueError("Both classes must contain images")
    return np.asarray(files),np.asarray(labels),classes

def run_epoch(model,loader,criterion,optimizer,device):
    training=optimizer is not None
    model.train(training)
    total_loss=0.0
    total_correct=0
    total=0
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for images,labels in loader:
            images=images.to(device)
            labels=labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=criterion(logits,labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size=labels.size(0)
            total_loss+=loss.item()*batch_size
            predictions=(torch.sigmoid(logits)>=0.5).float()
            total_correct+=(predictions==labels).sum().item()
            total+=batch_size
    return total_loss/max(total,1),total_correct/max(total,1)

def main():
    reset_random(SEED)
    random.seed(SEED)
    os.makedirs(MODEL_DIR,exist_ok=True)
    files,labels,classes=collect_images(DATA_DIR)
    indices=np.arange(len(files))
    trainval_idx,test_idx=train_test_split(indices,test_size=0.30,random_state=SEED,stratify=labels)
    train_idx,val_idx=train_test_split(trainval_idx,test_size=0.10,random_state=SEED,stratify=labels[trainval_idx])
    train_data=LungCTDataset(DATA_DIR,files[train_idx],labels[train_idx],training=True)
    train_eval_data=LungCTDataset(DATA_DIR,files[train_idx],labels[train_idx],training=False)
    val_data=LungCTDataset(DATA_DIR,files[val_idx],labels[val_idx],training=False)
    test_data=LungCTDataset(DATA_DIR,files[test_idx],labels[test_idx],training=False)
    train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
    train_eval_loader=DataLoader(train_eval_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
    test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=HTDLELCD(num_classes=1,pretrained=True).to(device)
    criterion=nn.BCEWithLogitsLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3,min_lr=1e-6)
    history={"train_loss":[],"val_loss":[],"train_accuracy":[],"val_accuracy":[],"learning_rate":[]}
    best_val_loss=float("inf")
    best_epoch=0
    patience_counter=0
    for epoch in range(EPOCHS):
        train_loss,_=run_epoch(model,train_loader,criterion,optimizer,device)
        val_loss,val_accuracy=run_epoch(model,val_loader,criterion,None,device)
        scheduler.step(val_loss)
        train_loss_eval,train_accuracy=run_epoch(model,train_eval_loader,criterion,None,device)
        history["train_loss"].append(train_loss_eval)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss_eval:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.6g}")
        if val_loss<best_val_loss-MIN_DELTA:
            best_val_loss=val_loss
            best_epoch=epoch+1
            patience_counter=0
            torch.save({"model_state_dict":model.state_dict(),"classes":classes,"best_epoch":best_epoch,"best_val_loss":best_val_loss,"seed":SEED},os.path.join(MODEL_DIR,"best_model.pth"))
        else:
            patience_counter+=1
        with open(os.path.join(MODEL_DIR,"history.json"),"w",encoding="utf-8") as file:
            json.dump({**history,"best_epoch":best_epoch,"best_val_loss":best_val_loss,"classes":classes},file,indent=2)
        if patience_counter>=EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}; best epoch: {best_epoch}")
            break
    checkpoint_path=os.path.join(MODEL_DIR,"best_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError("Training ended without a saved best checkpoint")
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss,test_accuracy=run_epoch(model,test_loader,criterion,None,device)
    with open(os.path.join(MODEL_DIR,"training_summary.json"),"w",encoding="utf-8") as file:
        json.dump({"best_epoch":best_epoch,"best_val_loss":best_val_loss,"test_loss":test_loss,"test_accuracy":test_accuracy,"train_samples":len(train_idx),"validation_samples":len(val_idx),"test_samples":len(test_idx),"classes":classes},file,indent=2)
    print(f"Training completed | Best Epoch: {best_epoch} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(f"Saved checkpoint and history in: {MODEL_DIR}")

if __name__=="__main__":
    main()
