
import os,cv2,numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SOURCE_DIR="Data/source"
OUTPUT_DIR="Data/preprocessed"
IMAGE_SIZE=(224,224)
ADF_ITERATIONS=10
ADF_KAPPA=30
ADF_GAMMA=0.15
ACLAHE_BASE_CLIP=2.0

def anisotropic_diffusion(image,iterations=ADF_ITERATIONS,kappa=ADF_KAPPA,gamma=ADF_GAMMA):
    image=image.astype(np.float32)
    for _ in range(iterations):
        north=np.roll(image,-1,axis=0)-image
        south=np.roll(image,1,axis=0)-image
        east=np.roll(image,-1,axis=1)-image
        west=np.roll(image,1,axis=1)-image
        cn=np.exp(-(north/kappa)**2)
        cs=np.exp(-(south/kappa)**2)
        ce=np.exp(-(east/kappa)**2)
        cw=np.exp(-(west/kappa)**2)
        image+=gamma*(cn*north+cs*south+ce*east+cw*west)
        image[0]=image[1]
        image[-1]=image[-2]
        image[:,0]=image[:,1]
        image[:,-1]=image[:,-2]
    return np.clip(image,0,255).astype(np.uint8)

def adaptive_clahe(image):
    lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    luminance,a,b=cv2.split(lab)
    contrast=float(np.std(luminance))
    clip_limit=float(np.clip(ACLAHE_BASE_CLIP*(contrast/64.0),1.0,4.0))
    clahe=cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=(8,8))
    luminance=clahe.apply(luminance)
    return cv2.cvtColor(cv2.merge((luminance,a,b)),cv2.COLOR_LAB2RGB)

def minmax_normalize(image):
    image=image.astype(np.float32)
    minimum=float(image.min())
    maximum=float(image.max())
    if maximum>minimum:
        image=(image-minimum)/(maximum-minimum)
    else:
        image=np.zeros_like(image,dtype=np.float32)
    return np.round(image*255).astype(np.uint8)

def preprocess_image(image_path):
    image=cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,IMAGE_SIZE,interpolation=cv2.INTER_AREA)
    image=anisotropic_diffusion(image)
    image=adaptive_clahe(image)
    image=minmax_normalize(image)
    return image

def get_train_transform():
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomAffine(degrees=15,translate=(0.08,0.08),scale=(0.9,1.1)),transforms.ToTensor()])

def get_eval_transform():
    return transforms.ToTensor()

def main():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {SOURCE_DIR}")
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    supported_extensions=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    classes=sorted(name for name in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR,name)))
    if not classes:
        raise ValueError(f"No class folders found in {SOURCE_DIR}")
    summary={}
    for class_name in classes:
        source_class=os.path.join(SOURCE_DIR,class_name)
        output_class=os.path.join(OUTPUT_DIR,class_name)
        os.makedirs(output_class,exist_ok=True)
        files=sorted(name for name in os.listdir(source_class) if name.lower().endswith(supported_extensions))
        count=0
        for filename in tqdm(files,desc=f"Preprocessing {class_name}"):
            try:
                image=preprocess_image(os.path.join(source_class,filename))
                output_path=os.path.join(output_class,os.path.splitext(filename)[0]+".png")
                if not cv2.imwrite(output_path,cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Unable to save image: {output_path}")
                count+=1
            except (ValueError,OSError,cv2.error) as error:
                print(f"Skipped {filename}: {error}")
        summary[class_name]=count
    with open(os.path.join(OUTPUT_DIR,"preprocess_summary.json"),"w",encoding="utf-8") as file:
        json.dump(summary,file,indent=2)
    print(f"Preprocessing completed: {OUTPUT_DIR}")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
