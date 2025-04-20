from torchvision import transforms
from torchvision.transforms.functional import crop

def crop_google_logo(img):
    return crop(img, top=0, left=0, height=img.size[1] - 25, width=img.size[0])  # D裁剪google logo底部25个像素

# 定义一种transforms --对应的是AlexNet
transform_cnn = transforms.Compose([
    transforms.Lambda(crop_google_logo),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# 改进后的transform--对应的是AlexNet
transform_cnn_2 = transforms.Compose([
    transforms.Lambda(crop_google_logo),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计量
                         std=[0.229, 0.224, 0.225])
])

# TODO 添加其他基础网络的transform