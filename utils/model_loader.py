import torch
import os
from models.eeg_models import SSBCINet
from models.image_models import RSSCNN, SSCNN


class ModelLoader:
    def __init__(self, base_path="../outputs"):
        # 获取当前文件的绝对路径，然后构建相对于项目根目录的路径
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)  # 回到项目根目录

        if base_path.startswith("../"):
            self.base_path = os.path.join(project_root, base_path[3:])
        else:
            self.base_path = os.path.join(project_root, base_path)

        # 直接使用正确的路径结构
        self.image_model_path = os.path.join(self.base_path, "outputs_images", "models")
        self.eeg_model_path = os.path.join(self.base_path, "outputs_eeg")

        print(f"🔍 ModelLoader 初始化:")
        print(f"  基础路径: {self.base_path}")
        print(f"  EEG路径: {self.eeg_model_path}")
        print(f"  图像路径: {self.image_model_path}")

    def load_eeg_model(self, model_name="EEGNet", in_chans=64, n_classes=2, input_window_samples=2000):
        """
        加载EEG预训练模型
        """
        # 构建模型路径
        model_path = os.path.join(self.eeg_model_path, model_name, "best_model.pth")

        print(f"🔍 加载EEG模型:")
        print(f"  查找路径: {model_path}")
        print(f"  文件存在: {os.path.exists(model_path)}")

        # 检查目录内容
        model_dir = os.path.join(self.eeg_model_path, model_name)
        if os.path.exists(model_dir):
            print(f"  EEG目录内容: {os.listdir(model_dir)}")

        if not os.path.exists(model_path):
            print(f"⚠️ EEG预训练模型不存在: {model_path}")
            return None

        try:
            # 创建模型结构
            model = SSBCINet(
                base_model_name=model_name,
                in_chans=in_chans,
                n_classes=n_classes,
                input_window_samples=input_window_samples
            )

            # 加载权重
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"  Checkpoint键: {checkpoint.keys()}")

            # 根据checkpoint的结构加载权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # 尝试直接加载
                model.load_state_dict(checkpoint)

            print(f"✅ 成功加载EEG预训练模型: {model_path}")
            return model

        except Exception as e:
            print(f"❌ 加载EEG预训练模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_image_model(self, model_type="rsscnn", base_model_name="PlacesNet"):
        """
        加载图像预训练模型
        """
        if model_type == "rsscnn":
            model_class = RSSCNN
            filename = f"best_rsscnn_{base_model_name}.pth"
        elif model_type == "sscnn":
            model_class = SSCNN
            filename = f"best_sscnn_{base_model_name}.pth"
        else:
            raise ValueError("model_type 必须是 'rsscnn' 或 'sscnn'")

        model_path = os.path.join(self.image_model_path, model_type, filename)

        print(f"🔍 加载图像模型:")
        print(f"  查找路径: {model_path}")
        print(f"  文件存在: {os.path.exists(model_path)}")

        # 检查目录内容
        type_dir = os.path.join(self.image_model_path, model_type)
        if os.path.exists(type_dir):
            print(f"  图像目录内容: {os.listdir(type_dir)}")

        if not os.path.exists(model_path):
            print(f"⚠️ 图像预训练模型不存在: {model_path}")
            return None

        try:
            # 创建模型结构
            model = model_class(base_model_name=base_model_name)

            # 加载权重
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"  Checkpoint键: {checkpoint.keys()}")

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print(f"✅ 成功加载图像预训练模型: {model_path}")
            return model

        except Exception as e:
            print(f"❌ 加载图像预训练模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_available_models(self):
        """获取可用的预训练模型列表"""
        available = {
            "eeg_models": [],
            "image_models": {
                "rsscnn": [],
                "sscnn": []
            }
        }

        # 检查EEG模型
        for model_name in ["EEGNet", "ShallowFBCSPNet"]:
            model_path = os.path.join(self.eeg_model_path, model_name, "best_model.pth")
            if os.path.exists(model_path):
                available["eeg_models"].append(model_name)

        # 检查图像模型
        for model_type in ["rsscnn", "sscnn"]:
            type_path = os.path.join(self.image_model_path, model_type)
            if os.path.exists(type_path):
                for base_model in ["AlexNet", "VGG", "PlacesNet"]:
                    filename = f"best_{model_type}_{base_model}.pth"
                    model_path = os.path.join(type_path, filename)
                    if os.path.exists(model_path):
                        available["image_models"][model_type].append(base_model)

        return available