import torch
import torch.nn as nn
from braindecode.models import EEGNetv1, ShallowFBCSPNet, EEGNetv4


class EEGFeatureExtractor(nn.Module):
    def __init__(self, model_name="EEGNet", in_chans=64, n_classes=2, input_window_samples=2000):
        super(EEGFeatureExtractor, self).__init__()

        if model_name == "EEGNet":
            full_model = EEGNetv1(
                n_classes=n_classes,
                in_chans=in_chans,
                input_window_samples=input_window_samples,
                final_conv_length="auto"
            )
            # 截取所有层直到conv_classifier之前的部分（drop_3 之后为 classifier）
            self.feature_net = nn.Sequential(
                full_model.ensuredims,
                full_model.conv_1,
                full_model.bnorm_1,
                full_model.elu_1,
                full_model.permute_1,
                full_model.drop_1,
                full_model.conv_2,
                full_model.bnorm_2,
                full_model.elu_2,
                full_model.pool_2,
                full_model.drop_2,
                full_model.conv_3,
                full_model.bnorm_3,
                full_model.elu_3,
                full_model.pool_3,
                full_model.drop_3,
            )
            # 这里可以使用 dummy forward 计算输出维度
            dummy_input = torch.randn(1, in_chans, input_window_samples)
            with torch.no_grad():
                out = self.feature_net(dummy_input)
                self.out_dim = out.view(out.size(0), -1).shape[1]

        elif model_name == "ShallowFBCSPNet":
            full_model = ShallowFBCSPNet(
                n_classes=n_classes,
                in_chans=in_chans,
                input_window_samples=input_window_samples,
                final_conv_length="auto"
            )
            # 截取所有层直到conv_classifier之前的部分（drop 之后为 classifier）
            self.feature_net = nn.Sequential(
                full_model.ensuredims,
                full_model.dimshuffle,
                full_model.conv_time,
                full_model.conv_spat,
                full_model.bnorm,
                full_model.conv_nonlin_exp,
                full_model.pool,
                full_model.pool_nonlin_exp,
                full_model.drop
            )
            # 这里可以使用 dummy forward 计算输出维度
            dummy_input = torch.randn(1, in_chans, input_window_samples)
            with torch.no_grad():
                out = self.feature_net(dummy_input)
                self.out_dim = out.view(out.size(0), -1).shape[1]

        else:
            raise ValueError("Unsupported EEG model. Choose 'EEGNet' or 'ShallowFBCSPNet'.")

    def forward(self, x):
        out = self.feature_net(x)
        return out.view(out.size(0), -1)  # 展平为 [B, out_dim]



class EEGFusionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(EEGFusionNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512) # 因为ImageModel的输出是512，所以这里也设置为512
        )

    def forward(self, f1, f2):
        fused = torch.cat([f1, f2], dim=1)
        return self.classifier(fused)


# 整个SSBCI模型都是后续多模态模型的特征提取器
class SSBCINet(nn.Module):
    def __init__(self, base_model_name="EEGNet", in_chans=64, n_classes=2, input_window_samples=2000):
        super(SSBCINet, self).__init__()
        self.feature_extractor = EEGFeatureExtractor(
            model_name=base_model_name,
            in_chans=in_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
        )
        self.fusion = EEGFusionNetwork(self.feature_extractor.out_dim)
        # 初始化权重
        self._initialize_weights(mean=0.0, variance=0.01)

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        return self.fusion(f1, f2)

    def _initialize_weights(self, mean=0.0, variance=0.01):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=mean, std=variance)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                # 初始化 BatchNorm 层的权重和偏置
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # 测试模型
    # model = ShallowFBCSPNet(input_window_samples=200, in_chans=64, n_classes=2)
    # print(model)
    model = SSBCINet(base_model_name="ShallowFBCSPNet", in_chans=64, n_classes=2, input_window_samples=2000)
    print(model)
    x1 = torch.randn(8, 64, 2000)  # 假设 EEG 信号的形状为 [batch_size, channels, time]
    x2 = torch.randn(8, 64, 2000)
    output = model(x1, x2)
    print(output.shape)  # 输出形状应该是 [8, 512]
