import torch
import torch.nn as nn


class conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, padding_size):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class deconv(nn.Module):
    def __init__(self, in_ch, double=True):
        super(deconv, self).__init__()
        if double:
            out = in_ch // 2
        else:
            out = in_ch

        self.up = nn.ConvTranspose2d(
            in_ch, out, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        x = self.up(x)
        return x


class downconv(nn.Module):
    def __init__(self, in_ch, double=True):
        super(downconv, self).__init__()
        if double:
            out_ch = in_ch * 2
        else:
            out_ch = in_ch
        self.dwcov = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.dwcov(x)
        return x


class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()

        # Encoder
        self.conv1 = conv_bn_relu(in_ch=1, out_ch=64, padding_size=1)
        self.conv1_1 = conv_bn_relu(in_ch=64, out_ch=64, padding_size=1)
        self.max1 = downconv(in_ch=64)

        self.conv2 = conv_bn_relu(in_ch=128, out_ch=128, padding_size=1)
        self.conv2_1 = conv_bn_relu(in_ch=128, out_ch=128, padding_size=1)
        self.max2 = downconv(in_ch=128)

        self.conv3 = conv_bn_relu(in_ch=256, out_ch=256, padding_size=1)
        self.conv3_1 = conv_bn_relu(in_ch=256, out_ch=256, padding_size=1)
        self.max3 = downconv(in_ch=256)

        self.conv4 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)
        self.conv4_1 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)
        self.max4 = downconv(in_ch=512, double=False)

        self.conv5 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)
        self.conv5_1 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)
        self.max5 = downconv(in_ch=512, double=False)

        self.gap = nn.AvgPool2d([7, 7])

        self.linear = nn.Sequential(nn.Linear(512 + 14, 7 * 7 * 512), nn.LeakyReLU(0.2))

        # Decoder
        self.up1 = deconv(in_ch=512, double=False)
        self.conv6 = conv_bn_relu(in_ch=1024, out_ch=512, padding_size=1)
        self.conv6_1 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)

        self.up2 = deconv(in_ch=512, double=False)
        self.conv7 = conv_bn_relu(in_ch=1024, out_ch=512, padding_size=1)
        self.conv7_1 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)

        self.up3 = deconv(in_ch=512)
        self.conv8 = conv_bn_relu(in_ch=512, out_ch=256, padding_size=1)
        self.conv8_1 = conv_bn_relu(in_ch=256, out_ch=256, padding_size=1)

        self.up4 = deconv(in_ch=256)
        self.conv9 = conv_bn_relu(in_ch=256, out_ch=128, padding_size=1)
        self.conv9_1 = conv_bn_relu(in_ch=128, out_ch=128, padding_size=1)

        self.up5 = deconv(in_ch=128)
        self.conv10 = conv_bn_relu(in_ch=128, out_ch=64, padding_size=1)
        self.conv10_1 = conv_bn_relu(in_ch=64, out_ch=64, padding_size=1)

        self.out_conv = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, padding=1
        )

    def forward(self, x, condition_z):
        # Encoder
        x1 = self.conv1(x)
        x1_1 = self.conv1_1(x1)
        max1 = self.max1(x1_1)

        x2 = self.conv2(max1)
        x2_1 = self.conv2_1(x2)
        max2 = self.max2(x2_1)

        x3 = self.conv3(max2)
        x3_1 = self.conv3_1(x3)
        max3 = self.max3(x3_1)

        x4 = self.conv4(max3)
        x4_1 = self.conv4_1(x4)
        max4 = self.max4(x4_1)

        x5 = self.conv5(max4)
        x5_1 = self.conv5_1(x5)
        max5 = self.max5(x5_1)

        # Concat Latent Vectors
        latent_z = self.gap(max5)

        combined_z = torch.cat([latent_z.squeeze(-1).squeeze(-1), condition_z], -1)

        input_z = self.linear(combined_z)
        input_z = input_z.view(max5.size())

        # Decoder
        up_x5_1 = self.up1(input_z)
        concat1 = torch.cat([up_x5_1, x5_1], dim=1)
        x6 = self.conv6(concat1)
        x6_1 = self.conv6_1(x6)

        up_x6_1 = self.up2(x6_1)
        concat2 = torch.cat([up_x6_1, x4_1], dim=1)
        x7 = self.conv7(concat2)
        x7_1 = self.conv7_1(x7)

        up_x7_1 = self.up3(x7_1)
        concat3 = torch.cat([up_x7_1, x3_1], dim=1)
        x8 = self.conv8(concat3)
        x8_1 = self.conv8_1(x8)

        up_x8_1 = self.up4(x8_1)
        concat4 = torch.cat([up_x8_1, x2_1], dim=1)
        x9 = self.conv9(concat4)
        x9_1 = self.conv9_1(x9)

        up_x9_1 = self.up5(x9_1)
        concat5 = torch.cat([up_x9_1, x1_1], dim=1)
        x10 = self.conv10(concat5)
        x10_1 = self.conv10_1(x10)

        output = self.out_conv(x10_1)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = conv_bn_relu(in_ch=1, out_ch=64, padding_size=1)
        self.conv1_1 = conv_bn_relu(in_ch=64, out_ch=64, padding_size=1)
        self.max1 = downconv(in_ch=64)

        self.conv2 = conv_bn_relu(in_ch=128, out_ch=128, padding_size=1)
        self.conv2_1 = conv_bn_relu(in_ch=128, out_ch=128, padding_size=1)
        self.max2 = downconv(in_ch=128)

        self.conv3 = conv_bn_relu(in_ch=256, out_ch=256, padding_size=1)
        self.conv3_1 = conv_bn_relu(in_ch=256, out_ch=256, padding_size=1)
        self.max3 = downconv(in_ch=256)

        self.conv4 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)
        self.conv4_1 = conv_bn_relu(in_ch=512, out_ch=512, padding_size=1)
        self.max4 = downconv(in_ch=512)

        self.conv5 = conv_bn_relu(in_ch=1024, out_ch=1024, padding_size=1)
        self.conv5_1 = conv_bn_relu(in_ch=1024, out_ch=1024, padding_size=1)
        self.max5 = downconv(in_ch=1024, double=False)

        self.avg_pool = nn.AvgPool2d(7)

        # Discriminator
        self.fc = nn.Sequential(
            nn.Linear(1024 + 14, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x, y):
        # Encoder
        x1 = self.conv1(x)
        x1_1 = self.conv1_1(x1)
        max1 = self.max1(x1_1)

        x2 = self.conv2(max1)
        x2_1 = self.conv2_1(x2)
        max2 = self.max2(x2_1)

        x3 = self.conv3(max2)
        x3_1 = self.conv3_1(x3)
        max3 = self.max3(x3_1)

        x4 = self.conv4(max3)
        x4_1 = self.conv4_1(x4)
        max4 = self.max4(x4_1)

        x5 = self.conv5(max4)
        x5_1 = self.conv5_1(x5)
        max5 = self.max5(x5_1)

        avg = self.avg_pool(max5).squeeze(3).squeeze(2)

        # Discriminator
        d_out = self.fc(torch.cat((avg, y), dim=1))

        return d_out
