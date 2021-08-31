import os
import torch
import random
import numpy as np
from torch.autograd import Variable
from PIL import Image, ImageDraw
import matplotlib.cm as cmx
import cv2
import math
import os
import torch

class_name = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def conditionZ(oneHotPrediction, originalPrediction):
    vectorZ = []

    for i in range(oneHotPrediction.size(0)):
        classNum = oneHotPrediction[i].nonzero().nelement()

        if classNum == 0:
            vectorZ.append(originalPrediction[i].unsqueeze(0))

        elif classNum == 1:
            index = oneHotPrediction[i].nonzero().item()
            originalPrediction[i][index] = 0.0
            vectorZ.append(originalPrediction[i].unsqueeze(0))

        else:
            multiIndex = oneHotPrediction[i].nonzero().view(-1)
            selectIndex = np.random.permutation(multiIndex.cpu().numpy())
            selectNum = random.choice(np.arange(len(multiIndex) + 1))

            originalPrediction[i][
                torch.from_numpy(selectIndex[: selectNum + 1]).long()
            ] = 0.0
            vectorZ.append(originalPrediction[i].unsqueeze(0))

            oneHotPrediction[i][
                torch.from_numpy(selectIndex[: selectNum + 1]).long()
            ] = 0.0

    return torch.cat(vectorZ, dim=0)


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def gan_weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv2d") != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("BatchNorm") != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    elif classname.find("Linear") != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.zeros(m.bias.data.size())


def preprocess_image(img, device):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img_tensor = torch.from_numpy(preprocessed_img).to(device)

    preprocessed_img_tensor.unsqueeze_(0)

    return Variable(preprocessed_img_tensor, requires_grad=False)


def numpy_to_torch(img, device, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output).to(device)

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def visualization(
    saliency,
    root,
    bboxCoordinate,
    bboxLabel,
    index,
    saveDir,
):

    # Define Variables
    boxWidth = 13
    saliencyMin = np.mean(saliency) * 4
    saliencyMax = np.max(saliency)

    # Create Blended Image
    saliency = np.uint8(
        cmx.jet((saliency - saliencyMin) / (saliencyMax - saliencyMin)) * 255
    )[:, :, 0:3]
    originalImg = cv2.cvtColor(cv2.imread(root), cv2.COLOR_BGR2RGB)
    blendedImg = Image.fromarray(np.uint8(originalImg * 0.4 + saliency * 0.6))

    # Draw Bounding Box
    bboxDraw = ImageDraw.Draw(blendedImg)
    bboxDraw.rectangle(
        [
            (round(bboxCoordinate[0].item()), round(bboxCoordinate[1].item())),
            (
                round((bboxCoordinate[0] + bboxCoordinate[2]).item()),
                round((bboxCoordinate[1] + bboxCoordinate[3]).item()),
            ),
        ],
        outline="red",
        width=boxWidth,
    )

    # Output Visualization Results
    blendedImg.save(
        saveDir
        + "/"
        + "CheXGAN_Visualization_%s_%3d.png" % (class_name[bboxLabel.item()], index)
    )


class SimpleToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x)


def tensor_centercrop(img, keepdim=True):
    img = img.squeeze()
    num_ceil = math.ceil(img.size(-1) * 0.11)
    num_floor = math.floor(img.size(-1) * 0.11)

    img[:num_ceil, :] = 0.0
    img[:, :num_ceil] = 0.0

    img[img.size(-1) - num_floor :, :] = 0.0
    img[:, img.size(-1) - num_floor :] = 0.0

    return img
