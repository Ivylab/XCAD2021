import os
from PIL import Image
import torchvision.transforms as transforms
from ops import *
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetGenerator(Dataset):
    def __init__(
        self, pathImageDirectory, pathDatasetFile, transform, gan=False, gray=False
    ):

        self.listImagePaths = []
        self.listImageLabels = []
        self.listImageBoxs = []
        self.listBoxLabels = []
        self.transform = transform
        self.gan = gan
        self.gray = gray

        fileDescriptor = open(pathDatasetFile, "r")

        line = True

        while line:
            line = fileDescriptor.readline()

            if line:
                lineItems = line.split()

                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:15]
                imageLabel = [int(i) for i in imageLabel]

                bBox = lineItems[15:19]
                bBox = [float(i) for i in bBox]

                bBoxLabel = lineItems[19:]
                bBoxLabel = [float(i) for i in bBoxLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
                self.listImageBoxs.append(bBox)
                self.listBoxLabels.append(bBoxLabel)

        fileDescriptor.close()

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]
        if self.gray:
            imageData = Image.open(imagePath)  # .convert('L')
        else:
            imageData = Image.open(imagePath).convert("RGB")

        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        imageBox = torch.FloatTensor(self.listImageBoxs[index])
        boxLabel = torch.FloatTensor(self.listBoxLabels[index])

        if self.transform != None:
            imageData = self.transform(imageData)

        if self.gan:
            imageData = imageData * 2 - 1

        if self.gray:
            if imageData.size(0) == 1:
                pass
            else:
                imageData = imageData[0, :, :].unsqueeze(0)

        return imageData, imageLabel, imagePath, imageBox, boxLabel

    def __len__(self):

        return len(self.listImagePaths)


def get_xai_data_loader(
    img_dir, path_list, batch_size, shuffle=False, gan=False, gray=False
):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transformList = []
    transformList.append(transforms.Resize(224))
    transformList.append(transforms.ToTensor())
    if not gan:
        transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)

    datasetTest = DatasetGenerator(
        pathImageDirectory=img_dir,
        pathDatasetFile=path_list,
        transform=transformSequence,
        gan=gan,
        gray=gray,
    )
    loader = DataLoader(
        dataset=datasetTest,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle,
        pin_memory=True,
    )

    return loader
