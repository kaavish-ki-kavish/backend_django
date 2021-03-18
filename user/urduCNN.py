import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Urdu_CNN(nn.Module):
    def __init__(self):
        super(Urdu_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 40)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class UrduCnnScorer:
    def __init__(self, x, y):
        file = open(os.path.join(__location__, 'urdu_model_state_dict_blur.pt'), 'rb')
        # Load
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            map_location = 'cpu' #lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        checkpoint = torch.load(file, map_location=map_location)
        # checkpoint = torch.load(file)
        self.model = Urdu_CNN().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        self.x = x
        self.y = y
        self.NUM2LABEL = ['alif', 'alif mad aa', 'ayn', 'baa', 'bari yaa', 'cheey', 'choti yaa', 'daal', 'dhaal', 'faa',
                          'gaaf', 'ghain', 'haa1', 'haa2', 'haa3', 'hamza', 'jeem', 'kaaf', 'khaa', 'laam', 'meem',
                          'noon', 'noonghunna', 'paa', 'qaaf', 'raa', 'rhraa', 'seen', 'seey', 'sheen', 'swaad', 'taa',
                          'ttaa', 'twa', 'waw', 'zaaa', 'zaal', 'zhaa', 'zwaa', 'zwaad']

    def preprocessing(self):
        x_0, y_0 = min(self.x), min(self.y)
        x_n, y_n = max(self.x), max(self.y)
        w = x_n - x_0 + 1
        h = y_n - y_0 + 1
        padding = 20
        x_origin = [x_cord + padding//2 - x_0 for x_cord in self.x]
        y_origin = [y_cord + padding//2 - y_0 for y_cord in self.y]
        x_y = list(zip(x_origin, y_origin))

        im = Image.new('RGB', (w + padding, h + padding), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        draw.line(x_y)
        img_array = np.array(im)
        plt.imshow(img_array[:, :, 0])
        plt.axis('off')
        plt.show()
        return img_array[:, :, 0]

    def crop_image(self, array):
        ret3, img = cv.threshold(array, 200, 255, cv.THRESH_BINARY)  # +cv.THRESH_OTSU)

        # unique_elements, counts_elements = np.unique(img, return_counts=True)

        desired_size = 256
        # h, w = img.shape

        y, x = np.where(img == 0)
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()

        img = img[y_min: y_max, x_min: x_max]

        old_size = img.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format
        img = cv.resize(img, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 0, 0]
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT,
                                value=color)

        return img

    def test_img(self, array):
        img = self.crop_image(array)
        img = 255 - img
        img1 = cv.GaussianBlur(img, (3, 3), 0)
        img1 = cv.resize(img1, (28, 28), cv.INTER_AREA)
        ret3, img1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY)  # +cv.THRESH_OTSU)
        unique_elements, counts_elements = np.unique(img1, return_counts=True)
        img = cv.resize(img, (28, 28), cv.INTER_AREA)
        ret3, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY)  # +cv.THRESH_OTSU)
        unique_elements, counts_elements = np.unique(img, return_counts=True)
        # print(sorted(list(zip(unique_elements, counts_elements))))

        data_transform_main = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        data_transform_custom = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((28, 28)),
                                                    data_transform_main])

        crop_img = data_transform_custom(img.astype(np.float32))
        # crop_img1 = data_transform_custom(img1.astype(np.float32))

        self.model.eval()
        with torch.no_grad():
            crop_img = crop_img.view(1, 1, 28, 28)
            crop_img_p = crop_img.view(28, 28)
            crop_img = crop_img.to(self.device)

            #
            # crop_img1 = crop_img1.view(1, 1, 28, 28)
            # crop_img_p1 = crop_img1.view(28, 28)
            # crop_img1 = crop_img1.to(self.device)

            output = self.model(crop_img)
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(output)
            top3p, top3l = torch.topk(probabilities, 3)
            return self.NUM2LABEL[output.max(1)[1]], top3p.tolist()[0][0]

            # print([NUM2LABEL[i] for i in top3l.tolist()[0]])
            # predicted = output.max(1)[1]
            # plt.imshow(crop_img_p.cpu(), 'gray')
            # plt.title(f"{NUM2LABEL[predicted]}: {top3p.tolist()[0][0]}")
            # plt.show()

            # output = self.model(crop_img1)
            # sm = torch.nn.Softmax(dim=1)
            # probabilities = sm(output)
            # top3p, top3l = torch.topk(probabilities, 3)

            # print(top3l.tolist())
            # print(top3p.tolist())
            # print([NUM2LABEL[i] for i in top3l.tolist()[0]])
            predicted = output.max(1)[1]
            # plt.imshow(  crop_img_p1.cpu(), 'gray')
            # plt.title(f"{NUM2LABEL[predicted]}: {top3p.tolist()[0][0]}")
            # plt.show()


    def get_score(self):
        pre = self.preprocessing()
        score = self.test_img(pre)
        return score


# x = [106, 79, 61, 43, 25, 14, 3, 0, 1, 10, 23, 42, 65, 83, 99, 121, 139, 157, 174, 209, 221, 233, 251, 255, 255, 254,
#      248, 226, 201, 163, 109, 107, 104, 118, 139, 160, 162, 159, 139, 124, 117]
# y = [89, 73, 71, 82, 102, 121, 150, 168, 195, 219, 231, 241, 247, 245, 238, 213, 225, 230, 233, 231, 225, 213, 181, 165,
#      143, 130, 118, 100, 89, 83, 87, 85, 62, 33, 10, 0, 5, 12, 34, 57, 86]

# x = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# y = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# y = [50, 45, 40, 35, 30, 25, 15, 10, 5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# x = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
# y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# #
# urdu_scorer = UrduCnnScorer(x, y)
# score = urdu_scorer.get_score()
# print(score)

