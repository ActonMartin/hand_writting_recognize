import glob
import re
import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


class HandWriteDataSet(Dataset):

    def __init__(self, data_path, train_flag=True, test_flag=False):
        """
        在训练的时候data_path是csv文件的路径，在进行预测的时候data_path是test文件夹的路径
        """
        self.train_flag = train_flag
        self.test_flag = test_flag
        if self.test_flag:
            self.test_images = glob.glob(data_path)
            self.test_images_sorted = sorted(
                self.test_images, key=lambda x: int(x.split('.')[1][18:]))
            self.test_tf = transforms.Compose([
                transforms.Resize((280, 50)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.imgs_info = self.getimages(data_path)
            self.train_tf_by_albumentation = A.Compose([
                A.Resize(height=50, width=280),
                # A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            self.train_tf = transforms.Compose([
                transforms.Resize((280, 50)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.val_tf = transforms.Compose([
                transforms.Resize((280, 50)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def getimages(self, csv_path):
        train_df = pd.read_csv(csv_path)
        train_df.dropna(axis=0, how='any', inplace=True)  # 去掉空值
        files = train_df.filename
        label_conver2upper = train_df.label.str.upper()
        labels = label_conver2upper.tolist()
        images_path = ["./handWriting/train/" + str(i) + '.jpg' for i in files]
        images_info = dict(zip(images_path, labels))
        images_info = list(images_info.items())
        return images_info

    def padding_black(self, img):
        w, h = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg,
                     ((size_bg - size_fg[0]) // 2, (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):
        if self.test_flag:
            img_path = self.test_images_sorted[index]
            image_name = img_path.split('.')[1][18:]
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = self.padding_black(img)
            img = self.test_tf(img)
            return img, image_name
        else:
            img_path, label = self.imgs_info[index]
            # 使用pillow读取图片
            img = Image.open(img_path)
            # img = img.convert('RGB')
            img = np.array(img)
            # 使用opencv读取图片
            # img = cv2.imread(img_path)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # img = self.padding_black(img)
            if self.train_flag:
                img = self.train_tf_by_albumentation(image=img)['image']
                # img = self.train_tf(img)
            else:
                img = self.val_tf(img)
            # 标签中出现了空格，-，和单引号,`，这些去掉
            label = re.sub(r'[\'\`\-\s]*', '', label)

            if len(label) >= 21:
                label = list(label[:21])
            else:
                label = list(label[:20]) + (21 - len(label)) * ['Z']
            label_ascii = []
            for i in label:
                label_ascii.append(ord(i) - 65)
            label_ascii = torch.tensor(label_ascii)
            # label = label.squeeze(0)
            return img, label_ascii

    def __len__(self):
        if self.test_flag:
            return len(self.test_images)
        else:
            return len(self.imgs_info)


if __name__ == "__main__":
    train_dataset = HandWriteDataSet("./handWriting/train.csv",
                                     train_flag=True,
                                     test_flag=False)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=False)
    count = 0
    flag = False
    for image, label in train_loader:
        # print(image.shape)
        count += 1
        # if count > 1000:
        #     break
        if count % 10 == 0:
            print('count_where', count)
            print(image.shape)
            print(label.squeeze(0))
            print(len(label[0]))
        #检查有没有小于21个长度的
        # if len(label[0]) < 21:
        #     print(label)
        #     break
        # 检查label有没有大于25的
        # for i in range(21):
        #     if label[:, i] > 25:
        #         # print(image.shape)
        #         flag = True
        #         print(label)
        #         break
        # if flag:
        #     break
        #检查label有没有小于0的
        # for i in range(21):
        #     if label[:,i] < 0:
        #         print('count',count)
        #         print(image.shape)
        #         print(label)
        #         break
