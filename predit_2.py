import glob

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import DigitsMobilenet

""" 这个代码没有使用dataloader,直接从磁盘一个一个图片读取进行判断。速度慢，远远没有使用dataloader的快。"""


def predict(input, model, device):
    model.to(device)
    with torch.no_grad():
        input = input.to(device)
        out = model(input)
        ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21 = out
        print(ch1)
        ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21 = ch1.argmax(
            1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1), ch5.argmax(
                1), ch6.argmax(1), ch7.argmax(1), ch8.argmax(1), ch9.argmax(
                    1), ch10.argmax(1), ch11.argmax(1), ch12.argmax(
                        1), ch13.argmax(1), ch14.argmax(1), ch15.argmax(
                            1), ch16.argmax(1), ch17.argmax(1), ch18.argmax(
                                1), ch19.argmax(1), ch20.argmax(1), ch21.argmax(
                                    1)
        index_result = [
            ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13,
            ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21
        ]
        temporary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        result = ''
        for i in index_result:
            result += temporary[i]
        result = remove_z(result)
        # return ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21
        return result


def remove_z(str_a):
    if str_a[-1] == 'Z':
        str_a = str_a[:-1]
        return remove_z(str_a)
    return str_a


if __name__ == "__main__":
    model_path = './checkpoints/epoch-17_acc-32.27.pth'
    model = DigitsMobilenet(class_num=26).cuda()
    model.load_state_dict(torch.load(model_path)['model'])
    print('Load model from %s successfully' % model_path)
    model.eval()

    test_images_dir = "./handWriting/test/*.jpg"
    images = glob.glob(test_images_dir)
    images_sorted = sorted(images, key=lambda x: int(x.split('.')[1][18:]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_tf = transforms.Compose([
        transforms.Resize((280, 50)),
        transforms.ToTensor(),
    ])

    filename_list = []
    label_pred_list = []
    for i in images_sorted:
        img = Image.open(i)
        img = test_tf(img).unsqueeze(0)
        ans = predict(img, model, device)
        print("{}的预测结果是{}".format(i, ans))
        filename = i.split('.')[1][18:]
        filename_list.append(filename)
        label_pred_list.append(ans)
    data = list(zip(filename_list, label_pred_list))
    df = pd.DataFrame(data=data, columns=['filename', 'label_nums'])
    # df['label'] = df['label_nums'].apply(lambda x: status_dict[x])
    # df.drop('label_nums', axis=1, inplace=True)
    df.to_csv('./handWriting/pred.csv', index=False, header=False)
