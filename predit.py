import pandas as pd
import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import HandWriteDataSet
from model import DigitsMobilenet
from config import Config
""" 这个代码是使用了dataloader，在做测试的时候一个一个batch的进行测试，速度比直接读取图片快多了"""



def predicts(model_path, dataset):
    config = Config()
    test_loader = DataLoader(dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=False)
    model = DigitsMobilenet(config.class_num).cuda()
    model.load_state_dict(t.load(model_path)['model'])
    print('Load model from %s successfully' % model_path)
    data_frame = []
    tbar = tqdm(test_loader)
    model.eval()
    model.cuda()
    with t.no_grad():
        for i, (img, img_name) in enumerate(tbar):
            """ img, img_name是一个batch的"""
            img = img.cuda()
            pred = model(img)
            # print('i',i)
            # print('img_name',img_name)
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21 = pred
            for eachimgindex,(ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21) in enumerate(zip(
                    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14,
                    p15, p16, p17, p18, p19, p20, p21)):
                ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21 = ch1.argmax(
                    0), ch2.argmax(0), ch3.argmax(0), ch4.argmax(0), ch5.argmax(
                        0), ch6.argmax(0), ch7.argmax(0), ch8.argmax(0), ch9.argmax(
                            0), ch10.argmax(0), ch11.argmax(0), ch12.argmax(
                                0), ch13.argmax(0), ch14.argmax(0), ch15.argmax(
                                    0), ch16.argmax(0), ch17.argmax(0), ch18.argmax(
                                        0), ch19.argmax(0), ch20.argmax(
                                            0), ch21.argmax(0)
                index_result = [
                    ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12,
                    ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21
                ]
                temporary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                result = ''
                for i in index_result:
                    result += temporary[i]
                result = remove_z(result)
                print('{}预测的结果是{}'.format(img_name[eachimgindex],result))
                data_frame += [[int(img_name[eachimgindex]), result]]
    data_frame = sorted(data_frame, key=lambda x: x[0])
    return data_frame


def remove_z(str_a):
    if str_a[-1] == 'Z':
        str_a = str_a[:-1]
        return remove_z(str_a)
    return str_a


def write2csv(results):
    """
    results(list):
    """
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    save_name = './handWriting/pred_epoch-3_acc-55.55.csv'
    df.to_csv(save_name, sep=',', index=False, header=False)
    print('Results.saved to %s' % save_name)


if __name__ == "__main__":
    t.cuda.empty_cache()
    t.cuda.set_device(0)
    model_path = './checkpoints/epoch-147_acc-78.60.pth'
    test_dataset = HandWriteDataSet("./handWriting/test/*.jpg",
                                    train_flag=False,
                                    test_flag=True)
    result_list = predicts(model_path, test_dataset)
    write2csv(result_list)
