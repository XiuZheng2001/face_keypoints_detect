# Author：xzheng
# Date：2023.4.24
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as t
from torchvision.transforms import ToTensor
import yaml
from backbone import Model

with open('cfgs.yaml', 'r') as file:
    cfgs = yaml.safe_load(file)

data = np.load(cfgs['DATA_CONFIG']['TRAIN'], allow_pickle=True)
test = np.load(cfgs['DATA_CONFIG']['TEST'], allow_pickle=True)
img_train = np.float32(data['images'])  # the shape of this dataset is (1425, 256, 256, 3)
pts_train = np.float32(data['points'])  # (1425, 44, 2)

img_test = np.float32(test['images'])
# Default float in Numpy is float64, you must convert the Numpy tensor to np.float32 before converting it to Pytorch.
#           pic1     pic2     pic3 ....
# loc1_x
# loc1_y
# loc2_x
# loc2_y
# ....
# (1425, 256, 256, 3)->(1425, 3, 256, 256)的步骤, 在创建dataloader时会自动转换
pts_train = np.reshape(pts_train, (1425, -1))  # (1425,44,2)->(1425, 88)


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pts_train
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        image = img_train[idx, ...]  # numpy(256, 256, 3)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        landmarks = self.landmarks_frame[idx, ...]  # numpy(88, )
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample



training_dataset = FaceLandmarksDataset(
    transform=t.Compose([
        ToTensor(), t.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225], )
    ]))
training_dataloader = DataLoader(training_dataset, cfgs['TRAIN_CONFIG']['BATCH_SIZE'],
                                 shuffle=True, num_workers=0)

model = Model()  # output:(N, 88)
loss_fn_name = cfgs['TRAIN_CONFIG']['LOSS_FN']
loss_fn = getattr(nn, loss_fn_name)()  # get specific attribute value from a object
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train_one_epoch():
    running_loss = 0.

    for i, sample in enumerate(training_dataloader):
        # Every sample is batch size input + label pair, type of sample: dict
        inputs, labels = sample['image'], sample['landmarks']
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients, and backward propogation
        loss = loss_fn(outputs, labels)
        loss.backward()
        # 控制梯度的范数在一个合适的范围内，防止梯度爆炸问题。通过对梯度进行裁剪，可以保证梯度在一个较小的范围内，使得权重更新更稳定。
        # 这里的梯度阈值为5，是指梯度范数大于这个阈值时，那么梯度就会被等比例缩小，以使范数等于阈值。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / len(training_dataset)


def training(EPOCHS=cfgs['TRAIN_CONFIG']['EPOCH']):
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()
        print('AVG LOSS on TRAINING DATA: {}'.format(avg_loss))


def save_as_csv(points, location='.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
                    and no_test_points == 554 in this project
    :param location: Directory to save results.csv in. Default to current working directory
    """
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')


def visualise_pts(img, pts):
    """
    将（H，W，C）的numpy图像可视化
    :param img: numpy array in the shape of (H, W, C)
    :param pts: points in the array of (1, 88)
    :return: none
    """
    plt.imshow(np.uint8(img))  # 在显示图像时，一定要将浮点数转为整数，不然会显示一片空白或其他异常情况
    odd_pts = pts[:, ::2]
    even_pts = pts[:, 1::2]
    pt = np.vstack((odd_pts, even_pts))
    plt.plot(pt[0], pt[1], '+r', ms=7)  # （44， 2）
    plt.show()


def test_one_image(img, test_model):
    """
    :param img: numpy array in the shape of (H, W, C) on CPU
    :return: none
    """
    test_model.to('cpu')

    test_img = img.transpose(2, 0, 1)
    test_img = np.float32(test_img)
    test_img = torch.from_numpy(test_img)
    test_img = test_img[None, :, :, :]  # add a dimension to torch tensor

    out = test_model(test_img).detach().numpy()  # remove gradient and convert to numpy
    visualise_pts(img, out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The model is training on GPU: {}'.format(torch.cuda.is_available()))
    model.to(device)

    # train the model and save checkpoint
    training()
    torch.save(model.state_dict(), 'paras.pt')
    # load the model
    model.load_state_dict(torch.load('paras.pt'))
    # test one image and visualize it
    test_one_image(img_test[0], model)
