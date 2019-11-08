import gc
import torch
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

import constants
from model.center_net import MyUNet,criterion
from car_dataset import CarDataset

train = pd.read_csv(constants.PATH + '/train.csv')
test = pd.read_csv(constants.PATH + '/sample_submission.csv')
train_images_dir = constants.PATH + 'train_images/{}.jpg'
test_images_dir = constants.PATH + 'test_images/{}.jpg'

df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
df_test = test

# Create dataset objects
# Sets up a way for model to load the data
train_dataset = CarDataset(df_train, train_images_dir, training=True)
dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
test_dataset = CarDataset(df_test, test_images_dir, training=False)

# batch size is limited by GPU memory
BATCH_SIZE = 1
# loads data
# num_workers specifies the number of CPU cores that load data
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
gc.collect()

# learn.destroy() 
n_epochs = 15

model = MyUNet(8).to(device)
# print("Model is", model)
summary(model, (3,480,1536))
optimizer = optim.Adam(model.parameters(), lr=0.001)
# to degrade the learning rate as time progresses
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

def train_model(epoch, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        # print("Image batch shape is ",img_batch.shape)
        # print("Mask batch shape is ",mask_batch.shape)
        # print("regr batch shape is ",regr_batch.shape)
        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        # baackward prop the loss to calculate gradient
        loss.backward()
        # Update weights
        optimizer.step()
        # update learning rate ( decrease it )
        exp_lr_scheduler.step()
    
    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss.data))

def evaluate_model(epoch, history=None):
    model.eval()
    loss = 0
    # use no grad to tell model not to perform back prop calc optimizations in forward pass to save time
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss += criterion(output, mask_batch, regr_batch, size_average=False).data
    
    loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    print('Dev loss: {:.4f}'.format(loss))

if __name__ == "__main__":

    history = pd.DataFrame()
    
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train_model(epoch, history)
        evaluate_model(epoch, history)

    torch.save(model.state_dict(), './ckpt/model.pth')
    history['train_loss'].iloc[100:].plot()
