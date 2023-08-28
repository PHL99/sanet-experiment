import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.linalg import norm
from PIL import Image
from net import channel_norm


class ContentStyleDataset(Dataset):
    def __init__(self, root_dir, transform=None, shuffle=True):
        content = glob.glob(os.path.join(root_dir, "content/*"))
        style = glob.glob(os.path.join(root_dir, "style/*"))
        self.glob_list = np.array(list(zip(content, style)))
        if shuffle:
            np.random.shuffle(self.glob_list)
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.glob_list)
    
    def __getitem__(self, idx):
        content_img = self.transform(Image.open(self.glob_list[idx, 0]))
        style_img = self.transform(Image.open(self.glob_list[idx, 1]))
        return (content_img, style_img)
    

class STLoss(nn.Module):
    def __init__(self, lmbda_c=1, lmbda_s=3, lmbda_i1=30, lmbda_i2=1):
        super(STLoss, self).__init__()
        self.lmbda_c = lmbda_c
        self.lmbda_s = lmbda_s
        self.lmbda_i1 = lmbda_i1
        self.lmbda_i2 = lmbda_i2

    def forward(self, output_map, content_map, style_map, content_identity_map, style_identity_map,
                content, style, content_identity, style_identity):        
        # Content loss
        #content_loss = norm(channel_norm(output_map['relu4_1']) - channel_norm(content_map['relu4_1']))\
        #      + norm(channel_norm(output_map['relu5_1']) - channel_norm(content_map['relu5_1']))
        content_loss = F.mse_loss(channel_norm(output_map['relu4_1']), channel_norm(content_map['relu4_1']))\
              + F.mse_loss(channel_norm(output_map['relu5_1']), channel_norm(content_map['relu5_1']))
        

        # Style loss
        style_loss = 0.
        for layer in output_map.keys():
            # Mean/Variance (AdaIn) loss
        #    style_loss += norm(output_map[layer].mean(dim=(-1, -2)) - style_map[layer].mean(dim=(-1, -2)))\
        #            + norm(output_map[layer].std(dim=(-1, -2)) - style_map[layer].std(dim=(-1, -2)))
            style_loss += F.mse_loss(output_map[layer].mean(dim=(-1, -2)), style_map[layer].mean(dim=(-1, -2)))\
                    + F.mse_loss(output_map[layer].std(dim=(-1, -2)), style_map[layer].std(dim=(-1, -2)))

        # Identity loss
        i1_loss = norm(content_identity - content) + norm(style_identity - style)
        i2_loss = 0.
        for layer in output_map.keys():
            # Per layer feature loss
        #    i2_loss += norm(content_identity_map[layer] - content_map[layer])\
        #        + norm(style_identity_map[layer] - style_map[layer])
            i2_loss += F.mse_loss(content_identity_map[layer], content_map[layer])\
                + F.mse_loss(style_identity_map[layer], style_map[layer])
        identity_loss = self.lmbda_i1 * i1_loss + self.lmbda_i2 * i2_loss
        
        # Total Loss
        return self.lmbda_c * content_loss + self.lmbda_s * style_loss + identity_loss
    

def train_and_save(st_model, optimizer, loss_func, end_epoch, train_loader, save_path, load_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    st_model.to(device)
    params = sum(p.numel() for p in st_model.parameters() if p.requires_grad)
    print("Trainable parameters:", params)

    start_epoch = 0
    train_loss = torch.zeros(end_epoch-start_epoch)
    if load_path:
        checkpoint = torch.load(load_path)
        st_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_loss = torch.cat([checkpoint["loss"], train_loss])
        print("Load path successful")

    num_batch = len(train_loader)
    st_model.train()
    print("Begin training from epoch {} to epoch {}...\n".format(start_epoch+1, end_epoch))
    for epoch in range(start_epoch, end_epoch):
        start_time = time.time()
        batch_loss = 0.
        for i, (content, style) in enumerate(train_loader):
            content, style = content.to(device), style.to(device)
            optimizer.zero_grad()

            output, (content_map, style_map) = st_model(content, style, return_encode=True)
            output_map = st_model.encode_img(output)
            content_identity, style_identity = st_model.get_identity(content, style)
            content_identity_map = st_model.encode_img(content_identity)
            style_identity_map = st_model.encode_img(style_identity)
            t_loss = loss_func(output_map, content_map, style_map, content_identity_map, style_identity_map,
                                content, style, content_identity, style_identity)
            
            t_loss.backward()
            optimizer.step()

            batch_loss += t_loss.item()
            if i%500==0:
                print(f"Batch {i+1} ({(time.time()-start_time):0.1f}s) completed...")

        train_loss[epoch] = batch_loss / num_batch        
        print(f"Epoch {epoch+1}/{end_epoch} ({(time.time()-start_time):0.1f}s): loss - {train_loss[epoch]:0.2f}")

    torch.save({
        "epoch": end_epoch,
        "model_state_dict": st_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_loss,
    }, save_path)

    