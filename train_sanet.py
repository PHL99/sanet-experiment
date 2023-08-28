import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from train_utils import ContentStyleDataset, STLoss, train_and_save
from net import SANTransfer



def visualize(content, style, output, content_identity, style_identity):
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    ax[0][0].title.set_text("Content")
    ax[0][1].title.set_text("Style")
    ax[0][2].title.set_text("Output")
    ax[0][3].title.set_text("Content Identity")
    ax[0][4].title.set_text("Style Identity")
    for i in range(5):
        ax[i][0].imshow(content[i].cpu().permute(1, 2, 0))
        ax[i][0].axis("off")
        ax[i][1].imshow(style[i].cpu().permute(1, 2, 0))
        ax[i][1].axis("off")
        ax[i][2].imshow(output[i].cpu().permute(1, 2, 0))
        ax[i][2].axis("off")
        ax[i][3].imshow(content_identity[i].detach().cpu().permute(1, 2, 0))
        ax[i][3].axis("off")
        ax[i][4].imshow(style_identity[i].detach().cpu().permute(1, 2, 0))
        ax[i][4].axis("off")
    plt.show()



if __name__ == "__main__":
    ### Data loading
    batch_size = 5
    cs_train = ContentStyleDataset("output/train", transform=transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.RandomCrop(256),
    ]))
    train_loader = DataLoader(cs_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    ### Train model and save state as tar file
    start_epoch = 0
    end_epoch = 6
    lr = 0.0005
    load_path = "save_state/sanet_state_{}_ep.tar".format(start_epoch) if start_epoch > 0 else None
    save_path = "save_state/sanet_state_{}_ep.tar".format(end_epoch)
    
    st_model = SANTransfer(train_decoder=False)
    optimizer = torch.optim.Adam(st_model.parameters(), lr=lr)
    loss_func = STLoss(lmbda_c=2, lmbda_s=6, lmbda_i1=15, lmbda_i2=2)
    train_and_save(st_model, optimizer, loss_func, end_epoch, train_loader,
                   save_path=save_path, load_path=load_path)
 
    ### Evaluate on first batch of test samples
    print("Evaluating...")
    
    cs_test = ContentStyleDataset("output/val", transform=transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ]))
    test_loader = DataLoader(cs_test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SANTransfer()
    checkpoint = torch.load(save_path)
    st_model.load_state_dict(checkpoint["model_state_dict"])
    st_model.to(device)
    st_model.eval()

    content, style = next(iter(test_loader))
    content, style = content.to(device), style.to(device)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        output = st_model(content, style)
    print("Average inference time: ", (time.time()-start_time)/batch_size)
    
    # Visualize images
    content_identity, style_identity = st_model.get_identity(content, style)
    visualize(content, style, output.detach(), content_identity.detach(), style_identity.detach())
