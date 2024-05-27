import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.transforms as transforms

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from matplotlib import animation, rc

from models.vae_model import VAE
from models.loss import criterion

load_dotenv()
WANDB_API_KEY = os.environ["WANDB_API_KEY"]


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)

    BATCH_SIZE = 100
    z_dim = 2
    num_epochs = 20

    config = dict(
        batch_size=BATCH_SIZE,
        z_dim=z_dim,
        num_epochs=num_epochs
    )

    wandb.init(
        project="vae_mnist",
        entity="hiroto-weblab",
        config=config
    )

    trainval_data = MNIST("./data", 
                    train=True, 
                    download=True, 
                    transform=transforms.ToTensor())

    train_size = int(len(trainval_data) * 0.8)
    val_size = int(len(trainval_data) * 0.2)
    train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

    train_loader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)

    print("train data size: ",len(train_data))
    print("train iteration number: ",len(train_data)//BATCH_SIZE)
    print("val data size: ",len(val_data))
    print("val iteration number: ",len(val_data)//BATCH_SIZE)

    images, labels = next(iter(train_loader))
    print("images_size:",images.size())
    print("label:",labels[:10])

    image_numpy = images.detach().numpy().copy()
    plt.imshow(image_numpy[0,0,:,:], cmap='gray')
    plt.show()

    plt.imsave("figure/sample.png", image_numpy[0,0,:,:], cmap='gray')


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = VAE(z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels":[]}

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
    for epoch in range(num_epochs):
        model.train()
        for i, (x, labels) in enumerate(train_loader):
            input = x.to(device).view(-1, 28*28).to(torch.float32)
            output, z, ave, log_dev = model(input)

            history["ave"].append(ave)
            history["log_dev"].append(log_dev)
            history["z"].append(z)
            history["labels"].append(labels)
            loss = criterion(output, input, ave, log_dev)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
            history["train_loss"].append(loss.item())

            # wandbにログを追加
            wandb.log({
                "train_loss": loss.item(),
                "ave": ave.mean().item(),  
                "log_dev": log_dev.mean().item(),  
                "z": z.mean().item(),  
            })

        model.eval()
        with torch.no_grad():
            for i, (x, labels) in enumerate(val_loader):
                input = x.to(device).view(-1, 28*28).to(torch.float32)
                output, z, ave, log_dev = model(input)

                loss = criterion(output, input, ave, log_dev)
                history["val_loss"].append(loss.item())

            print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')

            # wandbにval_lossをログ
            wandb.log({
                "val_loss": loss.item()
            })

        scheduler.step()


    ave_tensor = torch.stack(history["ave"])
    log_var_tensor = torch.stack(history["log_dev"])
    z_tensor = torch.stack(history["z"])
    labels_tensor = torch.stack(history["labels"])
    print(ave_tensor.size())
    print(log_var_tensor.size())
    print(z_tensor.size())
    print(labels_tensor.size())

    ave_np = ave_tensor.to('cpu').detach().numpy().copy()
    log_var_np = log_var_tensor.to('cpu').detach().numpy().copy()
    z_np = z_tensor.to('cpu').detach().numpy().copy()
    labels_np = labels_tensor.to('cpu').detach().numpy().copy()
    print(ave_np.shape)
    print(log_var_np.shape)
    print(z_np.shape)
    print(labels_np.shape)


    cmap_keyword = "tab10"
    cmap = plt.get_cmap(cmap_keyword)

    batch_num =10

    plt.figure(figsize=[10,10])
    for label in range(10):
        x = z_np[:batch_num,:,0][labels_np[:batch_num,:] == label]
        y = z_np[:batch_num,:,1][labels_np[:batch_num,:] == label]
        plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
        plt.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
    plt.legend(loc="upper left")
    plt.savefig("figure/vae_latent_space_start.png")

    batch_num = 9580

    plt.figure(figsize=[10,10])
    for label in range(10):
        x = z_np[batch_num:,:,0][labels_np[batch_num:,:] == label]
        y = z_np[batch_num:,:,1][labels_np[batch_num:,:] == label]
        plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
        plt.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
    plt.legend(loc="upper left")
    plt.savefig("figure/vae_latent_space_end.png")

    model.to("cpu")

    label = 0
    x_zero_mean = np.mean(ave_np[batch_num:,:,0][labels_np[batch_num:,:] == label])
    y_zero_mean = np.mean(ave_np[batch_num:,:,1][labels_np[batch_num:,:] == label])

    label = 1
    x_one_mean = np.mean(ave_np[batch_num:,:,0][labels_np[batch_num:,:] == label])
    y_one_mean = np.mean(ave_np[batch_num:,:,1][labels_np[batch_num:,:] == label])

    z_zero = torch.tensor([x_zero_mean,y_zero_mean], dtype = torch.float32)
    z_one = torch.tensor([x_one_mean,y_one_mean], dtype = torch.float32)

    plt.figure(figsize=[10,10])
    output = model.decoder(z_zero)
    np_output = output.to('cpu').detach().numpy().copy()
    np_image = np.reshape(np_output, (28, 28))
    plt.imshow(np_image, cmap='gray')
    plt.savefig("figure/vae_zero.png")

    plt.figure(figsize=[10,10])
    output = model.decoder(z_one)
    np_output = output.to('cpu').detach().numpy().copy()
    np_image = np.reshape(np_output, (28, 28))
    plt.imshow(np_image, cmap='gray')
    plt.savefig("figure/vae_one.png")


    def plot(frame):
        plt.cla()
        z_zerotoone = ((99 - frame) * z_zero +  frame * z_one) / 99
        output = model.decoder(z_zerotoone)
        np_output = output.detach().numpy().copy()
        np_image = np.reshape(np_output, (28, 28))
        plt.imshow(np_image, cmap='gray')
        plt.xticks([]);plt.yticks([])
        plt.title("frame={}".format(frame))

    fig = plt.figure(figsize=(4,4))
    ani = animation.FuncAnimation(fig, plot, frames=99, interval=100)
    rc('animation', html='jshtml')
    # save as gif
    ani.save("figure/vae_zerotone.gif", writer="imagemagick")
