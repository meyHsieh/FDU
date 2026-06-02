import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
from multiprocessing import freeze_support  # 导入多进程支持

# 支持双重 OpenMP 库，防止 Windows 环境下 matplotlib 与 pytorch 冲突崩溃
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

if __name__ == '__main__':
    freeze_support()

    # ## Constants (parameters) initialization
    device_id = [0,1,2,3]
    num_workers = 4
    batch_size = 128

    # add our package dir to path 
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path, 'reports', 'figures')
    models_path = os.path.join(home_path, 'reports', 'models')

    # 自动选择可用显卡 (首选 cuda:0，无显卡则回退到 cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")


    # Initialize your data loader and
    # make sure that dataloader works
    # as expected by observing one
    # sample from it.
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    for X, y in train_loader:
        print(f"Image batch shape: {X.shape}, Label batch shape: {y.shape}")
        
        # 将通道重排为 [H, W, C] 并反归一化还原为彩色图像
        img = np.transpose(X[0].numpy(), (1, 2, 0)) * 0.5 + 0.5
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"CIFAR-10 Sample (Label: {y[0].item()})")
        plt.savefig("./codes/VGG_BatchNorm/figs/cifar10_sample.png")
        break


    # This function is used to calculate the accuracy of model classification
    def get_accuracy(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                prediction = model(x)
                _, predicted = prediction.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        model.train()
        return correct / total


    # Set a random seed to ensure reproducible results
    def set_random_seeds(seed_value=0, device='cpu'):
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        if device != 'cpu': 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    # We use this function to complete the entire
    # training process. In order to plot the loss landscape,
    # you need to record the loss value of each step.
    def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None,
            epochs_n=100, best_model_path=None,save_name='./codes/VGG_BatchNorm/figs/training_curves.png'):
        model.to(device)
        
        train_losses = []
        train_accs = []
        val_accs = []
        
        batches_n = len(train_loader)
        losses_list = []
        grads = []
        
        for epoch in tqdm(range(epochs_n), unit='epoch'):
            if scheduler is not None:
                scheduler.step()
            model.train()

            loss_list = []  # 记录该 epoch 的每一步 loss
            grad = []       # 记录该 epoch 的每一步梯度范数
            epoch_loss = 0.0

            for data in train_loader:
                x, y = data
                x = x.to(device)
                y = y.to(device)
                
                optimizer.zero_grad()
                prediction = model(x)
                loss = criterion(prediction, y)

                loss_list.append(loss.item())
                epoch_loss += loss.item()

                loss.backward()
                
                # 记录梯度 L2 范数
                grad_val = model.classifier[4].weight.grad.norm().item()
                grad.append(grad_val)

                optimizer.step()

            train_losses.append(epoch_loss / batches_n)
            train_accs.append(get_accuracy(model, train_loader))
            val_accs.append(get_accuracy(model, val_loader))
            
            losses_list.append(loss_list)
            grads.append(grad)

        epochs_range = range(1, epochs_n + 1) 
        f, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左子图: 训练 Loss 曲线
        axes[0].plot(epochs_range, train_losses, '-', color='#4c72b0', label='Train Loss')
        axes[0].set_title("Training Loss Curve", fontsize=12)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()
        
        # 右子图: 准确率曲线
        axes[1].plot(epochs_range, train_accs, '-', color='#55a868', label='Train Acc')
        axes[1].plot(epochs_range, val_accs, '-', color='#c44e52', label='Val Acc')
        axes[1].set_title("Accuracy Curve", fontsize=12)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=800)

        return losses_list, grads



    epo = 20
    lrs = [1e-3, 2e-3, 1e-4, 5e-4]
    all_losses_std = []
    all_losses_bn = []


    print("--- Training Standard VGG_A ---")
    for lr_val in lrs:
        print(f"Training Standard VGG_A with lr={lr_val}")
        set_random_seeds(seed_value=2020, device=device)
        model_std = VGG_A()
        opt_std = torch.optim.Adam(model_std.parameters(), lr=lr_val)
        criterion = nn.CrossEntropyLoss()
        
        losses_std, grads_std = train(model_std, opt_std, criterion, train_loader, 
            val_loader, epochs_n=epo,save_name=f'./codes/VGG_BatchNorm/figs/VGG_A_curves_lr_{lr_val}.png')
        
        flat_losses_std = [l for epoch_loss in losses_std for l in epoch_loss]
        flat_grads_std = [g for epoch_grad in grads_std for g in epoch_grad]
        
        all_losses_std.append(flat_losses_std)
        
        np.savetxt(f'./codes/VGG_BatchNorm/outputs/loss_std_lr_{lr_val}.txt', flat_losses_std, fmt='%.6f')
        np.savetxt(f'./codes/VGG_BatchNorm/outputs/grads_std_lr_{lr_val}.txt', flat_grads_std, fmt='%.6f')
        torch.save(model_std.state_dict(), f'./codes/VGG_BatchNorm/weights/model_std_lr_{lr_val}.pth')


    print("--- Training VGG_A_BatchNorm ---")
    for lr_val in lrs:
        print(f"Training VGG_A_BatchNorm with lr={lr_val}")
        set_random_seeds(seed_value=2020, device=device)
        model_bn = VGG_A_BatchNorm()
        opt_bn = torch.optim.Adam(model_bn.parameters(), lr=lr_val)
        criterion = nn.CrossEntropyLoss()
        
        losses_bn, grads_bn = train(model_bn, opt_bn, criterion, train_loader, 
            val_loader, epochs_n=epo,save_name=f'./codes/VGG_BatchNorm/figs/VGG_A_BN_curves_lr_{lr_val}.png')
        
        flat_losses_bn = [l for epoch_loss in losses_bn for l in epoch_loss]
        flat_grads_bn = [g for epoch_grad in grads_bn for g in epoch_grad]
        
        all_losses_bn.append(flat_losses_bn)
        
        np.savetxt(f'./codes/VGG_BatchNorm/outputs/loss_bn_lr_{lr_val}.txt', flat_losses_bn, fmt='%.6f')
        np.savetxt(f'./codes/VGG_BatchNorm/outputs/grads_bn_lr_{lr_val}.txt', flat_grads_bn, fmt='%.6f')
        torch.save(model_bn.state_dict(), f'./codes/VGG_BatchNorm/weights/model_bn_lr_{lr_val}.pth')

    #Maintain two lists: max_curve and min_curve, 
    #select the maximum value of loss in all models on the same step, 
    # add it to max_curve, and the minimum value to min_curve;
    std_losses_array = np.array(all_losses_std)
    min_curve = np.min(std_losses_array, axis=0).tolist()
    max_curve = np.max(std_losses_array, axis=0).tolist()

    bn_losses_array = np.array(all_losses_bn)
    min_curve_bn = np.min(bn_losses_array, axis=0).tolist()
    max_curve_bn = np.max(bn_losses_array, axis=0).tolist()


    # Use this function to plot the final loss landscape,
    # fill the area between the two curves can use plt.fill_between()
    def plot_loss_landscape():
        plt.style.use('seaborn-v0_8-darkgrid')
 
        steps = np.arange(len(min_curve))
        plt.figure(figsize=(10, 6.5))
        
        # 绘制 Standard VGG 区域
        plt.plot(steps, min_curve, color='#55a868', alpha=0.8, linewidth=1)
        plt.plot(steps, max_curve, color='#55a868', alpha=0.8, linewidth=1)
        plt.fill_between(steps, min_curve, max_curve, color='#8ebb9b', alpha=0.55, label='Standard VGG')

        # Standard VGG + BatchNorm 区域
        plt.plot(steps, min_curve_bn, color='#c44e52', alpha=0.8, linewidth=1)
        plt.plot(steps, max_curve_bn, color='#c44e52', alpha=0.8, linewidth=1)
        plt.fill_between(steps, min_curve_bn, max_curve_bn, color='#d59496', alpha=0.55, label='Standard VGG + BatchNorm')
        
        plt.title('Loss Landscape', fontsize=14)
        plt.xlabel('Steps', fontsize=11)
        plt.ylabel('Loss Landscape', fontsize=11)
        plt.legend(fontsize=12, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('./codes/VGG_BatchNorm/figs/loss_landscape.png', dpi=800)

    plot_loss_landscape()