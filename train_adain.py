import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import net
from dataset import ImageDataset

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset and VGG
    parser.add_argument('--content_dir', type=str, required=True) # Content dataset
    parser.add_argument('--style_dir', type=str, required=True) # Style dataset
    parser.add_argument('--vgg', type=str, default='models/vgg_normalized.pth') #Path to VGG

    # Logging and checkpoints
    parser.add_argument('--ckpt_dir', default='./checkpoints')
    parser.add_argument('--log_dir', default='./logs_train')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--lap_weight', type=float, default=5.0)
    parser.add_argument('--lap_poolsize', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10000)
    args = parser.parse_args()

    exp_name = "adain"
    
    device = torch.device('cuda')

    ckpt_dir = Path(args.ckpt_dir)
    log_dir = Path(args.log_dir)
    
    run_name = exp_name + "_" + str(args.style_weight) + "_" + str(args.content_weight)
    run_name += "_" + str(args.lap_weight) + "_" + str(args.lap_poolsize)
    import datetime
    run_date = datetime.datetime.now().strftime('%H%M%S')
    run_name = run_name + '_' + run_date
    ckpt_dir = ckpt_dir / run_name
    log_dir = log_dir / run_name
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    
    decoder = net.decoder
    vgg = net.vgg

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = net.Net(vgg, decoder, poolsize=args.lap_poolsize, device=device)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = ImageDataset(args.content_dir, content_tf)
    style_dataset = ImageDataset(args.style_dir, style_tf)

    content_dataloader = data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        shuffle=True,
        num_workers=6)
    style_dataloader = data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        shuffle=True,
        num_workers=6)
    

    content_iterloader = iter(content_dataloader)
    style_iterloader = iter(style_dataloader)

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i)
        try:
            style_images = next(style_iterloader)
        except StopIteration:
            style_iterloader = iter(style_dataloader)
            style_images = next(style_iterloader)
        try:
            content_images = next(content_iterloader)
        except StopIteration:
            content_iterloader = iter(content_dataloader)
            content_images = next(content_iterloader)
        content_images = content_images.to(device)
        style_images = style_images.to(device)
        if content_images.shape[0] != style_images.shape[0] or content_images.shape[1] != style_images.shape[1]:
            continue

        loss_c, loss_s, loss_lap = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss_lap = args.lap_weight * loss_lap
        loss = loss_c + loss_s + loss_lap

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss/content_loss', loss_c.item(), i + 1)
        writer.add_scalar('loss/style_loss', loss_s.item(), i + 1)
        writer.add_scalar('loss/lap_loss', loss_lap.item(), i + 1)

        if (i + 1) % args.save_freq == 0 or (i + 1) == args.max_iter:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, ckpt_dir /
                    'decoder_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()
