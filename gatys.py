import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import PIL
import numpy as np
from tqdm import tqdm
import argparse
import os

def preprocess(img, image_shape):
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


def save_image(tensor, path):
    image = postprocess(tensor)
    image.save(path)
    return image

def extract_features(X, net, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(image, image_shape, net, device):
    content_X = preprocess(image, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, net, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image, image_shape, net, device):
    style_X = preprocess(image, image_shape).to(device)
    _, styles_Y = extract_features(style_X, net, content_layers, style_layers)
    return style_X, styles_Y

def get_laplacian(image, poolsize, device):
    avg_pool_layer = nn.AvgPool2d(kernel_size=poolsize, stride=poolsize)
    lap_X = avg_pool_layer(image)
    lap_filter = torch.tensor([
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        ]).float().to(device)
    lap_X = F.conv2d(lap_X, lap_filter.unsqueeze(0))
    #print("lap_X:", lap_X.shape)
    return lap_X


# 14.12.5. Defining the Loss Function

def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 14.12.5.3. Total Variation Loss
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
    
# 14.12.5.4. Loss Function¶
content_weight, style_weight, tv_weight = 1, 1e4, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(styles_l + contents_l + [tv_l])
    #l = sum(styles_l  + [tv_l])
    return contents_l, styles_l, tv_l, l

# 14.12.6. Initializing the Synthesized Image
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    #gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

# 14.12.7. Training

def train(X, contents_Y, styles_Y, content_lap, device, lr, num_epochs, lr_decay_epoch, method, poolsize, run_name, save_every=100, save_gif=False):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    images = []
    pil_image = save_image(X, "outputs/" + run_name + "/epoch_0000.jpg")
    images.append(pil_image)
    for epoch in tqdm(range(num_epochs)):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, net, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        if method == "lapstyle":
            lap_X = get_laplacian(X, poolsize, device)
            lap_loss = content_loss(lap_X, content_lap)
            l += lap_loss * 1
            
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % save_every == 0:
            pil_image = save_image(X, "outputs/" + run_name + "/epoch_%.4d.jpg" % (epoch+1))
            images.append(pil_image)
    if save_gif:
        images[0].save("outputs/" + run_name + "/animation.gif", save_all=True, append_images=images[1:], duration=200, loop=0)
    return X


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a stylized image based on a content image and a style image")
    parser.add_argument("--style", type=str, default="input/style/monet2.jpg")
    parser.add_argument("--content", type=str, default="input/content/content3.jpg")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("-H", type=int, default=384)
    parser.add_argument("-W", type=int, default=512)

    parser.add_argument("--method", type=str, default="gatys")
    parser.add_argument("--poolsize", type=int, default=4)

    args = parser.parse_args()

    style_name = args.style.split('/')[-1][:-4]
    content_name = args.content.split('/')[-1][:-4]
    run_name = args.method + "_" + style_name + "_" + content_name + "_" + str(args.H) + "x" + str(args.W)

    style_img = PIL.Image.open(args.style)
    content_img = PIL.Image.open(args.content)


    pretrained_net = torchvision.models.vgg19(pretrained=True)


    style_layers, content_layers = [0, 5, 10, 19, 28], [25]

    net = nn.Sequential(*[pretrained_net.features[i] for i in
                        range(max(content_layers + style_layers) + 1)])

    device, image_shape = "cuda" if torch.cuda.is_available() else "cpu", (args.H, args.W)  # PIL Image (h, w)
    net = net.to(device)
    content_X, contents_Y = get_contents(content_img, image_shape, net, device)
    _, styles_Y = get_styles(style_img, image_shape, net, device)
    content_lap = get_laplacian(content_X, args.poolsize, device)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("outputs/" + run_name):
        os.makedirs("outputs/" + run_name)
    save_image(content_lap, "outputs/" + run_name + "/content_lap.jpg")

    output = train(content_X, contents_Y, styles_Y, content_lap, device, args.lr, args.epochs, args.epochs//10, args.method, args.poolsize, run_name)
































