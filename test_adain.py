import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from PIL import Image
import net
from function import adaptive_instance_normalization, coral
import saverloader
import os


def test_transform(size=512):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor()
    ])
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--content', type=str)
    parser.add_argument('--content_dir', type=str)
    parser.add_argument('--style', type=str)
    parser.add_argument('--style_dir', type=str)
    parser.add_argument('--vgg', type=str, default='models/vgg_normalized.pth')
    parser.add_argument('--ckpt_dir', type=str, default='../../adain_results/checkpoints')
    parser.add_argument('--init_dir', type=str, default='adain_10.0_1.0_5.0_4_084825')

    parser.add_argument('--content_size', type=int, default=512)
    parser.add_argument('--style_size', type=int, default=512)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--alpha', type=float, default=1.0) # style-content ratio

    args = parser.parse_args()

    assert (args.init_dir)

    output_dir = Path(args.output) / args.init_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)
    if args.style:
        style_paths = args.style.split(',')
        if len(style_paths) > 1:
            assert (args.style)
        else:
            style_paths = [Path(args.style)]
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    #decoder.load_state_dict(torch.load(args.decoder))
    saverloader.load(os.path.join(args.ckpt_dir, args.init_dir), decoder)
    vgg.load_state_dict(torch.load(args.vgg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)
    decoder.to(device)

    vgg = nn.Sequential(*list(vgg.children())[:31]) # until ReLU4_1

    content_tf = test_transform(args.content_size)
    style_tf = test_transform(args.style_size)

    for content_path in content_paths:
        for style_path in style_paths:
            # process one content and one style
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            # if args.preserve_color:
            #     style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, args.alpha)
            output = output.cpu()

            output_name = output_dir / '{:s}_{:s}{:s}'.format(
                content_path.stem, style_path.stem, ".jpg")
            save_image(output, str(output_name))

