import torch
import os
import numpy as np
from datasets.crowd_count import Crowd
import models.vgg
import argparse
import matplotlib.pyplot as plt
import time

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--model', type=str, default='vgg19',
                        help='the model use to test')
    parser.add_argument('--need-map', action='store_true',
                        help='whether draw density map')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    st = time.time()
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    if args.model == 'vgg19':
        model = models.vgg.vgg19()
    elif args.model == 'vgg16':
        model = models.vgg.vgg16()
    elif args.model == 'resnet18':
        model = models.vgg.resnet18()
    else:
        print("Invalid Model Type!")
        exit(0)

    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))

    i = 0
    print("loading time: ", time.time() - st)
    for inputs, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            print(name, format(torch.sum(outputs).item(), ".1f"))

            if args.need_map:
                dm = outputs.squeeze().detach().cpu().numpy()
                dm_normalized = dm / np.max(dm)
                plt.imshow(dm_normalized, cmap=plt.cm.jet, vmin=0, vmax=1)
                i += 1
                plt.savefig("./image/{}.png".format(i))
