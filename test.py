import torch
import os
import numpy as np
# from datasets.crowd import Crowd
from datasets.crowd_sh import Crowd
import models.vgg
import argparse
import matplotlib.pyplot as plt

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
    epoch_minus = []

    i = 0
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)

            if args.need_map:
                dm = outputs.squeeze().detach().cpu().numpy()
                dm_normalized = dm / np.max(dm)
                plt.imshow(dm_normalized, cmap=plt.cm.jet, vmin=0, vmax=1)
                i += 1
                plt.savefig("./image/{}.png".format(i))

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
