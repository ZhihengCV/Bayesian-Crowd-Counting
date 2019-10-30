import torch
from torchvision import transforms
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19



patch = 1
# test_dir = '/home/teddy/crowd_data/Sh_A_Train_Val_NP/val'
test_dir = '/home/teddy/UCF-Train-Val-Test/test'
model_path = '/home/teddy/iccv-reproduce-new/1029-225909/best_model_3.pth'
root_dir = '/home/teddy/iccv-reproduce-new/1029-230053'
vis_dir = os.path.join(root_dir, 'vis_test')
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

datasets = Crowd(test_dir, 512, 8, is_gray=False, method='val')

dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                         num_workers=8, pin_memory=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set vis gpu
device = torch.device('cuda')

model = vgg19()
model.to(device)
# model.load_state_dict(torch.load(model_path, device)['model_state_dict'])
model.load_state_dict(torch.load(model_path, device))
epoch_minus = []
for inputs, count, name in dataloader:
    inputs = inputs.to(device)
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        temp_minu = count[0].item() - torch.sum(outputs).item()
        print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
        epoch_minus.append(temp_minu)

epoch_minus = np.array(epoch_minus)
epoch_minus = epoch_minus.reshape([-1, patch])
epoch_minus = np.sum(epoch_minus, axis=1)
mse = np.sqrt(np.mean(np.square(epoch_minus)))
mae = np.mean(np.abs(epoch_minus))
log_str = 'mae {}, mse {}'.format(mae, mse)
print(log_str)
with open(os.path.join(root_dir, 'test_results.txt'), 'w') as f:
    f.write(log_str+'\n')
