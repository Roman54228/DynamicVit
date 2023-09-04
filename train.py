import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
#import tome
import timm
import torch.optim as optim
import argparse
import os
from models.dyswin import AdaSwinTransformer, SwinTransformer_Teacher
from losses import ConvNextDistillDiffPruningLoss, DistillDiffPruningLoss_dynamic
from models.dylvvit import LVViTDiffPruning, LVViT_Teacher
from dynvit import get_models

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=True)
parser.add_argument("--base_rate", type=float, default='0.9')
parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

args = parser.parse_args()



transform = transforms.Compose(
    [transforms.Resize((224,224)),
      transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# timm_net_name = "vit_base_patch16_384"
# net = timm.create_model(timm_net_name, pretrained=False)
# net.head = nn.Linear(768, 10)

# if args.tome:
#     print(args.tome)
#     tome.patch.timm(net)
#     net.r = 45

# SPARSE_RATIO = [args.base_rate, args.base_rate - 0.2, args.base_rate - 0.4]

# net = AdaSwinTransformer(
#             embed_dim=96,
#             depths=[2, 2, 6, 2],
#             num_heads=[3, 6, 12, 24],
#             window_size=7,
#             drop_rate=0.0,
#             drop_path_rate=args.drop_path,
#             num_classes=10,
#             pruning_loc=[1,2,3], sparse_ratio=SPARSE_RATIO
#         )
# # pretrained = torch.load('./pretrained/swin_tiny_patch4_window7_224.pth', map_location='cpu')
# teacher_model = SwinTransformer_Teacher(
#             embed_dim=96,
#             depths=[2, 2, 6, 2],
#             num_heads=[3, 6, 12, 24],
#             num_classes=10,
#             window_size=7).to(device)

# PRUNING_LOC = [4,8,12] 
# KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
# print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
# model = LVViTDiffPruning(
#                 patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
#             p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, num_classes=10,
#             pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True
#         )
# teacher_model = LVViT_Teacher(
#                 patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
#             p_emb='4_2',skip_lam=2., num_classes=10, return_dense=True, mix_token=True
#         )
model, teacher_model, criterion = get_models(args.arch, 10)

net = model
# criterion = torch.nn.CrossEntropyLoss()
# # criterion = ConvNextDistillDiffPruningLoss(
# #                 teacher_model, criterion, clf_weight=1.0, keep_ratio=SPARSE_RATIO, mse_token=True, ratio_weight=10.0, swin_token=True)
# criterion = DistillDiffPruningLoss_dynamic(
#             teacher_model, criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=False, ratio_weight=2.0, distill_weight=0.5
#         )
print(net)
net.to(device)
teacher_model.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # outputs = net(inputs)
        output = net(inputs)
        #pdb.set_trace()
        loss, loss_part = criterion(inputs, output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # outputs = net(images)
        pred, *_ = net(images)
        #pdb.set_trace()
        _, pred = torch.max(pred.data, 1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
suffix_tome = 'tome' if args.tome else 'no_merge'

model_save_path = f'{args.arch}@{suffix_tome}.pth'
torch.save(net.state_dict(), 'dir_ckpt/'+model_save_path)
net.eval().cpu()
x = torch.randn(1, 3, 224, 224, requires_grad=True)

torch.onnx.export(net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"onnx_weights/{os.path.basename(model_save_path).split('.')[0]}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})