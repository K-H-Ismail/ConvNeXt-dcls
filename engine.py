# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma, AverageMeter

from models.convnext import ConvNeXt
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import plotly.express as px

import utils


from PIL import Image




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, use_dcls=False, dcls_kernel_size=7):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if param_group["group_name"] == "no_decay_dcls":
                        param_group["lr"] = lr_schedule_values[it] * 5
                    else:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_rep = torch.zeros_like(loss)
        loss_fit = loss.item()
        if use_dcls:
            layer_count = 0
            for name, param in model.named_parameters():
                if name.endswith(".P"):
                    layer_count += 1
                    chout, chin, k_count = param.size(1), param.size(2), param.size(3)
                    P = param.view(2, chout * chin, k_count)
                    P = P.permute(1,2,0).contiguous()
                    distances = torch.cdist(P,P,p=2)
                    distances_triu = (1-distances).triu(diagonal=1)
                    loss_rep += 2*torch.sum(torch.clamp_min(distances_triu , min=0)) / (k_count*(k_count-1)*chout*chin)
            loss_rep /= layer_count

            loss = loss + loss_rep ** 2 if epoch > 20 else loss

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        if use_dcls:
            with torch.no_grad():
                lim = dcls_kernel_size // 2
                for i in range(4):
                    if hasattr(model, 'module'):
                        for j in range(len(model.module.stages[i])):
                            model.module.stages[i][j].dwconv.P.clamp_(-lim, lim)
                    else:
                        for j in range(len(model.stages[i])):
                            model.stages[i][j].dwconv.P.clamp_(-lim, lim)
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_fit=loss_fit)
        metric_logger.update(loss_rep=loss_rep)
        metric_logger.update(lr_pos=lr_schedule_values[it] * 5)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class convnextForERF(ConvNeXt):

    def __init__(self, args, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__(in_chans=3, num_classes=args.nb_classes, depths=depths, dims=dims,
                                     drop_path_rate=args.drop_path, layer_scale_init_value=args.layer_scale_init_value,
                                     head_init_scale=args.head_init_scale, use_dcls=args.use_dcls,
                                     dcls_kernel_size=args.dcls_kernel_size, dcls_kernel_count=args.dcls_kernel_count,
                                     dcls_sync=args.dcls_sync)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
        # return self.norm(x) # Using the feature maps after the final norm also makes sense. Observed very little difference.


def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu()
    return grad_map

def visualize_erf(model, args):

    #   ================================= transform: resize to 1024x1024
    t = [
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    print("reading from datapath", args.data_path)
    root = os.path.join(args.data_path, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
        batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    if args.model == 'convnext_tiny':
        depths, dims = [3, 3, 9, 3], [96, 192, 384, 768]
    elif args.model == 'convnext_small':
        depths, dims = [3, 3, 27, 3], [96, 192, 384, 768]
    elif args.model == 'convnext_base':
        depths, dims = [3, 3, 27, 3], [128, 256, 512, 1024]
    elif args.model == 'convnext_large':
        depths, dims = [3, 3, 27, 3], [192, 384, 768, 1536]
    elif args.model == 'convnext_xlarge':
        depths, dims = [3, 3, 27, 3], [256, 512, 1024, 2048]
    else:
        raise ValueError('Unsupported model. Please add it here.')

    model_for_erf = convnextForERF(args, depths=depths, dims=dims)

    device = torch.device(args.device)
    model_for_erf.to(device)
    model_for_erf.load_state_dict(model.state_dict())
    model_for_erf.eval()    #   fix BN and droppath

    optimizer = torch.optim.SGD(model_for_erf.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for i, (samples, _) in enumerate(data_loader_val):
        print("ERF construction: ", i, "/200")
        if meter.count == 200:
            break

        samples = samples.to(device, non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model_for_erf, samples)

        if contribution_scores.sum().isnan():
            print('got NAN, next image')
            continue
        else:
            meter.update(contribution_scores)

    data = meter.avg
    print(data.max())
    print(data.min())
    data = (data + 1).log10()       #   the scores differ in magnitude. take the logarithm for better readability
    data = data / data.max()      #   rescale to [0,1] for the comparability among models
    print('======================= the high-contribution area ratio =====================')

    fig = px.imshow(data)
    #ticks = dict(tickmode = 'linear', tick0 = 0, dtick = 50)
    #fig.update_layout(
    #    xaxis = ticks,
    #    yaxis = ticks
    #)
    image_name = "erf_{model}{dcls}.png".format(model=args.model, dcls="_dcls" if args.use_dcls else "")
    fig.write_image(image_name)
    print('heatmap saved at ', image_name)

