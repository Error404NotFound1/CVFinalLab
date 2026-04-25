import torch
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_tta=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    autocast_context = torch.cuda.amp.autocast() if amp and device.type == 'cuda' else nullcontext()

    with autocast_context:
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            if use_tta:
                mask_pred = tta_predict(net, image)
            else:
                mask_pred = net(image)

            if hasattr(net, 'n_classes'):
                n_classes = net.n_classes
            else:
                n_classes = mask_pred.shape[1]

            if n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes['
                mask_true_oh = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
                mask_pred_oh = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred_oh[:, 1:], mask_true_oh[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


def tta_predict(net, image):
    """Test-Time Augmentation: horizontal flip + average."""
    pred_orig = net(image)
    pred_flip = net(torch.flip(image, dims=[3]))
    pred_flip = torch.flip(pred_flip, dims=[3])
    return (pred_orig + pred_flip) / 2


@torch.inference_mode()
def evaluate_full(net, dataloader, device, amp, num_classes=3, use_postprocess=False):
    """Full evaluation with per-class IoU, mIoU, Dice, and pixel accuracy."""
    net.eval()
    class_names = ['foreground', 'background', 'boundary'][:num_classes]

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    correct = 0
    total = 0
    autocast_context = torch.cuda.amp.autocast() if amp and device.type == 'cuda' else nullcontext()

    if use_postprocess:
        from utils.postprocess import postprocess_mask

    with autocast_context:
        for batch in tqdm(dataloader, desc='Evaluating', unit='batch'):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image).argmax(dim=1)

            if use_postprocess:
                mask_pred_np = mask_pred.cpu().numpy()
                for i in range(mask_pred_np.shape[0]):
                    mask_pred_np[i] = postprocess_mask(mask_pred_np[i])
                mask_pred = torch.from_numpy(mask_pred_np).to(device)

            correct += (mask_pred == mask_true).sum().item()
            total += mask_true.numel()

            for c in range(num_classes):
                pred_c = (mask_pred == c)
                true_c = (mask_true == c)
                intersection[c] += (pred_c & true_c).sum()
                union[c] += (pred_c | true_c).sum()

    iou_per_class = (intersection + 1e-6) / (union + 1e-6)
    miou = iou_per_class.mean().item()
    pixel_acc = correct / max(total, 1)

    net.train()
    return {
        'miou': miou,
        'pixel_accuracy': pixel_acc,
        'per_class_iou': {name: iou_per_class[i].item() for i, name in enumerate(class_names)},
    }
