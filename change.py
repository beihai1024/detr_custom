import torch

pretrained_weights = torch.load("/content/coco_pretrain/detr-r50-e632da11.pth")

num_class = 3
pretrained_weights["model"]["class_embed.weight"].resize_(num_class + 1, 256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_class + 1)

torch.save(pretrained_weights, '/content/detr_r50_%d.pth' % num_class)
