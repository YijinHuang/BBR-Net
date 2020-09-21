import torch
import numpy as np


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = to_corner(box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3])
    b2_x1, b2_y1, b2_x2, b2_y2 = to_corner(box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3])

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = torch.clamp(b1_x2 - b1_x1, 0) * torch.clamp(b1_y2 - b1_y1, 0)
    b2_area = torch.clamp(b2_x2 - b2_x1, 0) * torch.clamp(b2_y2 - b2_y1, 0)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_giou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = to_corner(box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3])
    b2_x1, b2_y1, b2_x2, b2_y2 = to_corner(box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3])

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    min_x1 = torch.min(b1_x1, b2_x1)
    min_y1 = torch.min(b1_y1, b2_y1)
    max_x2 = torch.max(b1_x2, b2_x2)
    max_y2 = torch.max(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    closure_area = torch.clamp(max_x2 - min_x1, 0) * torch.clamp(max_y2 - min_y1, 0)

    b1_area = torch.clamp(b1_x2 - b1_x1, 0) * torch.clamp(b1_y2 - b1_y1, 0)
    b2_area = torch.clamp(b2_x2 - b2_x1, 0) * torch.clamp(b2_y2 - b2_y1, 0)

    union = b1_area + b2_area - inter_area
    iou = inter_area / (union + 1e-16)
    giou = iou - ((closure_area - union) / closure_area)

    return giou


def to_corner(cx, cy, w, h):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return x1, y1, x2, y2
