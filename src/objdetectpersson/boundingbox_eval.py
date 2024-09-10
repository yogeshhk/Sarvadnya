# https://www.youtube.com/watch?v=XXYG5ZWtjj0&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq
# Find Intersection and Union of two bounding boxes, one actual and another predicted, to quantify correctness
# Intersection Over Union, IOU = IntersectionArea/ UnionArea; 1 is perfect, 0 is worst.
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/iou.py
# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

import torch


def intersection_over_union(boxes_prediction, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_prediction (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_prediction[..., 0:1] - boxes_prediction[..., 2:3] / 2
        box1_y1 = boxes_prediction[..., 1:2] - boxes_prediction[..., 3:4] / 2
        box1_x2 = boxes_prediction[..., 0:1] + boxes_prediction[..., 2:3] / 2
        box1_y2 = boxes_prediction[..., 1:2] + boxes_prediction[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_prediction[..., 0:1]
        box1_y1 = boxes_prediction[..., 1:2]
        box1_x2 = boxes_prediction[..., 2:3]
        box1_y2 = boxes_prediction[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)

    intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  # clamp when they don't intersect
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    union_area = (box1_area + box2_area - intersection_area) + 1e-6  # to avoid divide by zero

    ratio = intersection_area / union_area

    return ratio


if __name__ == "__main__":
    # Set up some sample data for testing
    boxes_pred_midpoint = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    boxes_label_midpoint = torch.tensor([[0.5, 0.5, 1.0, 1.0]])

    boxes_pred_corners = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    boxes_label_corners = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    iou = intersection_over_union(boxes_pred_midpoint, boxes_label_midpoint, box_format="midpoint")
    assert iou, 1.0

    iou = intersection_over_union(boxes_pred_corners, boxes_label_corners, box_format="corners")
    assert iou, 1.0