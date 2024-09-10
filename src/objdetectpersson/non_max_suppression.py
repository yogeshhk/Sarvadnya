# Clean up bounding boxes
# https://www.youtube.com/watch?v=YDkjWEN8jNA&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=3
# Test pairwise all bounding boxes,
# starting with the largest probability
# use IOU, and discard the wrong ones ie less than threshold

import torch
from boundingbox_eval import intersection_over_union


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


if __name__ == "__main__":
    # Set up some sample data for testing
    bboxes = [
        [0, 0.9, 0.1, 0.1, 0.3, 0.3],
        [0, 0.8, 0.2, 0.2, 0.4, 0.4],
        [0, 0.7, 0.3, 0.3, 0.5, 0.5],
        [1, 0.95, 0.1, 0.1, 0.3, 0.3],  # Different class
    ]
    iou_threshold = 0.3
    threshold = 0.6
    box_format = "corners"

    result = nms(bboxes, iou_threshold, threshold, box_format)
    print(result)
    assert len(result), 2
    assert result[0], [0, 0.9, 0.1, 0.1, 0.3, 0.3]
    assert result[1], [1, 0.95, 0.1, 0.1, 0.3, 0.3]
