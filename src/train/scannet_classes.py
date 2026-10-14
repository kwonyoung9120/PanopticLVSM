"""ScanNet-20 panoptic class names, stuff ids and colour palette."""

PANOPTIC_SEMANTIC2NAME = {
    0: "unlabeled",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "desk",
    14: "curtain",
    15: "refrigerator",
    16: "shower curtain",
    17: "toilet",
    18: "sink",
    19: "bathtub",
    20: "otherfurniture",
}
STUFF_CLASSES = [
    0,
    1,
]  # Zero-based wall and floor labels.

PANOPTIC_COLOR_PALLETE = {
    0: [0, 0, 0],  # unlabeled
    1: [174, 199, 232],  # wall
    2: [152, 223, 138],  # floor
    3: [31, 119, 180],  # cabinet
    4: [255, 187, 120],  # bed
    5: [188, 189, 34],  # chair
    6: [140, 86, 75],  # sofa
    7: [255, 152, 150],  # table
    8: [214, 39, 40],  # door
    9: [197, 176, 213],  # window
    10: [148, 103, 189],  # bookshelf
    11: [196, 156, 148],  # picture
    12: [23, 190, 207],  # counter
    13: [247, 182, 210],  # desk
    14: [219, 219, 141],  # curtain
    15: [255, 127, 14],  # refrigerator
    16: [158, 218, 229],  # shower curtain
    17: [44, 160, 44],  # toilet
    18: [112, 128, 144],  # sink
    19: [227, 119, 194],  # bathtub
    20: [82, 84, 163],  # otherfurniture
}
