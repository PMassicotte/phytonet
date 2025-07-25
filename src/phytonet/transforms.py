"""Custom transforms for phytoplankton image preprocessing."""

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class ResizeKeepAspectPad:
    """Resize while keeping aspect ratio and pad to square."""

    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        # Keep aspect ratio and resize to fit within target size
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = F.resize(img, [new_h, new_w], antialias=True)

        # Pad to square
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)
        return img


def get_train_transform(image_size=224):
    """Get training data transforms."""
    return transforms.Compose(
        [
            ResizeKeepAspectPad(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(image_size=224):
    """Get validation data transforms."""
    return transforms.Compose(
        [
            ResizeKeepAspectPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
