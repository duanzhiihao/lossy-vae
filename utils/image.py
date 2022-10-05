import math
import torchvision.transforms.functional as tvf


def pad_divisible_by(img, div=64):
    """ Pad a PIL.Image such that both its sides are divisible by `div`

    Args:
        img (PIL.Image): input image
        div (int, optional): denominator. Defaults to 64.

    Returns:
        PIL.Image: padded image
    """
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_tgt = round(div * math.ceil(h_old / div))
    w_tgt = round(div * math.ceil(w_old / div))
    # left, top, right, bottom
    padding = (0, 0, (w_tgt - w_old), (h_tgt - h_old))
    padded = tvf.pad(img, padding=padding, padding_mode='edge')
    return padded


def crop_divisible_by(img, div=64):
    ''' Center crop a PIL.Image such that both its sides are divisible by `div`

    Args:
        img (PIL.Image): input image
        div (int, optional): denominator. Defaults to 64.

    Returns:
        PIL.Image: cropped image
    '''
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_new = div * (h_old // div)
    w_new = div * (w_old // div)
    cropped = tvf.center_crop(img, output_size=(h_new, w_new))
    return cropped
