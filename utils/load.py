from PIL import Image
from .process import preprocess_image


def load_image(img_full_dir, size=224, center_crop=True, gpu=True):
    img = Image.open(img_full_dir).convert('RGB')
    x, x0 = preprocess_image(img, size, center_crop)
    x, x0 = x[None], x0[None]

    if gpu:
        x, x0 = x.to(0), x0.to(0)
    return x, x0
