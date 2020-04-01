from cv2 import cv2
import torchvision.transforms as transforms


def preprocessing(images, masks):
    fin_images = []
    fin_masks = []
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    for i in range(len(images)):
        tmp_i = cv2.resize(images[i], dsize=(256, 256), interpolation=cv2.INTER_AREA)
        tmp_m = cv2.resize(masks[i], dsize=(256, 256), interpolation=cv2.INTER_AREA)
        tmp_m = cv2.cvtColor(tmp_m, cv2.COLOR_BGR2GRAY)
        for x in range(tmp_m.shape[0]):
            for y in range(tmp_m.shape[1]):
                if tmp_m[y, x] == 29:
                    tmp_m[y, x] = 255
        fin_images.append(image_transform(tmp_i).float())
        fin_masks.append(image_transform(tmp_m).float())

    return fin_images, fin_masks
