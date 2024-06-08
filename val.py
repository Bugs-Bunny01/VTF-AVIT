import numpy as np
import cv2

def Val(idxs_to_keep):
    # Gelsight       Realsense
    image = cv2.imread(r"D:\dl\slip_dataset\visual\other\other333\Data0\Sliding\Gelsight\mgw073_30_s10_00006823.jpg")
    image = cv2.resize(image, (320, 240))

    block_size = (12, 16)
    blocks = []
    idxs = []

    for i in range(0, image.shape[0], block_size[0]):
        for j in range(0, image.shape[1], block_size[1]):
            block = image[i:i + block_size[0], j:j + block_size[1]]
            blocks.append(block)

            idxs.append((i // block_size[0], j // block_size[1]))

    kept_blocks = [blocks[idx] for idx in idxs_to_keep]
    kept_idxs = [idxs[idx] for idx in idxs_to_keep]

    new_image = np.zeros_like(image)

    for block, idx in zip(kept_blocks, kept_idxs):
        i, j = idx[0], idx[1]
        new_image[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]] = block

    cv2.imshow("New Image", new_image)
    cv2.imwrite("./81.png", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Val1(idxs_to_keep):
    Val(idxs_to_keep)
def Val2(idxs_to_keep):
    Val(idxs_to_keep)
def Val3(idxs_to_keep):
    Val(idxs_to_keep)

