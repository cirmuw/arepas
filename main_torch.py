import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from preproc import (
    load_and_preprocess_images,
    prepare_dataset,
    build_hard_example_pairs
)

from train_reconstruction import train_generator, generator, discriminator
from train_siamese import train_siamese

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    BATCH_SIZE_GEN,
    EPOCHS_GEN,
)

def create_placeholder_dataset(X_train, Y_train, batch_size):
    x = X_train.astype('float32')
    y = Y_train.astype('float32')
    # Expecting N,H,W,C with C=1, convert to N,C,H,W and scale to [-1,1]
    x = np.transpose(x, (0,3,1,2))
    y = np.transpose(y, (0,3,1,2))
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def run_inference(preprocessed_input, ground_truth, generator_model):
    """preprocessed_input: (H,W,1) or (1,H,W,1) in [-1,1]."""
    if preprocessed_input.ndim == 3:
        preprocessed_input = preprocessed_input[None, ...]
    # to N,C,H,W
    inp = torch.from_numpy(np.transpose(preprocessed_input, (0,3,1,2))).float()
    with torch.no_grad():
        out = generator_model(inp).cpu().numpy()
    out_img = np.transpose(out[0], (1,2,0))

    fig, axs = plt.subplots(1, 4, figsize=(12.5, 5))
    axs[0].imshow(preprocessed_input[0, ..., 0], cmap='binary'); axs[0].set_title("Input"); axs[0].axis('off')
    axs[1].imshow(out_img[..., 0], cmap='binary_r'); axs[1].set_title("Output"); axs[1].axis('off')
    axs[2].imshow(ground_truth[..., 0], cmap='binary_r'); axs[2].set_title("GT"); axs[2].axis('off')
    axs[3].imshow(np.abs(out_img[..., 0]-ground_truth[..., 0]), cmap='viridis'); axs[3].set_title("Residual"); axs[3].axis('off')
    plt.tight_layout(); plt.show()
    return out_img

def main():
    example_dataset = np.random.rand(100, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    data_train = load_and_preprocess_images(example_dataset, pad_value=0)

    X_train, Y_train = prepare_dataset(data_train, augment=True)
    # X_test, Y_test = prepare_dataset(data_test, augment=False)  # define data_test in your pipeline

    dataloader = create_placeholder_dataset(X_train, Y_train, BATCH_SIZE_GEN)
    gen, disc = train_generator(dataloader, EPOCHS_GEN)

    # Example inference if you have X_test/Y_test prepared:
    # out = run_inference(X_test[0], Y_test[0], gen)

    # Build hard pairs for Siamese
    # X_pairs, y_pairs = build_hard_example_pairs(gen, X_train, Y_train)
    # siamese = train_siamese(X_pairs, y_pairs)

if __name__ == "__main__":
    main()
