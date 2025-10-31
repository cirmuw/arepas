import torch
from torch import nn, optim
from tqdm import tqdm

from config import (
    LAMBDA_L1,
    LAMBDA_VGG,
    LAMBDA_GP,
    BATCH_SIZE_GEN,
)
from model_reconstruction import (
    build_pix2pixhd_generator,
    build_discriminator,
    vgg_loss_fn,
    discriminator_loss,
    generator_loss,
    gradient_penalty,
    VGGFeatureExtractor,
)

# -----------------------------
# Device setup
# -----------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# -----------------------------
# Instantiate models
# -----------------------------
generator = build_pix2pixhd_generator()
discriminator = build_discriminator()
vgg_extractor = VGGFeatureExtractor()

generator.to(device)
discriminator.to(device)
vgg_extractor.to(device)

vgg_loss = vgg_loss_fn(vgg_extractor)

# -----------------------------
# Optimizers
# -----------------------------
gen_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))


# -----------------------------
# Single training step
# -----------------------------
def train_step(input_image, target):
    input_image = input_image.to(device)
    target = target.to(device)

    # ---- Generator step ----
    generator.train()
    discriminator.train()
    gen_optimizer.zero_grad(set_to_none=True)

    fake_output = generator(input_image)
    fake_disc = discriminator(input_image, fake_output)

    gen_gan_loss = generator_loss(fake_disc)
    gen_l1_loss = torch.mean(torch.abs(target - fake_output))
    gen_vgg_loss = vgg_loss(target, fake_output)

    total_gen_loss = gen_gan_loss + (LAMBDA_L1 * gen_l1_loss) + (LAMBDA_VGG * gen_vgg_loss)
    total_gen_loss.backward()
    gen_optimizer.step()

    # ---- Discriminator step ----
    disc_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_output_det = generator(input_image)

    real_disc = discriminator(input_image, target)
    fake_disc = discriminator(input_image, fake_output_det)

    disc_loss_val = discriminator_loss(real_disc, fake_disc)
    gp = gradient_penalty(discriminator, target, fake_output_det, input_image)
    disc_loss = disc_loss_val + LAMBDA_GP * gp
    disc_loss.backward()
    disc_optimizer.step()

    return {
        "gen_total_loss": total_gen_loss.detach().item(),
        "gen_gan_loss": gen_gan_loss.detach().item(),
        "gen_l1_loss": gen_l1_loss.detach().item(),
        "gen_vgg_loss": gen_vgg_loss.detach().item(),
        "disc_loss": disc_loss.detach().item(),
    }


# -----------------------------
# Full training loop
# -----------------------------
def train_generator(dataloader, epochs):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for input_image, target in pbar:
            stats = train_step(input_image, target)
            pbar.set_postfix({
                "G_total": f"{stats['gen_total_loss']:.4f}",
                "G_L1": f"{stats['gen_l1_loss']:.4f}",
                "G_VGG": f"{stats['gen_vgg_loss']:.4f}",
                "D": f"{stats['disc_loss']:.4f}",
            })
    return generator, discriminator


