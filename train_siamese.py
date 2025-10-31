import os
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from model_siamese import SiameseNet, contrastive_loss
from config import (
    BATCH_SIZE_SIAM,
    EPOCHS_SIAM,
    MARGIN,
)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def accuracy_from_outputs(y_true, y_prob, thresh=0.5):
    y_hat = (y_prob >= thresh).float()
    return (y_hat.eq(y_true)).float().mean().item()

def _split_pairs_batch(xb):
    """
    xb: (N, 2, 1, P, P)  -> returns a=(N,1,P,P), b=(N,1,P,P)
    """
    if xb.dim() != 5 or xb.size(1) != 2:
        raise ValueError("X_pairs must have shape (N, 2, 1, patch, patch).")
    a = xb[:, 0]  # (N,1,P,P)
    b = xb[:, 1]  # (N,1,P,P)
    return a, b

def compile_and_train_once(X_pairs, y_pairs, epochs: int = 1):
    model = SiameseNet().to(DEVICE)
    criterion = contrastive_loss(MARGIN)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    ds = TensorDataset(
        torch.tensor(X_pairs, dtype=torch.float32),    # (N,2,1,P,P)
        torch.tensor(y_pairs, dtype=torch.float32)     # (N,1)
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE_SIAM, shuffle=True)

    model.train()
    last_acc = 0.0
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            a, b = _split_pairs_batch(xb)

            out = model(a, b)  # (N,1), sigmoid probs
            loss = criterion(yb, out)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_acc = accuracy_from_outputs(yb, out)
    return model, last_acc

def train_siamese(X_pairs, y_pairs):
    best_path = "./weights_final/siamese_model.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    for _ in range(100):
        siamese, acc = compile_and_train_once(X_pairs, y_pairs, 1)
        #print(f"Accuracy: {acc:.4f}")
        if acc > 0.70:
            # Full training with validation split
            ds = torch.utils.data.TensorDataset(
                torch.tensor(X_pairs, dtype=torch.float32),
                torch.tensor(y_pairs, dtype=torch.float32)
            )
            val_size = max(1, int(0.2 * len(ds)))
            train_size = len(ds) - val_size
            train_ds, val_ds = random_split(ds, [train_size, val_size])
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE_SIAM, shuffle=True)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE_SIAM)

            criterion = contrastive_loss(MARGIN)
            optim_full = torch.optim.RMSprop(siamese.parameters(), lr=1e-3)

            best_acc = 0.0
            for epoch in range(EPOCHS_SIAM):
                siamese.train()
                for xb, yb in train_dl:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    a, b = _split_pairs_batch(xb)

                    out = siamese(a, b)
                    loss = criterion(yb, out)
                    optim_full.zero_grad(set_to_none=True)
                    loss.backward()
                    optim_full.step()

                # Validation
                siamese.eval()
                with torch.no_grad():
                    accs = []
                    for xb, yb in val_dl:
                        xb = xb.to(DEVICE)
                        yb = yb.to(DEVICE)
                        a, b = _split_pairs_batch(xb)
                        out = siamese(a, b)
                        accs.append(((out >= 0.5).float().eq(yb)).float().mean().item())
                    val_acc = sum(accs) / len(accs)
                    print(f"Epoch {epoch+1}/{EPOCHS_SIAM} - val_acc: {val_acc:.4f}")
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(siamese.state_dict(), best_path)
            break
    return siamese
