"""
Train the fold U-Net (segmentation-models-pytorch, resnet34 encoder) from a
folder with images/ and masks/ (same filenames, PNG/TIF; mask > 0 = fold).

Plain PyTorch loop (no Lightning/albumentations dependency), random crops of
patch_size from the full images, flips/rot90 augmentation, Dice+BCE loss, AMP,
best-by-validation-IoU checkpoint compatible with fold_predict.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    imgs = root / "images"
    msks = root / "masks"
    if not imgs.is_dir():
        # accept ds1/images + ds1/wrinkles style trees
        pairs = []
        for d in sorted(root.iterdir()):
            if (d / "images").is_dir():
                mdir = (d / "masks") if (d / "masks").is_dir() else (d / "wrinkles")
                for p in sorted((d / "images").glob("*.*")):
                    for cand in (mdir / p.name, mdir / p.name.replace("img", "mask", 1)):
                        if cand.is_file():
                            pairs.append((p, cand))
                            break
        return pairs
    pairs = []
    for p in sorted(imgs.glob("*.*")):
        if p.suffix.lower() not in (".png", ".tif", ".tiff", ".jpg"):
            continue
        for cand in (msks / p.name, msks / p.name.replace("img", "mask", 1)):
            if cand.is_file():
                pairs.append((p, cand))
                break
    return pairs


def main() -> int:
    spec = load_spec("fold U-Net training")
    import cv2
    import torch
    import segmentation_models_pytorch as smp

    root = Path(spec["dataset"])
    pairs = find_pairs(root)
    if len(pairs) < 2:
        raise RuntimeError(f"no image/mask pairs found under {root} (expected images/ and masks/ subfolders)")
    seed = int(spec.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(round(0.2 * len(pairs))))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    patch = int(spec.get("patch", 256)); bs = int(spec.get("batch", 16)); epochs = int(spec.get("epochs", 50))
    lr = float(spec.get("lr", 1e-3)); encoder = spec.get("encoder", "resnet34")
    steps = int(spec.get("steps_per_epoch", 0)) or max(20, len(train_pairs) * 4 // bs)
    out = Path(spec["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{len(train_pairs)} train / {len(val_pairs)} val images, patch {patch}, device {device}")

    def load(p: Path, m: Path):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise RuntimeError(f"cannot read {p} / {m}")
        if msk.shape != img.shape:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return img.astype(np.float32) / 255.0, (msk > 0).astype(np.float32)

    train_data = [load(p, m) for p, m in train_pairs]
    val_data = [load(p, m) for p, m in val_pairs]

    def crop(img, msk, aug=True):
        H, W = img.shape
        if H < patch or W < patch:
            ph, pw = max(0, patch - H), max(0, patch - W)
            img = np.pad(img, ((0, ph), (0, pw)), mode="reflect"); msk = np.pad(msk, ((0, ph), (0, pw)), mode="reflect")
            H, W = img.shape
        y = random.randint(0, H - patch); x = random.randint(0, W - patch)
        a = img[y:y + patch, x:x + patch]; b = msk[y:y + patch, x:x + patch]
        if aug:
            if random.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            if random.random() < 0.5:
                a, b = a[::-1], b[::-1]
            k = random.randint(0, 3)
            a, b = np.rot90(a, k), np.rot90(b, k)
            if random.random() < 0.3:
                a = np.clip(a * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08), 0, 1)
        return np.ascontiguousarray(a), np.ascontiguousarray(b)

    def batch_from(data, aug):
        # bias sampling toward images that contain folds
        xs, ys = [], []
        for _ in range(bs):
            for _try in range(4):
                img, msk = random.choice(data)
                a, b = crop(img, msk, aug)
                if b.any() or random.random() < 0.3:
                    break
            xs.append(a); ys.append(b)
        x = torch.from_numpy(np.stack(xs)[:, None]).to(device)
        y = torch.from_numpy(np.stack(ys)[:, None]).to(device)
        return x, y

    init = spec.get("init_checkpoint")
    # ImageNet encoder weights are downloaded from the HF hub on first use; pointless (and a
    # needless internet dependency) when a checkpoint is about to replace them anyway.
    pretrained = bool(spec.get("pretrained", True)) and not init
    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                     in_channels=1, classes=1).to(device)
    if init:
        from feabas_workbench.workers.fold_predict import load_checkpoint
        m0, _ = load_checkpoint(Path(init), device)
        model.load_state_dict(m0.state_dict())
        log(f"initialised from {init}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def loss_fn(logits, y):
        p = torch.sigmoid(logits)
        num = 2 * (p * y).sum(dim=(1, 2, 3)); den = (p + y).sum(dim=(1, 2, 3)) + 1e-6
        return 0.6 * (1 - (num / den).mean()) + 0.4 * bce(logits, y)

    best = -1.0
    hist = []
    t0 = time.time()
    val_batches = max(2, len(val_data) * 2 // bs + 1)
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for _ in range(steps):
            x, y = batch_from(train_data, True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item()
        sched.step()
        model.eval()
        inter = union = 0.0; vl = 0.0
        with torch.no_grad():
            for _ in range(val_batches):
                x, y = batch_from(val_data, False)
                logits = model(x)
                vl += float(loss_fn(logits, y))
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter += float((pred * y).sum()); union += float(((pred + y) > 0).float().sum())
        iou = inter / union if union > 0 else 0.0
        hist.append({"epoch": ep, "train_loss": tl / steps, "val_loss": vl / val_batches, "val_iou": iou})
        progress(ep, epochs, f"epoch {ep}/{epochs} loss {tl / steps:.4f} val_iou {iou:.3f}")
        ck = {"model_state": model.state_dict(), "meta": {"encoder": encoder, "normalization": "div255", "patch": patch,
                                                         "epoch": ep, "val_iou": iou, "dataset": str(root)}}
        torch.save(ck, out / "last.pt")
        if iou > best:
            best = iou
            torch.save(ck, out / "best.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    log(f"done in {(time.time() - t0) / 60:.1f} min, best val IoU {best:.3f}")
    result({"best_iou": best, "checkpoint": str(out / "best.pt")})
    return 0


if __name__ == "__main__":
    run(main)
