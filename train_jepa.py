"""Train the toy JEPA and print one row per epoch.

  python train_jepa.py --no-ema --no-var-reg     # 1: neither
  python train_jepa.py --no-var-reg              # 2: EMA only
  python train_jepa.py                           # 3: EMA + variance term
  python train_jepa.py --no-var-reg --normalize  # 4: EMA + normalize

Columns:
  enc_std   -> standard deviation of the online encoder's embeddings across a
               batch. Near 0 = collapse.
  probe_err -> pixels of error when a ridge probe on the frozen embeddings
               reads the largest shape's (x, y). On this dataset: always
               predicting the mean gives 5.62 px, an untrained encoder ~3.0 px.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bouncing_shapes import BouncingShapes
from jepa import ToyJEPA, embedding_std, jepa_loss, variance_regularizer


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def split_clip(clip):
    """(B, 12, 1, 32, 32) -> (context frames 0..3, target frames 8..11)."""
    return(clip[:,0:4],clip[:,8:12])


@torch.no_grad()
def probe_eval(model, loader, device, train_batches=40):
    """Ridge-regression probe, closed form: frozen embeddings -> the largest
    shape's (x, y). Returns the mean absolute error in pixels.

    Embeddings of train_batches + 10 batches are collected; the first
    train_batches fit the probe, the last 10 are held out.
    """
    Z, Y = [], []
    for i, (clip, state) in enumerate(loader):
        if i >= train_batches + 10:
            break
        ctx,_ = split_clip(clip.to(device))
        z = model.encoder(ctx)
        Z.append(z.cpu())
        Y.append(state[:,-1,:2])
    Z, Y = torch.cat(Z).double(), torch.cat(Y).double()
    n_train = (len(Y)*train_batches)//(train_batches+10)
    Ztr,Ytr,Zte,Yte = Z[:n_train],Y[:n_train],Z[n_train:],Y[n_train:]
    Ztr = torch.cat([Ztr,torch.ones(len(Ztr),1)],1)
    Zte = torch.cat([Zte,torch.ones(len(Zte),1)],1)
    ridge = 1e-3*torch.eye(Ztr.shape[1])
    W = torch.linalg.solve(Ztr.T@Ztr+ridge,Ztr.T@Ytr)
    return (Zte@W - Yte).abs().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-var-reg", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--var-weight", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"device: {device} | ema={not args.no_ema} var_reg={not args.no_var_reg}")

    ds = BouncingShapes(num_clips=8_000, return_state=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    model = ToyJEPA(dim=args.dim, use_ema=not args.no_ema).to(device)
    # Only parameters that require a gradient: the target encoder is frozen and
    # moves by EMA, not by the optimizer.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)

    print(f"{'epoch':>5} {'loss':>8} {'enc_std':>8} {'probe_err':>9}  (px)")
    for epoch in range(1, args.epochs + 1):
        t0, loss_sum, n = time.perf_counter(), 0.0, 0
        for clip, _ in loader:
            ctx,tgt = split_clip(clip.to(device))
            pred,z_tgt = model(ctx,tgt)
            if args.normalize:
                pred, z_tgt = F.normalize(pred,dim=-1),F.normalize(z_tgt,dim=-1)
            loss = jepa_loss(pred,z_tgt)
            if not args.no_var_reg:
                loss += args.var_weight * variance_regularizer(model.encoder(ctx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if not args.no_ema:
                model.ema_update()
            loss_sum+=loss.item()
            n+=1

        # collapse metric on the online encoder, last batch of the epoch
        with torch.no_grad():
            enc_std = embedding_std(model.encoder(ctx))
        err = probe_eval(model,loader,device)
        print(f"{epoch:>5} {loss_sum/n:>8.4f} {enc_std:>8.3f} {err:>9.2f}  (px)")

    print("\ncollapse: enc_std -> ~0, loss -> 0.0000, probe_err WORSE than an untrained encoder (~3.0px)")
    print("healthy:  enc_std O(1) AND probe_err -> ~1.8px (mean-predictor floor is 5.62px)")


if __name__ == "__main__":
    main()
