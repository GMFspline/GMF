import sys
from pathlib import Path

# Add local vendored mamba_ssm (openpoints/.../mamba/mamba_ssm) to sys.path
_root = Path(__file__).resolve().parent
_mamba_dir = _root / "openpoints" / "models" / "PCM" / "mamba"
if str(_mamba_dir) not in sys.path:
    sys.path.insert(0, str(_mamba_dir))

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F
from geomdl import BSpline
import os

import kdatapro
import bspshow
from models import ctrlmam

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def inverse_transform_tensor_points(points_xyz: torch.Tensor, affine_inv_4x4: torch.Tensor) -> torch.Tensor:
    """
    Apply inverse affine transform to point-cloud tensors (row-vector convention).

    Args:
        points_xyz: Shape [N, 3] or [B, N, 3], e.g. [12, 3].
        affine_inv_4x4: Shape [4, 4] or [B, 4, 4]; matches export field transform_affine_inv_4x4.

    Returns:
        Transformed points with the same shape as points_xyz (last dim 3).
    """
    if not isinstance(points_xyz, torch.Tensor):
        raise TypeError(f"points_xyz must be torch.Tensor, got {type(points_xyz)}")
    if not isinstance(affine_inv_4x4, torch.Tensor):
        raise TypeError(f"affine_inv_4x4 must be torch.Tensor, got {type(affine_inv_4x4)}")
    if points_xyz.ndim not in (2, 3) or points_xyz.shape[-1] != 3:
        raise ValueError(f"points_xyz shape must be [N,3] or [B,N,3], got {tuple(points_xyz.shape)}")
    if affine_inv_4x4.ndim not in (2, 3) or affine_inv_4x4.shape[-2:] != (4, 4):
        raise ValueError(f"affine_inv_4x4 shape must be [4,4] or [B,4,4], got {tuple(affine_inv_4x4.shape)}")

    A = affine_inv_4x4.to(device=points_xyz.device, dtype=points_xyz.dtype)

    if points_xyz.ndim == 2:
        ones = torch.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype, device=points_xyz.device)
        points_h = torch.cat([points_xyz, ones], dim=1)
        out_h = points_h @ (A[0] if A.ndim == 3 else A)
        return out_h[:, :3]

    bsz, n_pts, _ = points_xyz.shape
    ones = torch.ones((bsz, n_pts, 1), dtype=points_xyz.dtype, device=points_xyz.device)
    points_h = torch.cat([points_xyz, ones], dim=2)
    if A.ndim == 2:
        A = A.unsqueeze(0).expand(bsz, -1, -1)
    elif A.shape[0] != bsz:
        raise ValueError(f"Batch size mismatch: points B={bsz}, affine B={A.shape[0]}")
    out_h = torch.bmm(points_h, A)
    return out_h[:, :, :3]


def val(models, train_loader, device, numclass):
    models.eval()
    total_loss = 0
    numlen = 0
    for batch_idx, (point, utoken, ctoken, etoken, patten, uatten, catten, eatten, voxel, output) in enumerate(train_loader):
        if batch_idx < 1000:
            numlen += 1
            point, utoken, ctoken, etoken = point.float().to(device), utoken.to(device), ctoken.to(
                device), etoken.to(device)
            patten, uatten, catten, eatten = patten.half().to(device), uatten.to(device), catten.to(device), eatten.to(
                device)

            voxel = voxel.to(device)

            allatten = torch.cat((patten, uatten, catten, eatten, uatten, catten), dim=1)
            logits, knot_vector = models(point, utoken, ctoken, etoken, allatten)

            a = 12
            knot_vector = knot_vector.reshape(-1)
            knot_abs_restored = torch.zeros(28 - a, dtype=torch.float64)
            delta_restored = knot_abs_restored.clone()
            delta_restored[4:25 - a] = knot_vector
            for i in range(4, 25 - a):
                knot_abs_restored[i] = knot_abs_restored[i - 1] + delta_restored[i]
            knot_abs_restored[24 - a:] = 1.0
            knotu = knot_abs_restored.cpu().detach().numpy().tolist()

            coarpo = logits.reshape(36, numclass)
            coarpo = F.softmax(coarpo, 1)
            coarpov = torch.argmax(coarpo, 1, keepdim=False)
            revo = 1 / (numclass - 1)
            inter = revo / 2
            x = (coarpov) * revo + inter

            x = x.reshape(24 - a, 3) - 0.5

            point = point.reshape(-1, 3)
            inp = point.cpu().detach().numpy().tolist()
            nzinput = []
            for i in range(len(inp)):
                if inp[i][0] == 0 and inp[i][1] == 0 and inp[i][2] == 0:
                    break
                nzinput.append(inp[i])

            curve = BSpline.Curve()
            curve.degree = 3
            new_po = x.cpu().detach().numpy()
            po = new_po.tolist()
            po[0] = nzinput[0]

            po.pop()
            po.append(nzinput[len(nzinput) - 1])
            curve.ctrlpts = po

            curve.knotvector = knotu
            step = 0.001
            numstep = int((1 / step) + 1)
            prepo = []
            for uid in range(numstep):
                pu = uid * step
                prepo.append(curve.evaluate_single(pu))
            newx = torch.tensor(prepo).to(device)
            point = torch.tensor(nzinput).to(device)
            A_sq = (point ** 2).sum(dim=1).unsqueeze(1)
            B_sq = (newx ** 2).sum(dim=1).unsqueeze(0)
            cross = point @ newx.t()
            dist_sq = A_sq + B_sq - 2.0 * cross
            dist_sq.clamp_min_(0)
            min_dist_sq, min_indices = dist_sq.min(dim=1)
            min_dist = torch.sqrt(min_dist_sq)
            min_dist = min_dist.mean()
            total_loss += min_dist.item()
            print(min_dist)

            knotu = torch.tensor(knotu).to(device)
            torch.save(knotu, 'knot.pth')
            torch.save(x, 'newpo.pth')
            torch.save(point, 'oripo.pth')
            bspshow.shcurve(numlen)
        else:
            break
    print("Total loss: {}".format(total_loss))

    print("numlen: {}".format(numlen))


def train_model(models, train_loader, test_loader, criterion1, criterion2, optimizer, scheduler, device, numclass, knot_loss_weight=1.0, epochs=10):
    with torch.no_grad():
        val(models, test_loader, device, numclass)
    models.train()
    checkpoint_path = "net"
    checkpoint_path = os.path.join(checkpoint_path, 'init.pt')
    torch.save({"embedding_state_dict": models.embedding.state_dict(),
                "class_head_state_dict": models.class_head.state_dict(),
                "knot_head_state_dict": models.knot_head.state_dict(),
                "conv_state_dict": models.conv_k3.state_dict(),
                "udecoder_state_dict": models.umamba_blocks_list.state_dict(),
                "cdecoder_state_dict": models.cmamba_blocks_list.state_dict(),
                }, checkpoint_path)

    for epoch in range(epochs):
        total_loss = 0.0
        ktotal_loss = 0.0
        for batch_idx, (point, utoken, ctoken, etoken, patten, uatten, catten, eatten, voxel, knot_target) in enumerate(train_loader):
            point, utoken, ctoken, etoken, voxel = point.float().to(device), utoken.to(device), ctoken.to(device), etoken.to(device), voxel.long().to(device)
            knot_target = knot_target.float().to(device)
            patten, uatten, catten, eatten = patten.to(device), uatten.to(device), catten.to(device), eatten.to(device)
            allatten = torch.cat((patten, uatten, catten, eatten, uatten, catten), dim=1)

            logits, knot_vector = models(point, utoken, ctoken, etoken, allatten)

            loss_cls = criterion1(logits.reshape(-1, numclass), voxel.reshape(-1))
            loss_knot = torch.sqrt(criterion2(knot_vector, knot_target))
            loss = loss_cls + loss_knot

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss_cls.item()
            ktotal_loss += loss_knot.item()
            if batch_idx > 500:
                break

        avg_loss = total_loss / 100
        kavg_loss = ktotal_loss / 100
        scheduler.step()
        print(f"Step {epoch}: LR = {optimizer.param_groups[0]['lr']:.8f}")
        print(f"===== Epoch {epoch + 1}/{epochs}, cls Loss: {avg_loss:.4f} , knot Loss: {kavg_loss:.4f}=====")
        if (epoch + 1) % 10 == 0:
            checkpoint_path = "net"
            checkpoint_path = os.path.join(checkpoint_path, "12vo" + str(int(epoch)) + '.pt')
            torch.save({"embedding_state_dict": models.embedding.state_dict(),
                        "class_head_state_dict": models.class_head.state_dict(),
                        "knot_head_state_dict": models.knot_head.state_dict(),
                        "conv_state_dict": models.conv_k3.state_dict(),
                        "udecoder_state_dict": models.umamba_blocks_list.state_dict(),
                        "cdecoder_state_dict": models.cmamba_blocks_list.state_dict(),
                        }, checkpoint_path)

        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                val(models, test_loader, device, numclass)


if __name__ == "__main__":
    os.chdir(_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    numclass = 1001
    models = ctrlmam.PointCloudClassifier(model_name="bert-base-uncased", numclass=numclass).to(device)
    checkpoint = torch.load('net/24k110.pt')
    models.embedding.load_state_dict(checkpoint['embedding_state_dict'])
    models.class_head.load_state_dict(checkpoint['class_head_state_dict'])
    models.conv_k3.load_state_dict(checkpoint['conv_state_dict'])
    models.cmamba_blocks_list.load_state_dict(checkpoint['cdecoder_state_dict'])
    models.umamba_blocks_list.load_state_dict(checkpoint['udecoder_state_dict'])
    models.knot_head.load_state_dict(checkpoint['knot_head_state_dict'])
    optimizer = optim.AdamW(
        list(models.embedding.parameters()) + list(models.class_head.parameters()) + list(models.conv_k3.parameters())
        + list(models.umamba_blocks_list.parameters()) + list(models.cmamba_blocks_list.parameters()) + list(models.knot_head.parameters()),
        lr=7e-5,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1.0,
        total_iters=5
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=1000,
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[10]
    )
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    train_dataset = kdatapro.TrainclassDataset(numclass=numclass,
        in_path='/mnt/d/data/curfit/4w/input',
        uptxt_path='/mnt/d/data/curfit/4w/uptxt',
        curtxt_path='/mnt/d/data/curfit/4w/curtxt',
        out_path='/mnt/d/data/curfit/4w/voutput', knot_path='/mnt/d/data/curfit/4w/output')
    num_len = len(train_dataset)
    print(num_len)
    test_dataset = kdatapro.ShapeDataset(numclass=numclass,
        in_path='/mnt/d/code/larmo/fitdata/outputs/missing_boundary_curves',
        uptxt_path='/mnt/d/code/larmo/fitdata/outputs/uptxt',
        curtxt_path='/mnt/d/code/larmo/fitdata/outputs/curtxt',
        )

    train_loader = DataLoader(train_dataset, batch_size=44, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("\n===== Start Training =====")
    train_model(models, train_loader, test_loader, criterion1, criterion2, optimizer, scheduler, device, numclass=numclass, knot_loss_weight=1.0, epochs=500)
