import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------
# Image and coordinate utilities
# ---------------------------------------------------------------------

def load_nifti(path, device, is_label=False):
    nii = nib.load(str(path))
    arr = np.asarray(nii.dataobj)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 3D NIfTI volume, got shape {arr.shape}: {path}"
        )

    if is_label:
        arr = np.rint(arr).astype(np.int32)
    else:
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Robust intensity normalization. NCC is insensitive to global
        # affine intensity changes, but clipping reduces extreme outliers.
        foreground = arr[np.isfinite(arr) & (arr != 0)]
        if foreground.size > 100:
            lo, hi = np.percentile(foreground, [1.0, 99.0])
            if hi > lo:
                arr = np.clip(arr, lo, hi)

        nonzero = arr != 0
        if nonzero.any():
            vals = arr[nonzero]
            mean = float(vals.mean())
            std = float(vals.std())
            if std > 1e-6:
                arr[nonzero] = (arr[nonzero] - mean) / std
        arr[~np.isfinite(arr)] = 0.0

    tensor = torch.from_numpy(arr.copy()).to(device=device)
    tensor = tensor[None, None].float()
    return tensor, nii, arr


def make_coordinate_grid(shape, device, dtype=torch.float32):
    """
    Create normalized coordinates in [D,H,W,3], with channels (x,y,z).
    """
    D, H, W = shape

    z = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)


def get_spacing(nii):
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    if len(spacing) != 3 or any((not np.isfinite(v) or v <= 0) for v in spacing):
        return (1.0, 1.0, 1.0)
    return spacing


def get_physical_flow_scale(shape, spacing):
    # NIfTI array shape is kept in native axis order.
    # grid_sample x,y,z correspond to array axes 2,1,0.
    D, H, W = shape
    sx, sy, sz = spacing

    return torch.tensor(
        [
            max(W - 1, 1) * 0.5 * sz,
            max(H - 1, 1) * 0.5 * sy,
            max(D - 1, 1) * 0.5 * sx,
        ],
        dtype=torch.float32,
    )


def sample_volume(volume, grid, mode="bilinear", padding_mode="border"):
    return F.grid_sample(
        volume,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def get_random_patch_slices(shape, patch_size):
    D, H, W = shape
    pd = min(int(patch_size), D)
    ph = min(int(patch_size), H)
    pw = min(int(patch_size), W)

    sd = random.randint(0, max(D - pd, 0))
    sh = random.randint(0, max(H - ph, 0))
    sw = random.randint(0, max(W - pw, 0))

    return (
        slice(sd, sd + pd),
        slice(sh, sh + ph),
        slice(sw, sw + pw),
    )


def patch_coords_from_grid(coord_grid, slices):
    coords = coord_grid[slices].reshape(-1, 3)
    return coords.detach().clone().requires_grad_(True)


def extract_patch(volume, slices):
    return volume[:, :, slices[0], slices[1], slices[2]]


# ---------------------------------------------------------------------
# Deformation-adapted wavelet representation
# ---------------------------------------------------------------------

def make_wavelet_centres(scales, device):
    """
    Generate fixed centres for multiscale localized wavelet atoms.

    The centres are distributed uniformly in normalized coordinate space.
    Each scale contributes a separate set of centres.
    """
    centres = []
    sigmas = []

    for n in scales:
        if n < 2:
            raise ValueError("Each wavelet scale must contain at least 2 centres.")

        axis = torch.linspace(-1.0, 1.0, n, device=device)
        zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
        c = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)

        # Width proportional to spacing between centres.
        spacing = 2.0 / (n - 1)
        sigma = 0.85 * spacing

        centres.append(c)
        sigmas.append(
            torch.full((c.shape[0], 1), sigma, device=device)
        )

    return torch.cat(centres, dim=0), torch.cat(sigmas, dim=0)


class ModulusMLP(nn.Module):
    """Coordinate-dependent effective shear modulus."""

    def __init__(
        self,
        hidden_features=64,
        hidden_layers=3,
        mu_min=0.25,
        mu_max=4.0,
    ):
        super().__init__()
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)

        layers = []
        in_features = 3

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_features))
            layers.append(nn.SiLU())
            in_features = hidden_features

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

        # Initialize the final layer near the middle of the allowed range.
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, coords):
        raw = self.network(coords)
        return self.mu_min + (self.mu_max - self.mu_min) * torch.sigmoid(raw)


class DeformationWaveletINR(nn.Module):
    """
    A learnable multiscale wavelet expansion for a pairwise deformation.

    Each atom is a 3D Ricker/Mexican-hat-like function:
        psi(r) = (3 - ||r||^2) exp(-||r||^2 / 2)

    where r = (x - centre) / sigma.

    A low-order affine component is included so that global translation,
    rotation-like linear trends, and broad deformation do not have to be
    represented exclusively by localized wavelets.

    The wavelet coefficients are directly optimized for each image pair.
    The field is an implicit coordinate function, although its displacement
    parameters are basis coefficients rather than the weights of a deep
    displacement MLP.
    """

    def __init__(
        self,
        scales=(3, 5, 7),
        mu_min=0.25,
        mu_max=4.0,
        modulus_hidden=64,
        modulus_layers=3,
        atom_chunk_size=256,
    ):
        super().__init__()

        self.scales = tuple(int(s) for s in scales)
        self.atom_chunk_size = int(atom_chunk_size)

        centres, sigmas = make_wavelet_centres(self.scales, device="cpu")
        self.register_buffer("centres", centres)
        self.register_buffer("sigmas", sigmas)

        self.n_atoms = int(centres.shape[0])

        # Polynomial/affine coefficients: [constant, x, y, z] -> 3D flow.
        self.affine_coefficients = nn.Parameter(torch.zeros(4, 3))

        # One vector coefficient per wavelet atom.
        self.wavelet_coefficients = nn.Parameter(
            torch.zeros(self.n_atoms, 3)
        )

        # A small random initialization breaks perfect symmetry while keeping
        # the initial deformation very close to identity.
        nn.init.normal_(self.wavelet_coefficients, mean=0.0, std=1e-5)

        self.modulus_net = ModulusMLP(
            hidden_features=modulus_hidden,
            hidden_layers=modulus_layers,
            mu_min=mu_min,
            mu_max=mu_max,
        )

    def wavelet_features(self, coords):
        """
        Args:
            coords: [N,3]

        Returns:
            phi: [N, n_atoms]

        Atoms are normalized pointwise by the square root of their summed
        squared response. This moderates scale-dependent changes in basis
        amplitude where multiple atoms overlap.
        """
        feature_chunks = []

        centres = self.centres.to(dtype=coords.dtype)
        sigmas = self.sigmas.to(dtype=coords.dtype)

        for start in range(0, self.n_atoms, self.atom_chunk_size):
            end = min(start + self.atom_chunk_size, self.n_atoms)

            c = centres[start:end]
            s = sigmas[start:end]

            r = (coords[:, None, :] - c[None, :, :]) / s[None, :, :]
            r2 = torch.sum(r * r, dim=-1)

            psi = (3.0 - r2) * torch.exp(-0.5 * r2)
            feature_chunks.append(psi)

        phi = torch.cat(feature_chunks, dim=1)

        # Stabilize feature scale at each coordinate.
        norm = torch.sqrt(torch.sum(phi * phi, dim=1, keepdim=True) + 1e-6)
        phi = phi / norm

        return phi

    def forward_fields(self, coords):
        """
        Args:
            coords: [N,3] normalized (x,y,z) coordinates

        Returns:
            flow: [N,3] normalized sampling-grid offsets
            mu:   [N,1] effective shear modulus
        """
        ones = torch.ones_like(coords[:, :1])
        affine_input = torch.cat((ones, coords), dim=1)

        affine_flow = affine_input @ self.affine_coefficients
        wavelet_flow = self.wavelet_features(coords) @ self.wavelet_coefficients

        flow = affine_flow + wavelet_flow
        mu = self.modulus_net(coords)

        return flow, mu

    def forward(self, coords):
        return self.forward_fields(coords)


# ---------------------------------------------------------------------
# Registration losses
# ---------------------------------------------------------------------

def local_ncc_loss(fixed, warped, win=9, eps=1e-5):
    """
    Local NCC loss for 3D tensors [1,1,D,H,W].
    Returns 1 - mean(local NCC).
    """
    win = int(win)
    if win % 2 == 0:
        win += 1

    padding = win // 2

    mean_f = F.avg_pool3d(
        fixed, win, stride=1, padding=padding, count_include_pad=False
    )
    mean_w = F.avg_pool3d(
        warped, win, stride=1, padding=padding, count_include_pad=False
    )

    mean_ff = F.avg_pool3d(
        fixed * fixed, win, stride=1, padding=padding, count_include_pad=False
    )
    mean_ww = F.avg_pool3d(
        warped * warped, win, stride=1, padding=padding, count_include_pad=False
    )
    mean_fw = F.avg_pool3d(
        fixed * warped, win, stride=1, padding=padding, count_include_pad=False
    )

    var_f = torch.clamp(mean_ff - mean_f * mean_f, min=0.0)
    var_w = torch.clamp(mean_ww - mean_w * mean_w, min=0.0)
    cov_fw = mean_fw - mean_f * mean_w

    ncc = cov_fw / torch.sqrt(var_f * var_w + eps)
    return 1.0 - ncc.mean()


def flow_to_grid(flow, coords):
    return (coords + flow).reshape(
        1,
        *infer_patch_shape(coords),
        3,
    )


def infer_patch_shape(coords):
    """
    The flattened patch coordinate count is not sufficient to infer a
    non-cubic patch robustly. This helper is only retained for cubic patches.
    """
    n = coords.shape[0]
    p = round(n ** (1.0 / 3.0))
    if p ** 3 != n:
        raise ValueError(
            "This script expects cubic training patches; "
            f"cannot infer cubic shape from {n} coordinates."
        )
    return (p, p, p)


def smoothness_loss(flow_grid):
    """
    First-order displacement smoothness.
    flow_grid: [1,3,D,H,W]
    """
    dx = flow_grid[..., 1:] - flow_grid[..., :-1]
    dy = flow_grid[..., 1:, :] - flow_grid[..., :-1, :]
    dz = flow_grid[..., 1:, :, :] - flow_grid[..., :-1, :, :]

    return (
        dx.square().mean()
        + dy.square().mean()
        + dz.square().mean()
    ) / 3.0


def inverse_consistency_loss(model_xy, model_yx, coords, flow_xy, flow_yx):
    """
    Bidirectional inverse consistency at corresponding points.

    This evaluates each model at the transformed coordinates. It does not
    interpolate a separately rasterized reverse field.
    """
    reverse_at_xy, _ = model_yx(coords + flow_xy)
    forward_at_yx, _ = model_xy(coords + flow_yx)

    loss_xy = (flow_xy + reverse_at_xy).square().sum(dim=-1).mean()
    loss_yx = (flow_yx + forward_at_yx).square().sum(dim=-1).mean()

    return 0.5 * (loss_xy + loss_yx)


def hyperelastic_terms(
    coords,
    flow,
    mu,
    physical_flow_scale,
    coord_to_mm_scale,
    lame_lambda=1.0,
    eps=1e-6,
):
    """
    Compute compressible Neo-Hookean energy and first Piola stress at sampled
    points.

    flow is in normalized grid offsets. We first convert it to millimetres,
    then differentiate with respect to physical coordinates.

    coords ordering is (x,y,z). coord_to_mm_scale gives:
        d(normalized x)/d(mm x), etc.
    """
    flow_scale = physical_flow_scale.to(
        device=flow.device, dtype=flow.dtype
    )
    u_mm = flow * flow_scale[None, :]

    # Compute du_i / d(normalized coordinate j), then convert to du_i/dx_j
    # in physical units.
    grad_rows = []
    for i in range(3):
        grad_i = torch.autograd.grad(
            outputs=u_mm[:, i].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        grad_rows.append(grad_i)

    # [N, displacement_component, coordinate_component]
    grad_u_norm = torch.stack(grad_rows, dim=1)

    coord_scale = coord_to_mm_scale.to(
        device=flow.device, dtype=flow.dtype
    )
    grad_u = grad_u_norm * coord_scale.view(1, 1, 3)

    eye = torch.eye(3, device=flow.device, dtype=flow.dtype)[None]
    Fmat = eye + grad_u

    detF = torch.linalg.det(Fmat)

    # A clamp keeps the logarithm finite. The separate Jacobian penalty
    # handles non-positive or very small determinants.
    det_safe = torch.clamp(detF, min=eps)

    C = Fmat.transpose(1, 2) @ Fmat
    I1 = torch.diagonal(C, dim1=1, dim2=2).sum(dim=-1)

    mu_flat = mu.reshape(-1)
    logJ = torch.log(det_safe)

    # Compressible Neo-Hookean strain-energy density.
    energy = (
        0.5 * mu_flat * (I1 - 3.0)
        - mu_flat * logJ
        + 0.5 * lame_lambda * logJ.square()
    )

    # First Piola stress:
    # P = mu F + (lambda log(J) - mu) F^{-T}
    inv_F_T = torch.linalg.inv(Fmat.transpose(1, 2))
    stress = (
        mu_flat[:, None, None] * Fmat
        + (
            lame_lambda * logJ - mu_flat
        )[:, None, None] * inv_F_T
    )

    stress_norm = torch.linalg.matrix_norm(stress, ord="fro", dim=(-2, -1))

    return {
        "energy": energy.mean(),
        "detF": detF,
        "stress_norm": stress_norm.mean(),
        "F": Fmat,
    }


def stiffness_regularization(mu, mu_min, mu_max):
    """
    Regularize log(mu) toward the midpoint of the permitted range.
    """
    mu = torch.clamp(mu, min=mu_min, max=mu_max)
    target = math.sqrt(mu_min * mu_max)
    return (torch.log(mu) - math.log(target)).square().mean()


def jacobian_penalty(detF, threshold=0.01):
    """
    Penalize non-positive and near-zero Jacobian determinants.
    """
    return F.relu(threshold - detF).square().mean()


# ---------------------------------------------------------------------
# Full-volume inference and metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_fields(model, coord_grid, chunk_size=32768):
    coords = coord_grid.reshape(-1, 3)
    flow_chunks = []
    mu_chunks = []

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        flow, mu = model.forward_fields(coords[start:end])
        flow_chunks.append(flow)
        mu_chunks.append(mu)

    flow = torch.cat(flow_chunks, dim=0).reshape(*coord_grid.shape[:-1], 3)
    mu = torch.cat(mu_chunks, dim=0).reshape(*coord_grid.shape[:-1])

    return flow, mu


@torch.no_grad()
def warp_full_volume(moving, coord_grid, flow):
    grid = (coord_grid + flow).unsqueeze(0)
    warped = sample_volume(moving, grid, mode="bilinear")
    return warped


@torch.no_grad()
def warp_labels(moving_label, coord_grid, flow):
    grid = (coord_grid + flow).unsqueeze(0)
    warped = sample_volume(
        moving_label.float(),
        grid,
        mode="nearest",
        padding_mode="border",
    )
    return warped.round().to(torch.int32)


def dice_score(fixed_label, moving_label_warped):
    fixed = np.asarray(fixed_label).astype(np.int32)
    moving = np.asarray(moving_label_warped).astype(np.int32)

    labels = np.union1d(np.unique(fixed), np.unique(moving))
    labels = labels[labels != 0]

    if labels.size == 0:
        return float("nan"), {}

    per_label = {}
    for label in labels:
        a = fixed == label
        b = moving == label
        denom = int(a.sum() + b.sum())

        if denom == 0:
            continue

        per_label[int(label)] = float(2.0 * np.logical_and(a, b).sum() / denom)

    if not per_label:
        return float("nan"), {}

    return float(np.mean(list(per_label.values()))), per_label


def hd95_score(fixed_label, moving_label_warped, spacing=(1.0, 1.0, 1.0)):
    """
    Mean HD95 across non-background labels present in either image.
    Spacing is supplied in NIfTI array-axis order (D,H,W).
    """
    fixed = np.asarray(fixed_label).astype(np.int32)
    moving = np.asarray(moving_label_warped).astype(np.int32)

    labels = np.union1d(np.unique(fixed), np.unique(moving))
    labels = labels[labels != 0]

    values = []

    for label in labels:
        a = fixed == label
        b = moving == label

        if not a.any() and not b.any():
            continue
        if not a.any() or not b.any():
            values.append(float("inf"))
            continue

        da = distance_transform_edt(~a, sampling=spacing)
        db = distance_transform_edt(~b, sampling=spacing)

        surface_a = a & (distance_transform_edt(a, sampling=spacing) <= min(spacing))
        surface_b = b & (distance_transform_edt(b, sampling=spacing) <= min(spacing))

        d_ab = db[surface_a]
        d_ba = da[surface_b]

        if d_ab.size == 0 or d_ba.size == 0:
            values.append(float("inf"))
        else:
            values.append(
                float(np.percentile(np.concatenate((d_ab, d_ba)), 95))
            )

    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("inf")

    return float(np.mean(finite))


def jacobian_determinant_numpy(flow_normalized, shape, spacing):
    """
    Compute det(I + grad(u)) using finite differences in physical space.

    flow_normalized: [D,H,W,3], channels ordered (x,y,z).
    spacing: (sx,sy,sz) from NIfTI header.
    """
    D, H, W = shape
    sx, sy, sz = spacing

    scale = np.array(
        [
            max(W - 1, 1) * 0.5 * sx,
            max(H - 1, 1) * 0.5 * sy,
            max(D - 1, 1) * 0.5 * sz,
        ],
        dtype=np.float32,
    )

    u = flow_normalized * scale.reshape(1, 1, 1, 3)

    # np.gradient returns derivatives in array-axis order: D, H, W.
    # Convert these into coordinate order x, y, z.
    grad_u = np.zeros((D, H, W, 3, 3), dtype=np.float32)

    for component in range(3):
        du_dz, du_dy, du_dx = np.gradient(
            u[..., component],
            sz,
            sy,
            sx,
            edge_order=1,
        )
        grad_u[..., component, 0] = du_dx
        grad_u[..., component, 1] = du_dy
        grad_u[..., component, 2] = du_dz

    F_np = grad_u + np.eye(3, dtype=np.float32)
    return np.linalg.det(F_np)



# ---------------------------------------------------------------------
# Pairwise optimization: single-direction deformation INR
# ---------------------------------------------------------------------

def optimize_pair(
    fixed,
    moving,
    fixed_nii,
    moving_nii,
    fixed_label=None,
    moving_label=None,
    output_dir="outputs",
    subject_id="subject",
    args=None,
):
    """
    Optimize one deformation INR for pairwise image registration.

    Registration direction:
        moving -> fixed

    The model predicts a spatially varying deformation field and an
    effective stiffness field. Only one model is optimized.

    Retained:
        - Local NCC similarity
        - Deformation smoothness
        - Finite-strain hyperelastic regularization
        - Jacobian determinant penalty
        - Stiffness regularization
        - Full-volume inference and evaluation

    Removed:
        - Backward deformation model
        - Inverse-consistency loss
        - Backward image warping and inference

    Assumes the existing helper functions use the same coordinate,
    displacement, and tensor conventions as the original implementation.
    """

    device = fixed.device

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Validate inputs and initialize spatial information
    # -------------------------------------------------------------
    shape = tuple(int(v) for v in fixed.shape[-3:])

    if tuple(moving.shape[-3:]) != shape:
        raise ValueError(
            f"Fixed/moving shape mismatch: {shape} vs "
            f"{tuple(moving.shape[-3:])}. "
            "Resample one volume to the other before training."
        )

    if fixed.ndim != 5 or moving.ndim != 5:
        raise ValueError(
            "Expected fixed and moving tensors with shape [B,C,D,H,W]."
        )

    if fixed.shape[:2] != moving.shape[:2]:
        raise ValueError(
            "Fixed and moving batch/channel dimensions must match."
        )

    spacing = get_spacing(fixed_nii)
    flow_scale = get_physical_flow_scale(shape, spacing).to(device)

    # d(normalized coordinate) / d(mm coordinate), in x,y,z order.
    D, H, W = shape
    sx, sy, sz = spacing

    coord_to_mm_scale = torch.tensor(
        [
            2.0 / (max(W - 1, 1) * sx),
            2.0 / (max(H - 1, 1) * sy),
            2.0 / (max(D - 1, 1) * sz),
        ],
        dtype=torch.float32,
        device=device,
    )

    coord_grid = make_coordinate_grid(shape, device)

    # -------------------------------------------------------------
    # Initialize a single deformation INR
    # -------------------------------------------------------------
    model = DeformationWaveletINR(
        scales=args.wavelet_scales,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        modulus_hidden=args.modulus_hidden,
        modulus_layers=args.modulus_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.steps, 1),
        eta_min=args.lr * 0.05,
    )

    checkpoint_path = (
        output_dir / f"{subject_id}_checkpoint.pt"
    )
    log_path = output_dir / f"{subject_id}_losses.csv"

    loss_names = [
        "step",
        "total",
        "ncc",
        "smooth",
        "hyperelastic",
        "jacobian",
        "mu_reg",
        "stress_norm",
        "lr",
    ]

    start_time = time.time()

    # -------------------------------------------------------------
    # Pairwise optimization
    # -------------------------------------------------------------
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=loss_names,
        )
        writer.writeheader()

        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Sample a random spatial patch.
            slices = get_random_patch_slices(
                shape,
                args.patch_size,
            )

            fixed_patch = extract_patch(fixed, slices)
            moving_patch = extract_patch(moving, slices)

            coords = patch_coords_from_grid(
                coord_grid,
                slices,
            )

            patch_shape = tuple(
                int(v) for v in fixed_patch.shape[-3:]
            )

            # This implementation retains the original cubic-patch
            # assumption of the wavelet training grid.
            if len(set(patch_shape)) != 1:
                raise ValueError(
                    "The current implementation expects cubic "
                    f"training patches. Got {patch_shape}. "
                    "Use compatible patch dimensions or adapt "
                    "the wavelet-grid implementation."
                )

            # -----------------------------------------------------
            # Predict one deformation field and stiffness field
            # -----------------------------------------------------
            flow, mu = model.forward_fields(coords)

            grid = (
                coords + flow
            ).reshape(1, *patch_shape, 3)

            # Warp moving into fixed coordinates.
            warped_moving = sample_volume(
                moving,
                grid,
            )

            # -----------------------------------------------------
            # Image similarity
            # -----------------------------------------------------
            loss_ncc = local_ncc_loss(
                fixed_patch,
                warped_moving,
                args.ncc_win,
            )

            # -----------------------------------------------------
            # Deformation smoothness
            # -----------------------------------------------------
            flow_grid = (
                flow.reshape(*patch_shape, 3)
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
            )

            loss_smooth = smoothness_loss(flow_grid)

            # -----------------------------------------------------
            # Finite-strain hyperelastic physics
            # -----------------------------------------------------
            # Use a subsample of spatial coordinates to limit
            # the memory cost of coordinate derivatives.
            n_points = min(
                args.physics_points,
                coords.shape[0],
            )

            point_ids = torch.randperm(
                coords.shape[0],
                device=device,
            )[:n_points]

            physics_coords = (
                coords[point_ids]
                .detach()
                .clone()
                .requires_grad_(True)
            )

            physics_flow, physics_mu = model.forward_fields(
                physics_coords
            )

            phys = hyperelastic_terms(
                physics_coords,
                physics_flow,
                physics_mu,
                flow_scale,
                coord_to_mm_scale,
                lame_lambda=args.lame_lambda,
            )

            loss_hyperelastic = phys["energy"]

            loss_jacobian = jacobian_penalty(
                phys["detF"],
                args.jacobian_threshold,
            )

            # -----------------------------------------------------
            # Stiffness-field regularization
            # -----------------------------------------------------
            loss_mu = stiffness_regularization(
                mu,
                args.mu_min,
                args.mu_max,
            )

            # -----------------------------------------------------
            # Total objective
            # -----------------------------------------------------
            loss_total = (
                args.lambda_ncc * loss_ncc
                + args.lambda_smooth * loss_smooth
                + args.lambda_physics * loss_hyperelastic
                + args.lambda_jacobian * loss_jacobian
                + args.lambda_mu * loss_mu
            )

            loss_total.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                )

            optimizer.step()
            scheduler.step()

            # -----------------------------------------------------
            # Logging
            # -----------------------------------------------------
            stress_norm = phys["stress_norm"].detach()

            row = {
                "step": step,
                "total": float(
                    loss_total.detach().cpu()
                ),
                "ncc": float(
                    loss_ncc.detach().cpu()
                ),
                "smooth": float(
                    loss_smooth.detach().cpu()
                ),
                "hyperelastic": float(
                    loss_hyperelastic.detach().cpu()
                ),
                "jacobian": float(
                    loss_jacobian.detach().cpu()
                ),
                "mu_reg": float(
                    loss_mu.detach().cpu()
                ),
                "stress_norm": float(
                    stress_norm.cpu()
                ),
                "lr": optimizer.param_groups[0]["lr"],
            }

            writer.writerow(row)
            f.flush()

            if (
                step == 1
                or step % args.log_every == 0
                or step == args.steps
            ):
                print(
                    f"[{subject_id}] "
                    f"step {step:05d}/{args.steps} "
                    f"total={row['total']:.5f} "
                    f"NCC={row['ncc']:.5f} "
                    f"smooth={row['smooth']:.6f} "
                    f"neo={row['hyperelastic']:.6f} "
                    f"jac={row['jacobian']:.6f} "
                    f"mu={row['mu_reg']:.6f}"
                )

            # -----------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------
            if (
                step % args.save_every == 0
                or step == args.steps
            ):
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

    # -------------------------------------------------------------
    # Full-volume inference
    # -------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        flow, mu = predict_fields(
            model,
            coord_grid,
            chunk_size=args.inference_chunk,
        )

        warped_moving = warp_full_volume(
            moving,
            coord_grid,
            flow,
        )

    flow_np = (
        flow.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    mu_np = (
        mu.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    warped_moving_np = (
        warped_moving[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Convert normalized grid displacement into millimetres.
    flow_scale_np = flow_scale.detach().cpu().numpy()

    flow_mm = (
        flow_np
        * flow_scale_np.reshape(1, 1, 1, 3)
    ).astype(np.float32)

    # -------------------------------------------------------------
    # Save registration outputs
    # -------------------------------------------------------------
    nib.save(
        nib.Nifti1Image(
            flow_np,
            fixed_nii.affine,
            fixed_nii.header,
        ),
        str(
            output_dir
            / f"{subject_id}_flow_normalized.nii.gz"
        ),
    )

    nib.save(
        nib.Nifti1Image(
            flow_mm,
            fixed_nii.affine,
            fixed_nii.header,
        ),
        str(
            output_dir
            / f"{subject_id}_flow_mm.nii.gz"
        ),
    )

    nib.save(
        nib.Nifti1Image(
            warped_moving_np,
            fixed_nii.affine,
            fixed_nii.header,
        ),
        str(
            output_dir
            / f"{subject_id}_warped_moving.nii.gz"
        ),
    )

    nib.save(
        nib.Nifti1Image(
            mu_np,
            fixed_nii.affine,
            fixed_nii.header,
        ),
        str(
            output_dir
            / f"{subject_id}_effective_mu.nii.gz"
        ),
    )

    # -------------------------------------------------------------
    # Jacobian and folding evaluation
    # -------------------------------------------------------------
    det_np = jacobian_determinant_numpy(
        flow_np,
        shape,
        spacing,
    )

    folding_fraction = float(
        np.mean(det_np <= 0.0)
    )
    det_mean = float(np.mean(det_np))
    det_min = float(np.min(det_np))

    nib.save(
        nib.Nifti1Image(
            det_np.astype(np.float32),
            fixed_nii.affine,
            fixed_nii.header,
        ),
        str(
            output_dir
            / f"{subject_id}_jacobian_det.nii.gz"
        ),
    )

    metrics = {
        "subject": subject_id,
        "dice": float("nan"),
        "hd95_mm": float("nan"),
        "folding_fraction": folding_fraction,
        "jacobian_mean": det_mean,
        "jacobian_min": det_min,
        "time_seconds": float(
            time.time() - start_time
        ),
    }

    # -------------------------------------------------------------
    # Optional label evaluation
    # -------------------------------------------------------------
    if (
        fixed_label is not None
        and moving_label is not None
    ):
        warped_moving_label = warp_labels(
            moving_label,
            coord_grid,
            flow,
        )

        fixed_label_np = (
            fixed_label[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        warped_label_np = (
            warped_moving_label[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        mean_dice, per_label_dice = dice_score(
            fixed_label_np,
            warped_label_np,
        )

        mean_hd95 = hd95_score(
            fixed_label_np,
            warped_label_np,
            spacing=(
                spacing[2],
                spacing[1],
                spacing[0],
            ),
        )

        metrics["dice"] = mean_dice
        metrics["hd95_mm"] = mean_hd95

        nib.save(
            nib.Nifti1Image(
                warped_label_np.astype(np.int16),
                fixed_nii.affine,
                fixed_nii.header,
            ),
            str(
                output_dir
                / f"{subject_id}_warped_label.nii.gz"
            ),
        )

        with open(
            output_dir
            / f"{subject_id}_per_label_dice.json",
            "w",
        ) as f:
            json.dump(
                per_label_dice,
                f,
                indent=2,
            )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------
    print(
        f"[{subject_id}] finished: "
        f"Dice={metrics['dice']:.4f}, "
        f"HD95={metrics['hd95_mm']:.3f} mm, "
        f"folding={metrics['folding_fraction']:.5f}, "
        f"time={metrics['time_seconds']:.1f}s"
    )

    return metrics


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------

def extract_subject_id(path):
    match = re.search(r"(S\d+)", Path(path).name, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def find_label_file(label_dir, subject_id):
    if label_dir is None or not Path(label_dir).exists():
        return None

    candidates = [
        Path(label_dir) / f"{subject_id}.delineation.structure.label.nii.gz",
        Path(label_dir) / f"{subject_id}.label.nii.gz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(Path(label_dir).glob(f"{subject_id}*label*.nii.gz"))
    if matches:
        return matches[0]

    return None


def discover_images(input_dir, fixed_path):
    input_dir = Path(input_dir)
    fixed_path = Path(fixed_path).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    paths = []
    for path in sorted(input_dir.glob("*.nii.gz")):
        if path.resolve() == fixed_path:
            continue

        name = path.name.lower()
        if "label" in name or "seg" in name:
            continue

        subject_id = extract_subject_id(path)
        if subject_id is None:
            continue

        paths.append((subject_id, path))

    # Keep the first file for each subject ID.
    unique = {}
    for subject_id, path in paths:
        unique.setdefault(subject_id, path)

    return sorted(unique.items())


# ---------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Multiscale deformation-wavelet INR for LPBA40 registration"
    )

    parser.add_argument(
        "--datapath",
        type=str,
        default=(
            "/workspace/DIRAC/Data/LPBA40/"
            "LPBA40_delineation/delineation_l_norm/train"
        ),
    )
    parser.add_argument(
        "--testpath",
        type=str,
        default=(
            "/workspace/DIRAC/Data/LPBA40/"
            "LPBA40_delineation/delineation_l_norm/test"
        ),
    )
    parser.add_argument(
        "--fixed",
        type=str,
        default=(
            "/workspace/DIRAC/Data/LPBA40/"
            "LPBA40_delineation/delineation_l_norm/fixed.nii.gz"
        ),
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=(
            "/workspace/DIRAC/Data/LPBA40/"
            "LPBA40_delineation/label"
        ),
    )
    parser.add_argument(
        "--fixed_label",
        type=str,
        default=(
            "/workspace/DIRAC/Data/LPBA40/"
            "LPBA40_delineation/label/"
            "S01.delineation.structure.label.nii.gz"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/DIRAC/Results/wavelet_inr_lpba40",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Optimization
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--physics_points", type=int, default=1024)
    parser.add_argument("--inference_chunk", type=int, default=8192)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=250)

    # Wavelet representation
    parser.add_argument(
        "--wavelet_scales",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Number of atom centres per axis at each scale.",
    )

    # Modulus network
    parser.add_argument("--modulus_hidden", type=int, default=64)
    parser.add_argument("--modulus_layers", type=int, default=3)
    parser.add_argument("--mu_min", type=float, default=0.25)
    parser.add_argument("--mu_max", type=float, default=4.0)

    # Similarity and regularization
    parser.add_argument("--ncc_win", type=int, default=9)
    parser.add_argument("--lambda_ncc", type=float, default=1.0)
    parser.add_argument("--lambda_inverse", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_physics", type=float, default=1e-4)
    parser.add_argument("--lambda_jacobian", type=float, default=0.01)
    parser.add_argument("--lambda_mu", type=float, default=0.001)
    parser.add_argument("--jacobian_threshold", type=float, default=0.01)
    parser.add_argument("--lame_lambda", type=float, default=1.0)

    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Wavelet scales: {args.wavelet_scales}")

    fixed_path = Path(args.fixed)
    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed image not found: {fixed_path}")

    fixed, fixed_nii, _ = load_nifti(fixed_path, device, is_label=False)

    if Path(args.fixed_label).exists():
        fixed_label, fixed_label_nii, _ = load_nifti(
            args.fixed_label, device, is_label=True
        )
        if tuple(fixed_label.shape[-3:]) != tuple(fixed.shape[-3:]):
            raise ValueError(
                "Fixed image and fixed label have different array shapes. "
                "Resample the label to the fixed image geometry first."
            )
    else:
        print(
            f"Warning: fixed label not found: {args.fixed_label}. "
            "Registration will run without label metrics."
        )
        fixed_label = None

    # Prefer testpath for evaluation if it contains images; otherwise use
    # datapath. This avoids silently processing an empty directory.
    images = []
    if Path(args.testpath).exists():
        images = discover_images(args.testpath, fixed_path)

    if not images:
        print("No eligible images found in testpath; trying datapath.")
        images = discover_images(args.datapath, fixed_path)

    if not images:
        raise RuntimeError(
            "No subject images were discovered. Check --testpath, --datapath, "
            "and the subject filename convention (for example S02*.nii.gz)."
        )

    print(f"Found {len(images)} candidate moving images.")

    all_metrics = []

    for index, (subject_id, moving_path) in enumerate(images, start=1):
        print(
            f"\n[{index}/{len(images)}] "
            f"Registering {subject_id}: {moving_path}"
        )

        moving, moving_nii, _ = load_nifti(
            moving_path, device, is_label=False
        )

        if tuple(moving.shape[-3:]) != tuple(fixed.shape[-3:]):
            print(
                f"Skipping {subject_id}: image shape "
                f"{tuple(moving.shape[-3:])} differs from fixed "
                f"{tuple(fixed.shape[-3:])}. Resample first."
            )
            continue

        moving_label_path = find_label_file(args.label_dir, subject_id)
        if moving_label_path is not None:
            moving_label, moving_label_nii, _ = load_nifti(
                moving_label_path, device, is_label=True
            )

            if tuple(moving_label.shape[-3:]) != tuple(fixed.shape[-3:]):
                print(
                    f"Warning: label geometry mismatch for {subject_id}; "
                    "label metrics will be omitted for this pair."
                )
                moving_label = None
        else:
            moving_label = None

        pair_output = output_dir / subject_id
        pair_output.mkdir(parents=True, exist_ok=True)

        metrics = optimize_pair(
            fixed=fixed,
            moving=moving,
            fixed_nii=fixed_nii,
            moving_nii=moving_nii,
            fixed_label=fixed_label,
            moving_label=moving_label,
            output_dir=pair_output,
            subject_id=subject_id,
            args=args,
        )

        metrics["moving_image"] = str(moving_path)
        metrics["moving_label"] = (
            str(moving_label_path) if moving_label_path is not None else ""
        )
        all_metrics.append(metrics)

        with open(pair_output / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    if all_metrics:
        summary_path = output_dir / "summary.csv"
        fieldnames = list(all_metrics[0].keys())

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

        print(f"\nSaved summary to: {summary_path}")

        dice_values = [
            m["dice"] for m in all_metrics if np.isfinite(m["dice"])
        ]
        hd95_values = [
            m["hd95_mm"] for m in all_metrics
            if np.isfinite(m["hd95_mm"])
        ]
        folding_values = [
            m["folding_fraction"] for m in all_metrics
            if np.isfinite(m["folding_fraction"])
        ]

        if dice_values:
            print(f"Mean Dice: {np.mean(dice_values):.4f}")
        if hd95_values:
            print(f"Mean HD95: {np.mean(hd95_values):.3f} mm")
        if folding_values:
            print(f"Mean folding fraction: {np.mean(folding_values):.5f}")


if __name__ == "__main__":
    main()
