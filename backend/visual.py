# -*- coding: utf-8 -*-
"""
可视化看板渲染模块：生成形态学看板与生境异质性看板两张 PNG。
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import binary_erosion, binary_dilation

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

CLUSTER_COLORS = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6", "#1ABC9C"]
MESH_FACE = (0.85, 0.35, 0.30, 0.9)
MESH_EDGE = (0.35, 0.10, 0.08, 0.25)


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# 形态学看板
# =============================================================================
def _render_3d_mesh(ax, mask_arr, spacing):
    verts, faces, _, _ = measure.marching_cubes(
        mask_arr.astype(np.float32), level=0.5,
        spacing=(spacing[2], spacing[1], spacing[0]),
    )
    mesh = Poly3DCollection(verts[faces], alpha=0.85)
    mesh.set_facecolor(MESH_FACE)
    mesh.set_edgecolor(MESH_EDGE)
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)

    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_box_aspect((
        verts[:, 0].max() - verts[:, 0].min(),
        verts[:, 1].max() - verts[:, 1].min(),
        verts[:, 2].max() - verts[:, 2].min(),
    ))
    ax.set_xlabel("Z (mm)", fontsize=8, labelpad=2)
    ax.set_ylabel("Y (mm)", fontsize=8, labelpad=2)
    ax.set_zlabel("X (mm)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_title("3D Tumor Morphology", fontsize=11, pad=0)
    ax.view_init(elev=22, azim=-60)


def _render_2d_slice(ax, img_slice, mask_slice, aspect, title, origin="upper"):
    ax.imshow(img_slice, cmap="gray", aspect=aspect, origin=origin)
    if np.max(mask_slice) > 0:
        eroded = binary_erosion(mask_slice > 0)
        boundary = (mask_slice > 0) & (~eroded)
        thick = binary_dilation(boundary, iterations=1)
        h, w = mask_slice.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[mask_slice > 0] = [1.0, 0.2, 0.2, 0.20]
        rgba[thick] = [1.0, 0.1, 0.1, 1.0]
        ax.imshow(rgba, aspect=aspect, origin=origin)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_morphology_report(img_arr, mask_arr, spacing, shape):
    fig = plt.figure(figsize=(14, 8))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
    left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                            height_ratios=[1.25, 0.9], hspace=0.22)

    ax3d = fig.add_subplot(left[0], projection="3d")
    _render_3d_mesh(ax3d, mask_arr, spacing)

    cz = int(np.argmax(np.sum(mask_arr, axis=(1, 2))))
    cy = int(np.argmax(np.sum(mask_arr, axis=(0, 2))))
    cx = int(np.argmax(np.sum(mask_arr, axis=(0, 1))))

    bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=left[1], wspace=0.1)
    _render_2d_slice(fig.add_subplot(bottom[0]), img_arr[cz], mask_arr[cz],
                     spacing[1] / spacing[0], f"Axial (Z={cz})", "upper")
    _render_2d_slice(fig.add_subplot(bottom[1]), img_arr[:, cy, :], mask_arr[:, cy, :],
                     spacing[2] / spacing[0], f"Coronal (Y={cy})", "lower")
    _render_2d_slice(fig.add_subplot(bottom[2]), img_arr[:, :, cx], mask_arr[:, :, cx],
                     spacing[2] / spacing[1], f"Sagittal (X={cx})", "lower")

    ax_table = fig.add_subplot(outer[1])
    ax_table.axis("off")
    rows = [
        ["Mesh Volume (mm\u00b3)", f"{shape['mesh_volume']:.1f}"],
        ["Voxel Volume (mm\u00b3)", f"{shape['voxel_volume']:.1f}"],
        ["Surface Area (mm\u00b2)", f"{shape['surface_area']:.1f}"],
        ["Surface / Volume (1/mm)", f"{shape['surface_volume_ratio']:.4f}"],
        ["Sphericity", f"{shape['sphericity']:.4f}"],
        ["Compactness (IBSI-1)", f"{shape['compactness']:.5f}"],
    ]
    table = ax_table.table(cellText=rows, colWidths=[0.55, 0.45], loc="upper center",
                           cellLoc="left", bbox=[0.0, 0.3, 1.0, 0.5])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#1D3557")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F8F9FA" if r % 2 == 0 else "white")
        cell.set_fontsize(10)
        if c == 1:
            cell.get_text().set_ha("right")
    ax_table.set_title("Morphological Parameters", fontsize=13, fontweight="bold", pad=12)

    fig.suptitle("Morphology Dashboard", fontsize=15, fontweight="bold", y=0.98)
    return _fig_to_png(fig)


# =============================================================================
# 生境异质性看板
# =============================================================================
def _render_3d_clusters(ax, cluster_map):
    k = int(np.max(cluster_map))
    for c_id in range(1, k + 1):
        pts = np.argwhere(cluster_map == c_id)
        if len(pts) == 0:
            continue
        step = max(1, len(pts) // 1200)
        pts = pts[::step]
        ax.scatter(pts[:, 2], pts[:, 1], pts[:, 0],
                   c=CLUSTER_COLORS[(c_id - 1) % len(CLUSTER_COLORS)],
                   s=8, alpha=0.65, edgecolors="none", label=f"Sub-{c_id}")
    ax.set_title(f"3D Tumor Subregions (K = {k})", fontsize=11, fontweight="bold", pad=0)
    ax.set_xlabel("X (mm)", fontsize=8, labelpad=2)
    ax.set_ylabel("Y (mm)", fontsize=8, labelpad=2)
    ax.set_zlabel("Z (mm)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    ax.view_init(elev=22, azim=-60)


def _render_2d_overlay(ax, sub_img, cluster_map):
    ax.imshow(sub_img, cmap="gray", origin="upper")
    k = int(np.max(cluster_map))
    h, w = cluster_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for c_id in range(1, k + 1):
        rgb = matplotlib.colors.to_rgb(CLUSTER_COLORS[(c_id - 1) % len(CLUSTER_COLORS)])
        rgba[cluster_map == c_id] = [rgb[0], rgb[1], rgb[2], 0.55]
    ax.imshow(rgba, origin="upper")
    ax.set_title(f"2D In-Situ Subregion Overlay (K = {k})", fontsize=10, fontweight="bold")
    ax.axis("off")


def _render_proportions(ax, cluster_map):
    k = int(np.max(cluster_map))
    counts = [int(np.sum(cluster_map == c_id)) for c_id in range(1, k + 1)]
    total = sum(counts)
    props = [c / total * 100 for c in counts]
    labels = [f"Sub-{i}" for i in range(1, k + 1)]
    colors = [CLUSTER_COLORS[(i - 1) % len(CLUSTER_COLORS)] for i in range(1, k + 1)]
    bars = ax.bar(labels, props, color=colors, alpha=0.85, edgecolor="#2C3E50", linewidth=1.0)
    for bar, prop in zip(bars, props):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{prop:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, max(props) + 12)
    ax.set_ylabel("Volume Proportion (%)", fontsize=9)
    ax.set_title("3D Subregion Fractional Abundance", fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=8)


def render_ith_report(cluster_map, sub_img, topo_ith, n_voxels):
    fig = plt.figure(figsize=(14, 8))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.14)
    left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                            height_ratios=[1.15, 1.0], hspace=0.22)

    ax_3d = fig.add_subplot(left[0], projection="3d")
    _render_3d_clusters(ax_3d, cluster_map)

    # 2D 覆盖：取聚类图 z 轴体素最多的切片
    cz = int(np.argmax(np.sum(cluster_map, axis=(1, 2))))
    ax_2d = fig.add_subplot(left[1])
    _render_2d_overlay(ax_2d, sub_img[cz], cluster_map[cz])

    right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                             height_ratios=[0.85, 1.25], hspace=0.24)
    ax_bar = fig.add_subplot(right[0])
    _render_proportions(ax_bar, cluster_map)

    ax_table = fig.add_subplot(right[1])
    ax_table.axis("off")
    k = int(np.max(cluster_map))
    rows = [
        ["3D Topological ITH", f"{topo_ith:.4f}"],
        ["Cluster Count (K)", f"{k}"],
        ["Total Voxels", f"{n_voxels}"],
    ]
    table = ax_table.table(cellText=rows, colWidths=[0.55, 0.45], loc="upper center",
                           cellLoc="left", bbox=[0.0, 0.4, 1.0, 0.4])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#1D3557")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F8F9FA" if r % 2 == 0 else "white")
        cell.set_fontsize(10)
        if c == 1:
            cell.get_text().set_ha("right")
    ax_table.set_title("ITH Metrics", fontsize=12, fontweight="bold", pad=12)

    fig.suptitle("Intratumoral Heterogeneity (ITH) Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    return _fig_to_png(fig)
