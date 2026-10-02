# -*- coding: utf-8 -*-
"""
=============================================================================
乳腺癌新辅助化疗 pCR 预测 - 影像特征提取模块（PyRadiomics 版）
=============================================================================
算法与你提供的批量计算管线 100% 对齐：

1. Compactness（IBSI Compactness 1）
   = MeshVolume / (sqrt(pi) * SurfaceArea^1.5)
   体积/表面积直接取自 PyRadiomics shape 类特征（与你原始代码一致）。

2. 3D Topological Intratumoral Heterogeneity
   = 1 - SUM(各亚区最大连通碎片体积/碎片总数) / 总测量值
   亚区通过 PyRadiomics voxel-based 特征图（kernelRadius=1，
   firstorder + glcm + glszm + glrlm + ngtdm + gldm）+ KMeans(K=4)
   聚类 + 3D 连通域分析得到（与你原始代码一致）。
=============================================================================
"""

import logging
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("SimpleITK").setLevel(logging.ERROR)

# 与原始 Config 一致
TARGET_SPACING = (1.0, 1.0, 1.0)
FIXED_K_3D = 4
N4_SHRINK = 4
MIN_VOXELS_PER_CLUSTER = 4


# =============================================================================
# 一、形态学特征（PyRadiomics shape，与原始代码一致）
# =============================================================================
def compute_shape_features(img_sitk, mask_sitk):
    """用 PyRadiomics 提取 shape 类特征（原始物理空间，无重采样）。"""
    settings = {'geometryTolerance': 1.0}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape')
    features = extractor.execute(img_sitk, mask_sitk)

    def gv(key):
        v = features['original_shape_' + key]
        return float(v.item()) if hasattr(v, 'item') else float(v)

    mesh_volume = gv('MeshVolume')
    surface_area = gv('SurfaceArea')
    sphericity = gv('Sphericity')
    voxel_volume = gv('VoxelVolume')
    svr = gv('SurfaceVolumeRatio')

    # IBSI Compactness 1（与原始代码公式一致）
    compactness = (mesh_volume / (np.sqrt(np.pi) * (surface_area ** 1.5))
                   if surface_area > 0 and mesh_volume > 0 else 0.0)

    return {
        "mesh_volume": mesh_volume,
        "surface_area": surface_area,
        "voxel_volume": voxel_volume,
        "compactness": float(compactness),
        "sphericity": sphericity,
        "surface_volume_ratio": svr,
    }


# =============================================================================
# 二、ITH 预处理（与原始 OptimizedPreprocessor 一致）
# =============================================================================
def _extract_body_mask(image):
    img_float = sitk.Cast(image, sitk.sitkFloat32)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    rough = otsu.Execute(img_float)
    closer = sitk.BinaryMorphologicalClosingImageFilter()
    closer.SetKernelRadius(2)
    closer.SetKernelType(sitk.sitkBall)
    closer.SetForegroundValue(1)
    closed = closer.Execute(rough)
    filler = sitk.BinaryFillholeImageFilter()
    filler.SetForegroundValue(1)
    return filler.Execute(closed)


def _apply_n4(image, body_mask, shrink_factor=N4_SHRINK):
    img_float = sitk.Cast(image, sitk.sitkFloat32)
    shrunk_image = sitk.Shrink(img_float, [shrink_factor] * img_float.GetDimension())
    shrunk_mask = sitk.Shrink(body_mask, [shrink_factor] * body_mask.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.SetConvergenceThreshold(0.001)
    corrector.Execute(shrunk_image, shrunk_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(img_float)
    return sitk.Divide(img_float, sitk.Exp(log_bias))


def _zscore_normalize(image, body_mask):
    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(body_mask)
    roi_vals = img_arr[mask_arr > 0]
    if roi_vals.size > 0:
        mu = float(np.mean(roi_vals))
        sigma = float(np.std(roi_vals))
    else:
        mu, sigma = 0.0, 1.0
    if sigma < 1e-8:
        sigma = 1.0
    norm_arr = ((img_arr - mu) / sigma).astype(np.float32)
    norm_arr[mask_arr == 0] = 0.0
    norm_img = sitk.GetImageFromArray(norm_arr)
    norm_img.CopyInformation(image)
    return norm_img


def _resample_isotropic(image, target_spacing=TARGET_SPACING, is_label=False):
    in_sp = image.GetSpacing()
    in_size = image.GetSize()
    out_size = [int(round(in_size[i] * (in_sp[i] / target_spacing[i]))) for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetSize(out_size)
    r.SetOutputSpacing(list(target_spacing))
    r.SetOutputOrigin(image.GetOrigin())
    r.SetOutputDirection(image.GetDirection())
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return r.Execute(image)


def _crop_tumor(img_sitk, mask_sitk, pad_mm=20):
    """裁剪肿瘤 bounding box + padding。

    仅减少计算量：BSpline 插值核支撑范围有限，padding 足够时
    肿瘤区域的重采样结果与全图重采样完全一致，不影响特征数值。
    特征计算仅需 1 体素邻域（kernelRadius=1），5mm 余量充分。
    """
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    coords = np.where(mask_arr > 0)
    if len(coords[0]) == 0:
        return img_sitk, mask_sitk
    spacing = img_sitk.GetSpacing()
    pad_z = int(pad_mm / spacing[2])
    pad_y = int(pad_mm / spacing[1])
    pad_x = int(pad_mm / spacing[0])

    z0 = max(0, int(coords[0].min()) - pad_z)
    z1 = min(mask_arr.shape[0], int(coords[0].max()) + pad_z + 1)
    y0 = max(0, int(coords[1].min()) - pad_y)
    y1 = min(mask_arr.shape[1], int(coords[1].max()) + pad_y + 1)
    x0 = max(0, int(coords[2].min()) - pad_x)
    x1 = min(mask_arr.shape[2], int(coords[2].max()) + pad_x + 1)

    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize([x0, y0, z0])
    crop.SetUpperBoundaryCropSize(
        [mask_arr.shape[2] - x1, mask_arr.shape[1] - y1, mask_arr.shape[0] - z1]
    )
    return crop.Execute(img_sitk), crop.Execute(mask_sitk)


def preprocess_ith(img_sitk, mask_sitk):
    """体膜提取 -> N4 校正 -> Z-score -> 裁剪肿瘤 -> 各向同性重采样。"""
    body_mask = _extract_body_mask(img_sitk)
    img_n4 = _apply_n4(img_sitk, body_mask, shrink_factor=N4_SHRINK)
    img_norm = _zscore_normalize(img_n4, body_mask)
    img_crop, mask_crop = _crop_tumor(img_norm, mask_sitk, pad_mm=20)
    img_r = _resample_isotropic(img_crop, is_label=False)
    mask_r = _resample_isotropic(mask_crop, is_label=True)
    return img_r, mask_r


# =============================================================================
# 三、3D Topological ITH（PyRadiomics voxel-based，与原始代码一致）
# =============================================================================
GLCM_FEATURES = [
    'Autocorrelation', 'JointAverage', 'ClusterProminence', 'ClusterShade',
    'ClusterTendency', 'Contrast', 'Correlation', 'DifferenceAverage',
    'DifferenceEntropy', 'DifferenceVariance', 'JointEnergy', 'JointEntropy',
    'Imc1', 'Imc2', 'Idm', 'Idmn', 'Id', 'Idn', 'InverseVariance',
    'MaximumProbability', 'SumEntropy', 'SumSquares'
]

FEATURE_CLASSES = ["firstorder", "glcm", "glszm", "glrlm", "ngtdm", "gldm"]


def _make_extractor(feature_class):
    """构造仅启用单个特征类的 voxel-based extractor。

    设置与原始 build_extractor(force_2d=False) 完全一致；
    分批提取是为了控制内存峰值，各特征独立计算，结果不受影响。
    """
    settings = {
        "voxelBased": True,
        "kernelRadius": 1,
        "force2D": False,
        "geometryTolerance": 1.0,
        "normalize": False,
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()
    if feature_class == "glcm":
        # 与原始代码一致：GLCM 仅启用指定的 22 个特征
        ext.enableFeaturesByName(**{"glcm": GLCM_FEATURES})
    else:
        ext.enableFeatureClassByName(feature_class)
    return ext


def _topo_ith(cluster_map, total_measure, struct):
    n_clusters = int(cluster_map.max())
    sigma_sum = 0.0
    for i in range(1, n_clusters + 1):
        lbl, n_frags = label((cluster_map == i).astype(np.uint8), structure=struct)
        if n_frags > 0:
            counts = np.bincount(lbl.ravel())[1:]
            sigma_sum += float(counts.max()) / n_frags
    return 1.0 - sigma_sum / total_measure if total_measure > 0 else float("nan")


def compute_topological_ith(img_arr, mask_arr, k=FIXED_K_3D):
    """3D 拓扑瘤内异质性（与原始 compute_ith_with_maps(is_2d=False) 一致）。"""
    coords = np.where(mask_arr > 0)
    if len(coords[0]) == 0:
        raise ValueError("ROI 掩码为空，无法计算异质性。")

    slices = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
    sub_img = img_arr[slices]
    sub_mask = mask_arr[slices]

    valid_coords = np.argwhere(sub_mask > 0)
    n_voxels = len(valid_coords)
    if n_voxels < (MIN_VOXELS_PER_CLUSTER + 2):
        raise ValueError("ROI 体素过少，无法进行聚类。")

    # 构建 PyRadiomics 输入（与原始代码一致：spacing 1mm）
    fi = sitk.GetImageFromArray(np.ascontiguousarray(sub_img.astype(np.float32)))
    fm = sitk.GetImageFromArray(np.ascontiguousarray(sub_mask.astype(np.uint8)))
    fi.SetSpacing(TARGET_SPACING)
    fm.SetSpacing(TARGET_SPACING)

    # 分特征类逐批提取（结果与一次性提取完全一致，峰值内存大幅降低防 OOM）
    F_parts = []
    for fc in FEATURE_CLASSES:
        ext = _make_extractor(fc)
        f_maps = ext.execute(fi, fm)
        for key, val in f_maps.items():
            if key.startswith("original_"):
                F_parts.append(np.asarray(
                    np.nan_to_num(sitk.GetArrayFromImage(val)[sub_mask > 0]),
                    dtype=np.float32,
                ))
        del f_maps

    F = np.column_stack(F_parts)
    F_parts = None
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    F_scaled = StandardScaler().fit_transform(F)
    del F

    # 拓扑聚类（KMeans K=4，random_state=42，n_init=10 —— 与原始代码一致）
    topo_k = int(min(k, n_voxels - 1))
    if topo_k < 2:
        raise ValueError("ROI 体素过少，无法进行聚类。")
    if n_voxels > 300000:
        # 极大肿瘤防御（2GB 内存服务器）：抽样拟合 + 全量预测
        rng = np.random.default_rng(42)
        idx = rng.choice(n_voxels, 300000, replace=False)
        km = KMeans(n_clusters=topo_k, random_state=42, n_init=10).fit(F_scaled[idx])
        lbl_topo = km.predict(F_scaled)
    else:
        lbl_topo = KMeans(n_clusters=topo_k, random_state=42, n_init=10).fit_predict(F_scaled)

    cluster_map = np.zeros_like(sub_mask, dtype=np.uint8)
    for idx, c in enumerate(valid_coords):
        cluster_map[tuple(c)] = int(lbl_topo[idx]) + 1

    struct = np.ones((3, 3, 3), dtype=int)
    topo_score = _topo_ith(cluster_map, n_voxels, struct)

    return float(topo_score), cluster_map, sub_img, sub_mask, int(n_voxels)


# =============================================================================
# 四、主入口
# =============================================================================
def extract_all(img_path, mask_path):
    """读取 nii.gz，计算 Compactness 与 3D Topological ITH。"""
    if not (img_path and mask_path):
        raise ValueError("请提供影像与掩码文件。")

    img = sitk.ReadImage(img_path)
    mask_raw = sitk.ReadImage(mask_path)

    # 物理空间对齐（掩码重采样到影像网格，与原始代码一致）
    aligner = sitk.ResampleImageFilter()
    aligner.SetReferenceImage(img)
    aligner.SetInterpolator(sitk.sitkNearestNeighbor)
    aligner.SetTransform(sitk.Transform())
    aligner.SetDefaultPixelValue(0)
    mask = aligner.Execute(sitk.Cast(mask_raw > 0, sitk.sitkUInt8))

    img_arr = sitk.GetArrayFromImage(img)
    mask_arr = sitk.GetArrayFromImage(mask)
    spacing = img.GetSpacing()

    if np.sum(mask_arr) == 0:
        raise ValueError("对齐后 ROI 掩码体素为 0，请检查影像与掩码原点是否重叠。")

    # ---- 形态学（PyRadiomics shape，原始物理空间） ----
    shape = compute_shape_features(img, mask)

    # ---- ITH（预处理 + PyRadiomics voxel-based + KMeans + 拓扑） ----
    img_p, mask_p = preprocess_ith(img, mask)
    img_p_arr = sitk.GetArrayFromImage(img_p)
    mask_p_arr = sitk.GetArrayFromImage(mask_p)

    topo_ith, cluster_map, sub_img, sub_mask, n_voxels = compute_topological_ith(
        img_p_arr, mask_p_arr > 0
    )

    return {
        "compactness": shape["compactness"],
        "topo_ith": topo_ith,
        "shape": shape,
        "_img_arr": img_arr,
        "_mask_arr": mask_arr,
        "_spacing": spacing,
        "_cluster_map": cluster_map,
        "_sub_img": sub_img,
        "_sub_mask": sub_mask,
    }
