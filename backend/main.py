# -*- coding: utf-8 -*-
"""
=============================================================================
乳腺癌新辅助化疗 pCR 预测 Web 应用 - 后端入口（异步任务版 v2）
- 分析请求立即返回任务 ID，后台线程计算，前端轮询 /api/status 获取进度
- 彻底避免长连接空闲被切断导致的 Failed to fetch
=============================================================================
"""

import os
import base64
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from features import extract_all
from visual import render_morphology_report, render_ith_report

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(
    title="乳腺癌 pCR 预测",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# MSH 模型截断值 / 融合模型截断值
MSH_THRESHOLD = 0.442
FUSION_THRESHOLD = 0.391

# 内存任务表：task_id -> {status, progress, result, error}
TASKS = {}
TASKS_MAX = 30


# =============================================================================
# 概率计算
# =============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def calc_msh(compactness, topo_ith):
    """MSH 模型（仅影像特征）。"""
    z = -1.0096 + 61.8292 * compactness - 2.4438 * topo_ith
    return float(sigmoid(z)), float(z)


def calc_fusion(stage, hr, her2, compactness, topo_ith):
    """融合模型（影像 + 临床病理）。"""
    z = (-1.002260 + 0.002288 * stage - 1.037019 * hr + 1.362167 * her2
         - 2.495751 * topo_ith + 63.488501 * compactness)
    return float(sigmoid(z)), float(z)


# =============================================================================
# 后台任务
# =============================================================================
def _cleanup_tasks():
    """简单容量控制：只保留最近 TASKS_MAX 个任务。"""
    if len(TASKS) <= TASKS_MAX:
        return
    for k in list(TASKS.keys())[: len(TASKS) - TASKS_MAX]:
        TASKS.pop(k, None)


def _run_task(task_id, img_path, mask_path, stage, hr, her2, tmp_dir):
    """在后台线程中执行完整分析流程。"""
    try:
        TASKS[task_id]["progress"] = "N4 校正与预处理中…（整个流程约需 2~5 分钟，请耐心等待）"
        res = extract_all(img_path, mask_path)
        compactness = res["compactness"]
        topo_ith = res["topo_ith"]

        TASKS[task_id]["progress"] = "聚类完成，正在计算 pCR 预测概率…"
        msh_score, _ = calc_msh(compactness, topo_ith)
        fusion_prob, _ = calc_fusion(stage, hr, her2, compactness, topo_ith)

        TASKS[task_id]["progress"] = "正在生成可视化看板…"
        morph_png = render_morphology_report(
            res["_img_arr"], res["_mask_arr"], res["_spacing"], res["shape"]
        )
        ith_png = render_ith_report(
            res["_cluster_map"], res["_sub_img"], topo_ith,
            int(np.sum(res["_sub_mask"] > 0))
        )

        TASKS[task_id].clear()
        TASKS[task_id].update(
            status="completed",
            result={
                "compactness": compactness,
                "topo_ith": topo_ith,
                "shape": res["shape"],
                "msh_score": msh_score,
                "msh_threshold": MSH_THRESHOLD,
                "msh_positive": bool(msh_score > MSH_THRESHOLD),
                "fusion_prob": fusion_prob,
                "fusion_threshold": FUSION_THRESHOLD,
                "fusion_positive": bool(fusion_prob > FUSION_THRESHOLD),
                "morphology_img": base64.b64encode(morph_png).decode("ascii"),
                "ith_img": base64.b64encode(ith_png).decode("ascii"),
            },
        )
    except Exception as e:
        traceback.print_exc()
        TASKS[task_id].clear()
        TASKS[task_id].update(status="failed", error=str(e))
    finally:
        for p in (img_path, mask_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass
        _cleanup_tasks()


# =============================================================================
# API
# =============================================================================
@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    stage: int = Form(...),
    hr: int = Form(...),
    her2: int = Form(...),
):
    """接收文件与参数，立即返回任务 ID；计算在后台线程执行。"""
    tmp_dir = tempfile.mkdtemp(prefix="pcr_")
    img_path = os.path.join(tmp_dir, "image.nii.gz")
    mask_path = os.path.join(tmp_dir, "mask.nii.gz")

    try:
        img_bytes = await image.read()
        mask_bytes = await mask.read()
        if not img_bytes or not mask_bytes:
            return JSONResponse(status_code=400, content={"error": "请上传影像与掩码文件。"})
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"文件保存失败：{e}"})

    task_id = uuid.uuid4().hex
    TASKS[task_id] = {"status": "running", "progress": "文件上传完成，开始预处理…"}
    threading.Thread(
        target=_run_task,
        args=(task_id, img_path, mask_path, stage, hr, her2, tmp_dir),
        daemon=True,
    ).start()
    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
async def task_status(task_id: str):
    """前端轮询：返回任务状态 / 进度 / 结果。"""
    t = TASKS.get(task_id)
    if t is None:
        return JSONResponse(
            status_code=404,
            content={"error": "任务不存在（服务可能已重启），请重新分析。"},
        )
    if t.get("status") == "completed":
        return {"status": "completed", "result": t.get("result")}
    return {"status": t.get("status"), "progress": t.get("progress", "")}


# =============================================================================
# 静态前端
# =============================================================================
@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import os
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
