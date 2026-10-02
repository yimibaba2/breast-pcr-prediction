(function () {
  "use strict";

  const params = { stage: 0, hr: 0, her2: 0 };
  const files = { image: null, mask: null };

  const $ = (id) => document.getElementById(id);

  // ============ 上传区 ============
  function setupDropzone(dzId, inputId, statusId, key) {
    const dz = $(dzId);
    const input = $(inputId);
    const status = $(statusId);

    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });

    input.addEventListener("change", () => {
      if (input.files.length) setFile(key, input.files[0], status, dz);
    });

    ["dragover", "dragenter"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); })
    );
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); })
    );
    dz.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files[0];
      if (f) setFile(key, f, status, dz);
    });
  }

  function setFile(key, file, statusEl, dzEl) {
    files[key] = file;
    statusEl.textContent = "✓ " + file.name;
    dzEl.classList.add("filled");
    updateButton();
  }

  function updateButton() {
    const ready = files.image && files.mask;
    $("btn-analyze").disabled = !ready;
  }

  // ============ 参数 segmented ============
  document.querySelectorAll(".segmented").forEach((group) => {
    const key = group.dataset.key;
    group.querySelectorAll(".seg").forEach((btn) => {
      btn.addEventListener("click", () => {
        group.querySelectorAll(".seg").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        params[key] = parseInt(btn.dataset.value, 10);
      });
    });
  });

  // ============ 看板 tab ============
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const name = tab.dataset.tab;
      $("panel-morph").hidden = name !== "morph";
      $("panel-ith").hidden = name !== "ith";
    });
  });

  // ============ 分析 ============
  $("btn-analyze").addEventListener("click", analyze);

  async function analyze() {
    if (!files.image || !files.mask) return;

    $("error-box").hidden = true;
    const loading = $("loading");
    loading.hidden = false;
    const loadingText = $("loading-text");
    loadingText.textContent = "正在上传文件…";
    $("btn-text").textContent = "分析中…";

    const fd = new FormData();
    fd.append("image", files.image);
    fd.append("mask", files.mask);
    fd.append("stage", params.stage);
    fd.append("hr", params.hr);
    fd.append("her2", params.her2);

    let pollFails = 0;
    try {
      // 提交任务（立即返回 task_id，计算在服务器后台进行）
      const resp = await fetch("/api/analyze", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "提交失败，请检查输入文件。");
      }

      // 每 3 秒轮询任务进度，直到完成/失败
      while (true) {
        await new Promise((r) => setTimeout(r, 3000));
        let d;
        try {
          const r2 = await fetch("/api/status/" + data.task_id);
          d = await r2.json();
          if (!r2.ok) throw new Error(d.error || "状态查询失败");
          pollFails = 0;
        } catch (err) {
          pollFails++;
          if (pollFails >= 5) throw new Error("网络不稳定，请重试：" + err.message);
          continue;
        }
        if (d.progress) loadingText.textContent = d.progress;
        if (d.status === "completed") { renderResult(d.result); return; }
        if (d.status === "failed") throw new Error(d.error || "计算失败");
      }
    } catch (err) {
      $("error-box").textContent = "错误：" + (err.message || err);
      $("error-box").hidden = false;
    } finally {
      loading.hidden = true;
      $("btn-text").textContent = "重新分析";
    }
  }

  function renderResult(d) {
    // 概率
    const mshPct = (d.msh_score * 100).toFixed(2) + "%";
    const fusionPct = (d.fusion_prob * 100).toFixed(2) + "%";
    $("msh-value").textContent = mshPct;
    $("fusion-value").textContent = fusionPct;

    setBadge("msh-badge", d.msh_positive, d.msh_positive ? "pCR 阳性" : "pCR 阴性");
    setBadge("fusion-badge", d.fusion_positive, d.fusion_positive ? "pCR 阳性" : "pCR 阴性");

    // 指标
    $("metrics-strip").hidden = false;
    $("m-compactness").textContent = d.compactness.toFixed(5);
    $("m-topo").textContent = d.topo_ith.toFixed(4);

    // 看板
    $("morph-img").src = "data:image/png;base64," + d.morphology_img;
    $("morph-img").hidden = false;
    $("placeholder-morph").hidden = true;

    $("ith-img").src = "data:image/png;base64," + d.ith_img;
    $("ith-img").hidden = false;
    $("placeholder-ith").hidden = true;
  }

  function setBadge(id, positive, text) {
    const el = $(id);
    el.textContent = text;
    el.classList.remove("positive", "negative");
    el.classList.add(positive ? "positive" : "negative");
  }

  setupDropzone("drop-image", "file-image", "status-image", "image");
  setupDropzone("drop-mask", "file-mask", "status-mask", "mask");
})();
