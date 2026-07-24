(() => {
  const DEBOUNCE_MS = 300;

  // Full column order (primary first, detail trailing). JSON may alphabetize keys.
  const COLUMN_ORDER = {
    memory_footprint: [
      "Model",
      "GPU",
      "Fits",
      "Footprint",
      "Weights",
      "Free for KV",
      "KV / Request",
      "KV @ Max Ctx",
      "Arch",
      "Total VRAM",
    ],
    concurrent_capacity: [
      "Model",
      "GPU",
      "Concurrent @ Workload",
      "Concurrent @ Max Ctx",
      "Free for KV",
      "Max Context Window",
      "Max KV Tokens",
    ],
    performance: [
      "Model",
      "GPU",
      "Fits",
      "TTFT",
      "TPOT",
      "E2E",
      "Throughput",
      "Prefill / Token",
      "TPOT @ batch 1",
      "Max KV Tokens",
    ],
  };

  const DETAIL_COLS = {
    memory_footprint: ["Arch", "Total VRAM"],
    concurrent_capacity: ["Free for KV", "Max Context Window", "Max KV Tokens"],
    performance: ["Prefill / Token", "TPOT @ batch 1", "Max KV Tokens"],
  };

  const COLUMN_HELP = {
    Model: "Model under evaluation.",
    GPU: "GPU type used for this row.",
    Arch: "Attention / KV architecture family.",
    Fits: "Whether footprint fits in usable VRAM at this workload.",
    Footprint: "weights + overhead + (concurrent × KV per request).",
    Weights: "params (B) × bytes per weight — total params, not active.",
    "KV / Request": "KV cache (+ fixed state) for one request at input+output.",
    "Free for KV": "usable VRAM − overhead − weights.",
    "KV @ Max Ctx": "KV per request sized at the model max context window.",
    "Total VRAM": "num GPUs × raw GPU memory (not utilization-scaled).",
    "Concurrent @ Workload": "floor(free for KV / KV per request) at input+output.",
    "Concurrent @ Max Ctx": "Same capacity math at the model max context.",
    "Max Context Window": "Model-spec maximum context length.",
    "Max KV Tokens": "Token budget from free-for-KV ÷ KV bytes per token.",
    TTFT: "Time to first token: (input × prefill + TPOT) / 1000 seconds.",
    TPOT: "Time per output token at configured concurrency (ms).",
    E2E: "(input × prefill + output × TPOT) / 1000 seconds.",
    Throughput: "concurrent × output tokens / E2E (aggregate tok/s).",
    "Prefill / Token": "Compute-bound prefill ms per token, scaled by batch.",
    "TPOT @ batch 1": "Decode TPOT with no concurrency scaling — best-case.",
  };

  const els = {
    form: document.getElementById("config-form"),
    error: document.getElementById("error"),
    warningsBlock: document.getElementById("warnings-block"),
    warningsCount: document.getElementById("warnings-count"),
    warnings: document.getElementById("warnings"),
    gpuList: document.getElementById("gpu-list"),
    modelList: document.getElementById("model-list"),
    gpuCount: document.getElementById("gpu-count"),
    modelCount: document.getElementById("model-count"),
    quant: document.getElementById("quant"),
    kvQuant: document.getElementById("kv_quant"),
    calculateBtn: document.getElementById("calculate-btn"),
    methodology: document.getElementById("methodology"),
    methodologyOpen: document.getElementById("methodology-open"),
    methodologyClose: document.getElementById("methodology-close"),
    tables: {
      memory_footprint: document.getElementById("table-memory"),
      concurrent_capacity: document.getElementById("table-concurrent"),
      performance: document.getElementById("table-performance"),
    },
  };

  const detailOpen = {
    memory_footprint: false,
    concurrent_capacity: false,
    performance: false,
  };

  let debounceTimer = null;
  let requestId = 0;
  let lastRows = {
    memory_footprint: [],
    concurrent_capacity: [],
    performance: [],
  };

  function tableToJSON(table) {
    const headers = [];
    table.querySelectorAll("thead th").forEach((cell) => {
      const btn = cell.querySelector(".th-help");
      headers.push((btn || cell).textContent.replace(/\?$/, "").trim());
    });

    const data = [];
    table.querySelectorAll("tbody tr").forEach((row) => {
      const cells = row.querySelectorAll("td");
      if (!cells.length) return;
      const rowData = {};
      cells.forEach((cell, index) => {
        let val = cell.textContent.trim();
        if (val === "YES") val = true;
        else if (val === "NO") val = false;
        else if (val !== "" && !isNaN(val) && !/[a-zA-Z]/.test(val)) val = Number(val);
        rowData[headers[index] || `column_${index}`] = val;
      });
      data.push(rowData);
    });
    return data;
  }

  function setupTableTools() {
    document.querySelectorAll(".table-block").forEach((block) => {
      const key = block.dataset.table;
      const tools = block.querySelector(".table-tools");
      if (!tools || !key) return;
      tools.innerHTML = "";

      const detailsBtn = document.createElement("button");
      detailsBtn.type = "button";
      detailsBtn.className = "tool";
      detailsBtn.textContent = "Details";
      detailsBtn.setAttribute("aria-pressed", "false");
      detailsBtn.title = "Show secondary columns";
      detailsBtn.addEventListener("click", () => {
        detailOpen[key] = !detailOpen[key];
        detailsBtn.setAttribute("aria-pressed", String(detailOpen[key]));
        renderTable(els.tables[key], lastRows[key], key);
      });

      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "tool";
      copyBtn.textContent = "Copy JSON";
      copyBtn.title = "Copy visible columns as JSON";
      copyBtn.addEventListener("click", () => {
        const table = block.querySelector("table");
        if (!table) {
          alert("No calculation data available in this table yet.");
          return;
        }
        navigator.clipboard
          .writeText(JSON.stringify(tableToJSON(table), null, 2))
          .then(() => {
            const original = copyBtn.textContent;
            copyBtn.textContent = "Copied!";
            copyBtn.classList.add("copied");
            setTimeout(() => {
              copyBtn.textContent = original;
              copyBtn.classList.remove("copied");
            }, 2000);
          })
          .catch(() => alert("Failed to copy JSON. Please try again."));
      });

      tools.append(detailsBtn, copyBtn);
    });
  }

  function setupMethodology() {
    const dialog = els.methodology;
    if (!dialog) return;

    els.methodologyOpen?.addEventListener("click", () => dialog.showModal());
    els.methodologyClose?.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (e) => {
      if (e.target === dialog) dialog.close();
    });
  }

  function fillSelect(select, options, selected) {
    select.innerHTML = "";
    for (const opt of options) {
      const el = document.createElement("option");
      el.value = opt;
      el.textContent = opt;
      if (opt === selected) el.selected = true;
      select.appendChild(el);
    }
  }

  function fillGpuChecklist(container, details, selectedNames) {
    const selected = new Set(selectedNames);
    container.innerHTML = "";
    for (const gpu of details) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = "gpu_names";
      input.value = gpu.name;
      input.checked = selected.has(gpu.name);

      const body = document.createElement("span");
      body.className = "item-body";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = gpu.name;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = `${gpu.memory_gb} GB`;
      body.append(name, meta);

      label.append(input, body);
      container.appendChild(label);
    }
  }

  function fillModelChecklist(container, details, selectedNames) {
    const selected = new Set(selectedNames);
    container.innerHTML = "";
    for (const model of details) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = "model_names";
      input.value = model.name;
      input.checked = selected.has(model.name);

      const body = document.createElement("span");
      body.className = "item-body";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = model.name;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = `${model.params_billion}B · ${model.max_context_window.toLocaleString()} ctx`;
      body.append(name, meta);

      const badge = document.createElement("span");
      badge.className = `badge arch-${model.architecture}`;
      badge.textContent = model.arch_label;

      label.append(input, body, badge);
      container.appendChild(label);
    }
  }

  function updateSelectionCounts() {
    const gpus = checkedValues("gpu_names").length;
    const models = checkedValues("model_names").length;
    const gpuTotal = els.form.querySelectorAll('input[name="gpu_names"]').length;
    const modelTotal = els.form.querySelectorAll('input[name="model_names"]').length;
    els.gpuCount.textContent = `${gpus}/${gpuTotal}`;
    els.modelCount.textContent = `${models}/${modelTotal}`;
  }

  function checkedValues(name) {
    return Array.from(
      els.form.querySelectorAll(`input[name="${name}"]:checked`)
    ).map((el) => el.value);
  }

  function readConfig() {
    return {
      num_gpu: Number(els.form.num_gpu.value),
      prompt_sz: Number(els.form.prompt_sz.value),
      response_sz: Number(els.form.response_sz.value),
      n_concurrent_req: Number(els.form.n_concurrent_req.value),
      system_overhead_gb: Number(els.form.system_overhead_gb.value),
      vram_util: Number(els.form.vram_util.value),
      kv_frag: Number(els.form.kv_frag.value),
      quant: els.form.quant.value,
      kv_quant: els.form.kv_quant.value,
      gpu_names: checkedValues("gpu_names"),
      model_names: checkedValues("model_names"),
    };
  }

  function applyDefaults(defaults) {
    for (const key of [
      "num_gpu",
      "prompt_sz",
      "response_sz",
      "n_concurrent_req",
      "system_overhead_gb",
      "vram_util",
      "kv_frag",
    ]) {
      els.form[key].value = defaults[key];
    }
  }

  function cellClass(key, value) {
    if (key === "Fits") {
      if (value === "YES") return "fits-yes";
      if (value === "NO") return "fits-no";
    }
    if (typeof value === "string" && value.includes("INFEASIBLE")) {
      return "infeasible";
    }
    return "";
  }

  function visibleKeys(row, tableKey) {
    const order = COLUMN_ORDER[tableKey] || Object.keys(row);
    const ordered = order.filter((k) => Object.prototype.hasOwnProperty.call(row, k));
    if (detailOpen[tableKey]) return ordered;
    const detail = new Set(DETAIL_COLS[tableKey] || []);
    return ordered.filter((k) => !detail.has(k));
  }

  function renderTable(container, rows, tableKey) {
    container.innerHTML = "";
    if (!rows || !rows.length) {
      container.textContent = "No rows for this selection.";
      return;
    }
    const keys = visibleKeys(rows[0], tableKey);
    const detail = new Set(DETAIL_COLS[tableKey] || []);
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");

    for (const key of keys) {
      const th = document.createElement("th");
      if (key === "Model") th.classList.add("col-model");
      if (detail.has(key)) th.classList.add("detail-col");

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "th-help";
      btn.textContent = key;
      btn.title = COLUMN_HELP[key] || key;
      btn.setAttribute("aria-label", `${key}: ${COLUMN_HELP[key] || key}`);
      th.appendChild(btn);
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");
      const fitsNo = row.Fits === "NO";
      if (fitsNo) tr.classList.add("row-infeasible");

      for (const key of keys) {
        const td = document.createElement("td");
        const value = row[key];
        td.textContent = value == null ? "" : String(value);
        const classes = [];
        if (key === "Model") classes.push("col-model");
        if (detail.has(key)) classes.push("detail-col");
        const cls = cellClass(key, value);
        if (cls) classes.push(cls);
        if (classes.length) td.className = classes.join(" ");
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
  }

  function setWarnings(warnings) {
    if (warnings && warnings.length) {
      els.warningsBlock.hidden = false;
      els.warningsCount.textContent =
        warnings.length === 1
          ? "1 warning"
          : `${warnings.length} warnings`;
      els.warnings.textContent = warnings.join("\n\n");
    } else {
      els.warningsBlock.hidden = true;
      els.warningsCount.textContent = "";
      els.warnings.textContent = "";
    }
  }

  async function calculate() {
    const id = ++requestId;
    const body = readConfig();
    updateSelectionCounts();

    if (!body.gpu_names.length || !body.model_names.length) {
      els.error.hidden = false;
      els.error.textContent = "Select at least one GPU and one model.";
      return;
    }

    els.calculateBtn.disabled = true;
    els.error.hidden = true;

    try {
      const res = await fetch("/api/calculate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (id !== requestId) return;

      if (!res.ok) {
        els.error.hidden = false;
        els.error.textContent = data.error || `Request failed (${res.status})`;
        return;
      }

      lastRows.memory_footprint = data.memory_footprint || [];
      lastRows.concurrent_capacity = data.concurrent_capacity || [];
      lastRows.performance = data.performance || [];

      renderTable(els.tables.memory_footprint, lastRows.memory_footprint, "memory_footprint");
      renderTable(
        els.tables.concurrent_capacity,
        lastRows.concurrent_capacity,
        "concurrent_capacity"
      );
      renderTable(els.tables.performance, lastRows.performance, "performance");
      setWarnings(data.warnings);
    } catch (err) {
      if (id !== requestId) return;
      els.error.hidden = false;
      els.error.textContent = String(err);
    } finally {
      if (id === requestId) els.calculateBtn.disabled = false;
    }
  }

  function scheduleCalculate() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(calculate, DEBOUNCE_MS);
  }

  function toggleGroup(kind) {
    const name = kind === "gpus" ? "gpu_names" : "model_names";
    const boxes = Array.from(
      els.form.querySelectorAll(`input[name="${name}"]`)
    );
    const allOn = boxes.every((b) => b.checked);
    boxes.forEach((b) => {
      b.checked = !allOn;
    });
    updateSelectionCounts();
    scheduleCalculate();
  }

  async function init() {
    setupTableTools();
    setupMethodology();

    const res = await fetch("/api/options");
    if (!res.ok) throw new Error(`Failed to load options (${res.status})`);
    const opts = await res.json();

    applyDefaults(opts.defaults);
    fillSelect(els.quant, opts.weight_quants, opts.defaults.quant);
    fillSelect(els.kvQuant, opts.kv_quants, opts.defaults.kv_quant);
    fillGpuChecklist(els.gpuList, opts.gpu_details, opts.defaults.gpu_names);
    fillModelChecklist(els.modelList, opts.model_details, opts.defaults.model_names);
    updateSelectionCounts();

    // ponytail: URL override. Keys match input names; gpus/models = comma lists.
    // Ceiling: exact name match only. Upgrade: typed schema / shareable config URL.
    const urlParams = new URLSearchParams(window.location.search);
    for (const [key, val] of urlParams) {
      if (els.form[key]) els.form[key].value = val;
    }
    ["gpus", "models"].forEach((key) => {
      if (!urlParams.has(key)) return;
      const wanted = new Set(urlParams.get(key).split(","));
      const nameAttr = key === "gpus" ? "gpu_names" : "model_names";
      els.form.querySelectorAll(`input[name="${nameAttr}"]`).forEach((cb) => {
        cb.checked = wanted.has(cb.value);
      });
    });
    updateSelectionCounts();

    els.form.addEventListener("input", () => {
      updateSelectionCounts();
      scheduleCalculate();
    });
    els.form.addEventListener("change", () => {
      updateSelectionCounts();
      scheduleCalculate();
    });
    els.form.addEventListener("submit", (e) => {
      e.preventDefault();
      clearTimeout(debounceTimer);
      calculate();
    });

    document.querySelectorAll("[data-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => toggleGroup(btn.dataset.toggle));
    });

    await calculate();
  }

  init().catch((err) => {
    els.error.hidden = false;
    els.error.textContent = String(err);
  });
})();
