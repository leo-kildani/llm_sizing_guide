(() => {
  const DEBOUNCE_MS = 300;

  const els = {
    form: document.getElementById("config-form"),
    error: document.getElementById("error"),
    warnings: document.getElementById("warnings"),
    gpuList: document.getElementById("gpu-list"),
    modelList: document.getElementById("model-list"),
    quant: document.getElementById("quant"),
    kvQuant: document.getElementById("kv_quant"),
    calculateBtn: document.getElementById("calculate-btn"),
    tables: {
      memory_footprint: document.getElementById("table-memory"),
      concurrent_capacity: document.getElementById("table-concurrent"),
      performance: document.getElementById("table-performance"),
    },
  };

  let debounceTimer = null;
  let requestId = 0;

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

  function fillChecklist(container, names, selectedNames, nameAttr) {
    const selected = new Set(selectedNames);
    container.innerHTML = "";
    for (const name of names) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = nameAttr;
      input.value = name;
      input.checked = selected.has(name);
      label.appendChild(input);
      label.appendChild(document.createTextNode(name));
      container.appendChild(label);
    }
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
    els.form.num_gpu.value = defaults.num_gpu;
    els.form.prompt_sz.value = defaults.prompt_sz;
    els.form.response_sz.value = defaults.response_sz;
    els.form.n_concurrent_req.value = defaults.n_concurrent_req;
    els.form.system_overhead_gb.value = defaults.system_overhead_gb;
    els.form.vram_util.value = defaults.vram_util;
    els.form.kv_frag.value = defaults.kv_frag;
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

  function orderedKeys(row) {
    const keys = Object.keys(row);
    // Keep Model pinned as the leftmost column for horizontal scroll.
    if (!keys.includes("Model")) return keys;
    return ["Model", ...keys.filter((k) => k !== "Model")];
  }

  function renderTable(container, rows) {
    container.innerHTML = "";
    if (!rows || !rows.length) {
      container.textContent = "No rows for this selection.";
      return;
    }
    const keys = orderedKeys(rows[0]);
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const key of keys) {
      const th = document.createElement("th");
      th.textContent = key;
      if (key === "Model") th.className = "col-model";
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const key of keys) {
        const td = document.createElement("td");
        const value = row[key];
        td.textContent = value == null ? "" : String(value);
        const classes = [];
        if (key === "Model") classes.push("col-model");
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

  async function calculate() {
    const id = ++requestId;
    const body = readConfig();
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

      renderTable(els.tables.memory_footprint, data.memory_footprint);
      renderTable(els.tables.concurrent_capacity, data.concurrent_capacity);
      renderTable(els.tables.performance, data.performance);

      if (data.warnings && data.warnings.length) {
        els.warnings.hidden = false;
        els.warnings.textContent = data.warnings.join("\n\n");
      } else {
        els.warnings.hidden = true;
        els.warnings.textContent = "";
      }
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
    scheduleCalculate();
  }

  async function init() {
    const res = await fetch("/api/options");
    if (!res.ok) throw new Error(`Failed to load options (${res.status})`);
    const opts = await res.json();

    applyDefaults(opts.defaults);
    fillSelect(els.quant, opts.weight_quants, opts.defaults.quant);
    fillSelect(els.kvQuant, opts.kv_quants, opts.defaults.kv_quant);
    fillChecklist(
      els.gpuList,
      opts.gpus,
      opts.defaults.gpu_names,
      "gpu_names"
    );
    fillChecklist(
      els.modelList,
      opts.models,
      opts.defaults.model_names,
      "model_names"
    );

    els.form.addEventListener("input", scheduleCalculate);
    els.form.addEventListener("change", scheduleCalculate);
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
