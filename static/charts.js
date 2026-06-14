/* charts.js — Canvas-based chart utilities for the lending dashboard */

const Charts = (() => {
  const FONTS = { mono: "'JetBrains Mono', monospace", sans: "'Inter', sans-serif" };
  const COLORS = {
    primary: '#4F8EF7', accent: '#00E5C3', danger: '#FF4D6A',
    warning: '#FFB547', success: '#22C55E', purple: '#A78BFA',
    text2: '#A8B3CF', text3: '#6B7A9C', border: 'rgba(79,142,247,0.12)',
    GRADE: { AAA:'#22C55E', AA:'#4ADE80', A:'#86EFAC', BBB:'#FFB547', BB:'#FB923C', B:'#F87171', CCC:'#FF4D6A' },
    PALETTE: ['#4F8EF7','#00E5C3','#A78BFA','#FFB547','#FF4D6A','#22C55E','#FB923C','#F472B6'],
  };

  function dpr() { return window.devicePixelRatio || 1; }

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const d = dpr();
    canvas.width  = rect.width  * d;
    canvas.height = rect.height * d;
    const ctx = canvas.getContext('2d');
    ctx.scale(d, d);
    return { ctx, W: rect.width, H: rect.height };
  }

  // ── BAR CHART ────────────────────────────────────────────
  function barChart(canvas, data, opts = {}) {
    const { ctx, W, H } = setupCanvas(canvas);
    const { labels, values, color = COLORS.primary, horizontal = false } = data;
    const { padLeft = 60, padBottom = 36, padTop = 20, padRight = 20 } = opts;

    ctx.clearRect(0, 0, W, H);

    const n = values.length;
    const max = Math.max(...values) * 1.15 || 1;

    if (!horizontal) {
      const areaW = W - padLeft - padRight;
      const areaH = H - padTop - padBottom;
      const barW  = (areaW / n) * 0.65;
      const gap   = (areaW / n) * 0.35;

      // Grid lines
      const gridLines = 4;
      for (let i = 0; i <= gridLines; i++) {
        const y = padTop + areaH - (i / gridLines) * areaH;
        ctx.beginPath();
        ctx.strokeStyle = COLORS.border;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.moveTo(padLeft, y); ctx.lineTo(W - padRight, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.text3;
        ctx.font = `10px ${FONTS.mono}`;
        ctx.textAlign = 'right';
        const v = (max * i / gridLines);
        ctx.fillText(formatK(v), padLeft - 6, y + 3);
      }

      // Bars
      values.forEach((v, i) => {
        const x  = padLeft + i * (areaW / n) + gap / 2;
        const bh = (v / max) * areaH;
        const y  = padTop + areaH - bh;

        // Gradient fill
        const grad = ctx.createLinearGradient(x, y, x, padTop + areaH);
        const col  = Array.isArray(color) ? color[i % color.length] : color;
        grad.addColorStop(0, col);
        grad.addColorStop(1, col + '40');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, bh, [4, 4, 0, 0]);
        ctx.fill();

        // Label
        ctx.fillStyle = COLORS.text3;
        ctx.font = `10px ${FONTS.sans}`;
        ctx.textAlign = 'center';
        const lbl = String(labels[i]);
        ctx.fillText(lbl.length > 8 ? lbl.slice(0, 7) + '…' : lbl, x + barW / 2, H - 10);

        // Value on top
        if (v > 0) {
          ctx.fillStyle = COLORS.text2;
          ctx.font = `bold 10px ${FONTS.mono}`;
          ctx.fillText(formatK(v), x + barW / 2, y - 4);
        }
      });
    } else {
      // Horizontal bar
      const areaW = W - padLeft - padRight;
      const areaH = H - padTop - 10;
      const barH  = (areaH / n) * 0.55;
      const gap   = (areaH / n) * 0.45;

      values.forEach((v, i) => {
        const y   = padTop + i * (areaH / n) + gap / 2;
        const bw  = (v / max) * areaW;
        const col = Array.isArray(color) ? color[i % color.length] : color;
        const grad = ctx.createLinearGradient(padLeft, y, padLeft + bw, y);
        grad.addColorStop(0, col);
        grad.addColorStop(1, col + '60');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(padLeft, y, bw, barH, [0, 4, 4, 0]);
        ctx.fill();
        ctx.fillStyle = COLORS.text2;
        ctx.font = `11px ${FONTS.sans}`;
        ctx.textAlign = 'right';
        ctx.fillText(labels[i], padLeft - 6, y + barH / 2 + 4);
        ctx.fillStyle = COLORS.text3;
        ctx.font = `10px ${FONTS.mono}`;
        ctx.textAlign = 'left';
        ctx.fillText(formatK(v), padLeft + bw + 4, y + barH / 2 + 4);
      });
    }
  }

  // ── LINE CHART ────────────────────────────────────────────
  function lineChart(canvas, datasets, labels, opts = {}) {
    const { ctx, W, H } = setupCanvas(canvas);
    const { padLeft = 55, padBottom = 36, padTop = 20, padRight = 20 } = opts;

    ctx.clearRect(0, 0, W, H);

    const allVals = datasets.flatMap(d => d.values);
    const min = Math.min(0, ...allVals);
    const max = Math.max(...allVals) * 1.15 || 1;
    const range = max - min;
    const areaW = W - padLeft - padRight;
    const areaH = H - padTop - padBottom;
    const n = labels.length;

    const px = i => padLeft + (i / (n - 1)) * areaW;
    const py = v => padTop + areaH - ((v - min) / range) * areaH;

    // Grid
    const gl = 4;
    for (let i = 0; i <= gl; i++) {
      const y = padTop + (i / gl) * areaH;
      ctx.beginPath(); ctx.strokeStyle = COLORS.border; ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); ctx.moveTo(padLeft, y); ctx.lineTo(W - padRight, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.text3; ctx.font = `10px ${FONTS.mono}`; ctx.textAlign = 'right';
      ctx.fillText(formatK(max - (max - min) * i / gl), padLeft - 6, y + 3);
    }

    // X labels
    labels.forEach((l, i) => {
      ctx.fillStyle = COLORS.text3; ctx.font = `10px ${FONTS.sans}`; ctx.textAlign = 'center';
      const skip = Math.ceil(n / 8);
      if (i % skip === 0) ctx.fillText(l, px(i), H - 10);
    });

    // Lines & areas
    datasets.forEach(({ values, color, label }) => {
      // Area fill
      ctx.beginPath();
      ctx.moveTo(px(0), py(values[0]));
      values.forEach((v, i) => { if (i > 0) ctx.lineTo(px(i), py(v)); });
      ctx.lineTo(px(n - 1), py(min));
      ctx.lineTo(px(0), py(min));
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + areaH);
      grad.addColorStop(0, color + '30');
      grad.addColorStop(1, color + '00');
      ctx.fillStyle = grad; ctx.fill();

      // Line
      ctx.beginPath();
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      values.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
      ctx.stroke();

      // Dots at endpoints
      [0, n - 1].forEach(i => {
        ctx.beginPath();
        ctx.arc(px(i), py(values[i]), 4, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
        ctx.strokeStyle = '#141C2F'; ctx.lineWidth = 2; ctx.stroke();
      });
    });
  }

  // ── DONUT CHART ───────────────────────────────────────────
  function donutChart(canvas, data, opts = {}) {
    const { ctx, W, H } = setupCanvas(canvas);
    const { labels, values, colors = COLORS.PALETTE } = data;
    const { thickness = 0.38, gap = 0.025 } = opts;

    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const r  = Math.min(W, H) / 2 * 0.85;
    const ir = r * (1 - thickness);
    const total = values.reduce((a, b) => a + b, 0) || 1;
    let angle = -Math.PI / 2;

    values.forEach((v, i) => {
      const sweep = (v / total) * (Math.PI * 2 - gap * values.length);
      ctx.beginPath();
      ctx.arc(cx, cy, r, angle, angle + sweep);
      ctx.arc(cx, cy, ir, angle + sweep, angle, true);
      ctx.closePath();
      ctx.fillStyle = colors[i % colors.length];
      ctx.fill();
      angle += sweep + gap;
    });

    // Center glow
    const grd = ctx.createRadialGradient(cx, cy, ir * 0.3, cx, cy, ir);
    grd.addColorStop(0, 'rgba(79,142,247,0.08)');
    grd.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(cx, cy, ir, 0, Math.PI * 2);
    ctx.fill();
  }

  // ── SCATTER PLOT ──────────────────────────────────────────
  function scatterPlot(canvas, points, opts = {}) {
    const { ctx, W, H } = setupCanvas(canvas);
    const { padLeft = 55, padBottom = 40, padTop = 20, padRight = 20 } = opts;

    ctx.clearRect(0, 0, W, H);
    if (!points.length) return;

    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs) || 1;
    const yMin = 0, yMax = Math.max(...ys) * 1.2 || 1;
    const areaW = W - padLeft - padRight;
    const areaH = H - padTop - padBottom;

    const px = x => padLeft + ((x - xMin) / (xMax - xMin || 1)) * areaW;
    const py = y => padTop + areaH - (y / yMax) * areaH;

    // Grid
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (i / 4) * areaH;
      ctx.beginPath(); ctx.strokeStyle = COLORS.border; ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); ctx.moveTo(padLeft, y); ctx.lineTo(W - padRight, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.text3; ctx.font = `10px ${FONTS.mono}`; ctx.textAlign = 'right';
      ctx.fillText(formatK(yMax - yMax * i / 4), padLeft - 4, y + 3);
    }

    // Axis labels
    ctx.fillStyle = COLORS.text3; ctx.font = `10px ${FONTS.sans}`; ctx.textAlign = 'center';
    if (opts.xLabel) ctx.fillText(opts.xLabel, W / 2, H - 4);

    // Points
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), p.r || 4, 0, Math.PI * 2);
      ctx.fillStyle = (p.color || COLORS.primary) + 'CC';
      ctx.fill();
      ctx.strokeStyle = p.color || COLORS.primary;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  // ── SPARKLINE ─────────────────────────────────────────────
  function sparkline(canvas, values, color = COLORS.primary) {
    const { ctx, W, H } = setupCanvas(canvas);
    ctx.clearRect(0, 0, W, H);
    if (!values.length) return;
    const min = Math.min(...values), max = Math.max(...values) || 1;
    const range = max - min || 1;
    const px = i => (i / (values.length - 1)) * W;
    const py = v => H - ((v - min) / range) * H * 0.85 - 2;

    ctx.beginPath();
    values.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
    ctx.lineTo(px(values.length - 1), H);
    ctx.lineTo(0, H);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, color + '50'); grad.addColorStop(1, color + '00');
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath();
    values.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.lineJoin = 'round';
    ctx.stroke();
  }

  // ── STACKED BAR ───────────────────────────────────────────
  function stackedBar(canvas, data, opts = {}) {
    const { ctx, W, H } = setupCanvas(canvas);
    const { labels, series, colors = COLORS.PALETTE } = data;
    const { padLeft = 60, padBottom = 36, padTop = 20, padRight = 20 } = opts;

    ctx.clearRect(0, 0, W, H);
    const n = labels.length;
    const totals = labels.map((_, i) => series.reduce((s, sr) => s + (sr.values[i] || 0), 0));
    const max = Math.max(...totals) * 1.1 || 1;
    const areaW = W - padLeft - padRight;
    const areaH = H - padTop - padBottom;
    const barW = (areaW / n) * 0.65;
    const gapW = (areaW / n) * 0.35;

    // Grid
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (i / 4) * areaH;
      ctx.beginPath(); ctx.strokeStyle = COLORS.border; ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); ctx.moveTo(padLeft, y); ctx.lineTo(W - padRight, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.text3; ctx.font = `10px ${FONTS.mono}`; ctx.textAlign = 'right';
      ctx.fillText(formatK(max * (1 - i / 4)), padLeft - 6, y + 3);
    }

    labels.forEach((lbl, i) => {
      let baseY = padTop + areaH;
      series.forEach((sr, si) => {
        const v  = sr.values[i] || 0;
        const bh = (v / max) * areaH;
        const x  = padLeft + i * (areaW / n) + gapW / 2;
        const y  = baseY - bh;
        ctx.fillStyle = colors[si % colors.length] + 'CC';
        ctx.beginPath();
        if (si === series.length - 1) ctx.roundRect(x, y, barW, bh, [4, 4, 0, 0]);
        else ctx.rect(x, y, barW, bh);
        ctx.fill();
        baseY = y;
      });
      ctx.fillStyle = COLORS.text3; ctx.font = `10px ${FONTS.sans}`; ctx.textAlign = 'center';
      ctx.fillText(lbl, padLeft + i * (areaW / n) + barW / 2 + gapW / 2, H - 10);
    });
  }

  function formatK(v) {
    if (v === undefined || v === null || isNaN(v)) return '0';
    if (Math.abs(v) >= 1e7) return (v / 1e7).toFixed(1) + 'Cr';
    if (Math.abs(v) >= 1e5) return (v / 1e5).toFixed(1) + 'L';
    if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(0) + 'K';
    return v.toFixed(1);
  }

  return { barChart, lineChart, donutChart, scatterPlot, sparkline, stackedBar, formatK, COLORS };
})();
