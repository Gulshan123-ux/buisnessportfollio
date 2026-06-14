// app.js — LendIQ ML Frontend (calls Python Flask API)

const API = ''; // same origin

// ── AUTH FLOW ──────────────────────────────────────────────
function showScreen(name) {
  // name: 'signin' | 'signup' | 'upload'
  document.getElementById('signin-screen').style.display = name === 'signin' ? 'flex' : 'none';
  document.getElementById('signup-screen').style.display = name === 'signup' ? 'flex' : 'none';
  const us = document.getElementById('upload-screen');
  us.style.display = name === 'upload' ? 'flex' : 'none';
}

function handleSignIn() {
  const email = document.getElementById('si-email').value.trim();
  const pass  = document.getElementById('si-pass').value;
  const err   = document.getElementById('si-error');
  if (!email || !pass) { err.textContent = 'Please enter email and password.'; return; }
  if (!email.includes('@')) { err.textContent = 'Please enter a valid email.'; return; }
  err.textContent = '';
  // Store user session (demo)
  sessionStorage.setItem('lendiq_user', JSON.stringify({ email, name: email.split('@')[0] }));
  showScreen('upload');
}

function handleSignUp() {
  const name  = document.getElementById('su-name').value.trim();
  const email = document.getElementById('su-email').value.trim();
  const pass  = document.getElementById('su-pass').value;
  const err   = document.getElementById('su-error');
  if (!name || !email || !pass) { err.textContent = 'Please fill all required fields.'; return; }
  if (!email.includes('@')) { err.textContent = 'Please enter a valid email.'; return; }
  if (pass.length < 6) { err.textContent = 'Password must be at least 6 characters.'; return; }
  err.textContent = '';
  sessionStorage.setItem('lendiq_user', JSON.stringify({ email, name }));
  showScreen('upload');
}
let currentPage = 'dashboard', segmentKey = 'geography';

const fmt = {
  cr:   v => '₹' + (v >= 1e7 ? (v/1e7).toFixed(1)+'Cr' : v >= 1e5 ? (v/1e5).toFixed(1)+'L' : (v/1e3).toFixed(0)+'K'),
  pct:  v => v + '%',
  num:  v => Number(v).toLocaleString('en-IN'),
  prob: v => v.toFixed(1) + '%',
};

// ── UPLOAD SCREEN ─────────────────────────────────────────
function selectOption(type) {
  document.getElementById('opt-upload').classList.toggle('selected', type === 'upload');
  document.getElementById('opt-demo').classList.toggle('selected',  type === 'demo');
  if (type === 'demo') {
    document.getElementById('drop-zone').style.display = 'none';
    bootDemo();
  } else {
    document.getElementById('drop-zone').style.display = 'flex';
  }
}

async function bootDemo() {
  showStatus('Generating 2,000 synthetic loans…');
  try {
    const res = await fetch(`${API}/api/generate`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({n:2000})
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showStatus(`ML models trained! AUC: ${data.ml?.auc || '—'}`);
    await sleep(600);
    launchApp(`Demo • ${fmt.num(data.rows)} loans`);
  } catch(e) { showStatus('Error: ' + e.message, true); }
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  showStatus(`Uploading ${file.name} and training ML models…`);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch(`${API}/api/upload`, { method:'POST', body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showStatus(`✅ ${data.rows} loans loaded. ML AUC: ${data.ml?.auc || '—'}`);
    await sleep(800);
    launchApp(`Your Data • ${fmt.num(data.rows)} loans`);
  } catch(e) { showStatus('Error: ' + e.message, true); }
}

function showStatus(msg, isError=false) {
  const el = document.getElementById('upload-status');
  el.style.display = 'flex';
  document.getElementById('upload-status-text').textContent = msg;
  el.style.borderColor = isError ? 'var(--danger)' : 'var(--border-bright)';
}

async function launchApp(chipText) {
  const us = document.getElementById('upload-screen');
  us.style.opacity = '0'; us.style.transition = 'opacity 0.4s ease';
  await sleep(400);
  showScreen(''); // hide all screens
  us.style.display = 'none';

  document.getElementById('live-chip-text').textContent = chipText;
  const app = document.getElementById('app');
  app.style.display = 'flex';
  const bb = document.getElementById('back-btn');
  if (bb) bb.style.display = 'inline-flex';
  loadSidebar();
  renderPage('dashboard');
}

async function goBack() {
  // Reset server state
  try { await fetch('/api/reset', { method: 'POST' }); } catch(e) {}

  // Fade out app
  const app = document.getElementById('app');
  app.style.opacity = '0'; app.style.transition = 'opacity 0.3s ease';
  await sleep(300);
  app.style.display = 'none';
  app.style.opacity = '1';

  // Hide back button
  const bb = document.getElementById('back-btn');
  if (bb) bb.style.display = 'none';

  // Reset file input and status
  const fi = document.getElementById('csv-file-input');
  if (fi) fi.value = '';
  const statusEl = document.getElementById('upload-status');
  if (statusEl) statusEl.style.display = 'none';
  document.getElementById('opt-demo').classList.remove('selected');
  document.getElementById('opt-upload').classList.remove('selected');
  document.getElementById('drop-zone').style.display = 'none';

  // Go back to upload screen
  showScreen('upload');
}

// ── NAVIGATION ────────────────────────────────────────────
function navigate(page) {
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === page));
  document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  currentPage = page;
  const titles = {
    dashboard:       ['Portfolio Dashboard', 'ML-powered portfolio intelligence'],
    risk:            ['ML Risk Model', 'Random Forest default prediction'],
    'early-warning': ['Early Warning System', 'Isolation Forest anomaly detection'],
    segments:        ['Customer Segments', 'Performance by geography, product & channel'],
    cohorts:         ['Cohort Analysis', 'Origination vintage performance'],
    clusters:        ['ML Clusters (K-Means)', 'Unsupervised customer segmentation'],
    pricing:         ['Pricing & ROI', 'Rate optimisation & value creation'],
    recommendations: ['ML Recommendations', 'Credit policy memo for CRO'],
  };
  const [t, s] = titles[page] || ['Dashboard',''];
  document.getElementById('page-title').textContent = t;
  document.getElementById('page-sub').textContent   = s;
  renderPage(page);
  // Sync back button
  const bb = document.getElementById('back-btn');
  if (bb) bb.style.display = 'inline-flex';
}

function renderPage(page) {
  const fn = { dashboard, risk, 'early-warning': earlyWarning, segments, cohorts, clusters, pricing, recommendations }[page];
  if (fn) setTimeout(fn, 60);
}

async function refreshAll() {
  const status = await (await fetch(`${API}/api/status`)).json();
  if (status.source === 'demo') bootDemo();
  else loadSidebar(), renderPage(currentPage);
}

// ── SIDEBAR ───────────────────────────────────────────────
async function loadSidebar() {
  const m = await fetchJSON('/api/metrics');
  if (!m) return;
  document.getElementById('sb-aum').textContent   = fmt.cr(m.totalDisbursed);
  document.getElementById('sb-dr').textContent    = fmt.pct(m.defaultRate);
  document.getElementById('sb-ml').textContent    = fmt.prob(m.avgDefaultProb);
  document.getElementById('sb-loans').textContent = fmt.num(m.totalLoans);
  const ew = await fetchJSON('/api/early-warning');
  if (ew) document.getElementById('ew-badge').textContent = ew.stats?.high || 0;
}

// ── DASHBOARD ────────────────────────────────────────────
async function dashboard() {
  const [m, cohortData, prodData, geoData, empData] = await Promise.all([
    fetchJSON('/api/metrics'),
    fetchJSON('/api/cohorts'),
    fetchJSON('/api/segments?by=product_type'),
    fetchJSON('/api/segments?by=geography'),
    fetchJSON('/api/segments?by=employment_type'),
  ]);
  if (!m) return;

  document.getElementById('kpi-aum').textContent    = fmt.cr(m.totalDisbursed);
  document.getElementById('kpi-dr').textContent     = fmt.pct(m.defaultRate);
  document.getElementById('kpi-ml').textContent     = fmt.prob(m.avgDefaultProb);
  document.getElementById('kpi-del').textContent    = fmt.pct(m.delinquencyRate);
  document.getElementById('kpi-score').textContent  = m.avgCreditScore;
  document.getElementById('kpi-val').textContent    = fmt.cr(Math.abs(m.totalValueProxy));
  document.getElementById('kpi-atrisk').textContent = fmt.cr(m.atRiskAUM);
  document.getElementById('kpi-price').textContent  = fmt.pct(m.avgPricing);

  const cohorts = cohortData?.data || [];
  renderCanvas('chart-cohort-value', c => Charts.lineChart(c,
    [{ values: cohorts.map(x => x.valueCreated), color: Charts.COLORS.accent, label:'Value' }],
    cohorts.map(x => x.cohort)
  ));

  const prods = prodData?.data || [];
  renderCanvas('chart-product-donut', c => Charts.donutChart(c, {
    labels: prods.map(s => s.segment),
    values: prods.map(s => s.totalDisbursed),
    colors: Charts.COLORS.PALETTE,
  }));
  document.getElementById('donut-center').textContent = prods.length;
  document.getElementById('donut-legend').innerHTML = prods.map((s,i) => `
    <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text-2)">
      <div style="width:10px;height:10px;border-radius:2px;background:${Charts.COLORS.PALETTE[i]};flex-shrink:0"></div>
      <span style="flex:1">${s.segment}</span>
      <span style="font-family:monospace;color:var(--text-1)">${fmt.cr(s.totalDisbursed)}</span>
    </div>`).join('');

  const geo = geoData?.data || [];
  renderCanvas('chart-geo', c => Charts.barChart(c, { labels: geo.map(s=>s.segment), values: geo.map(s=>s.defaultRate), color: Charts.COLORS.danger }));
  const emp = empData?.data || [];
  renderCanvas('chart-emp', c => Charts.barChart(c, { labels: emp.map(s=>s.segment), values: emp.map(s=>s.totalDisbursed), color: Charts.COLORS.PALETTE }));
}

// ── RISK MODEL ───────────────────────────────────────────
async function risk() {
  const data = await fetchJSON('/api/risk');
  if (!data) return;
  const rd = data.distribution || [];
  const fi = data.featureImportance || [];
  const sc = data.scatter || [];

  const invGrade = rd.filter(r=>['AAA','AA','A','BBB'].includes(r.grade)).reduce((s,r)=>s+r.count,0);
  const spec     = rd.filter(r=>['BB','B'].includes(r.grade)).reduce((s,r)=>s+r.count,0);
  const dist     = rd.find(r=>r.grade==='CCC')?.count || 0;

  document.getElementById('risk-inv-grade').textContent = fmt.num(invGrade);
  document.getElementById('risk-spec').textContent      = fmt.num(spec);
  document.getElementById('risk-dist').textContent      = fmt.num(dist);
  document.getElementById('risk-auc').textContent       = '—';

  renderCanvas('chart-risk-dist', c => Charts.barChart(c, {
    labels: rd.map(r=>r.grade), values: rd.map(r=>r.count),
    color:  rd.map(r=>Charts.COLORS.GRADE[r.grade] || Charts.COLORS.primary),
  }));

  if (fi.length) {
    renderCanvas('chart-feat-imp', c => Charts.barChart(c, {
      labels: fi.map(f=>f.feature.replace('_',' ')),
      values: fi.map(f=>f.importance),
      color:  Charts.COLORS.accent,
    }));
  }

  renderCanvas('chart-scatter', c => Charts.scatterPlot(c,
    sc.map(p=>({ x: p.x, y: p.y, r: Math.sqrt(p.r||1)*1.2, color: Charts.COLORS.GRADE[p.label] || Charts.COLORS.primary })),
    { xLabel: 'Avg Credit Score →' }
  ));

  document.getElementById('risk-table-body').innerHTML = rd.map(r => `<tr>
    <td><span class="badge badge-grade grade-${r.grade}">${r.grade}</span></td>
    <td>${fmt.num(r.count)}</td><td>${fmt.cr(r.aum)}</td>
    <td><span class="badge ${r.defaultRate>15?'badge-danger':r.defaultRate>8?'badge-warning':'badge-success'}">${r.defaultRate}%</span></td>
    <td><span class="badge badge-grade">${r.avgDefaultProb?.toFixed(1)||'—'}%</span></td>
  </tr>`).join('');
}

// ── EARLY WARNING ────────────────────────────────────────
async function earlyWarning() {
  const grade = document.getElementById('ew-filter-grade')?.value || '';
  const data  = await fetchJSON(`/api/early-warning${grade ? '?grade='+grade : ''}`);
  if (!data) return;
  const stats   = data.stats || {};
  const signals = data.signals || [];

  document.getElementById('ew-high').textContent   = fmt.num(stats.high || 0);
  document.getElementById('ew-med').textContent    = fmt.num(stats.medium || 0);
  document.getElementById('ew-low').textContent    = fmt.num(stats.low || 0);
  document.getElementById('ew-shocks').textContent = fmt.num(stats.shocks || 0);

  const b = stats.buckets || {};
  renderCanvas('chart-ew-dist', c => Charts.barChart(c, {
    labels: Object.keys(b), values: Object.values(b),
    color: ['#22C55E','#86EFAC','#FFB547','#FB923C','#FF4D6A'],
  }));

  document.getElementById('warning-list').innerHTML = signals.slice(0,8).map(l => `
    <div class="warning-signal">
      <div class="ws-icon">${l.early_warning_score > 60 ? '🚨' : '⚠️'}</div>
      <div class="ws-info">
        <div class="ws-id">${l.loan_id}</div>
        <div class="ws-meta">${l.product_type} · ${l.geography} · <span class="grade-${l.credit_quality}">${l.credit_quality}</span></div>
      </div>
      <div class="ws-score">${l.early_warning_score}</div>
    </div>`).join('');

  document.getElementById('ew-table-body').innerHTML = signals.map(l => `<tr>
    <td>${l.loan_id}</td>
    <td><span class="badge ${l.early_warning_score>60?'badge-danger':l.early_warning_score>40?'badge-warning':'badge-success'}">${l.early_warning_score}</span></td>
    <td><span class="badge badge-grade">${((l.default_prob||0)*100).toFixed(1)}%</span></td>
    <td><span class="grade-${l.credit_quality}">${l.credit_quality}</span></td>
    <td>${l.product_type}</td><td>${l.geography}</td>
    <td>${((l.cashflow_consistency||0)*100).toFixed(0)}%</td>
    <td>${((l.dti_ratio||0)*100).toFixed(1)}%</td>
    <td>${l.spending_shock ? '⚡ Yes' : '—'}</td>
  </tr>`).join('');
}
async function loadEarlyWarning() { await earlyWarning(); }

// ── SEGMENTS ─────────────────────────────────────────────
let segTab = 'geography';
function setSegmentTab(key) {
  segTab = key;
  ['geo','emp','prod','channel'].forEach(t => document.getElementById('seg-tab-'+t)?.classList.remove('active'));
  const m = { geography:'geo', employment_type:'emp', product_type:'prod', acquisition_channel:'channel' };
  document.getElementById('seg-tab-'+m[key])?.classList.add('active');
  segments();
}

async function segments() {
  const data = await fetchJSON(`/api/segments?by=${segTab}`);
  if (!data) return;
  const seg = data.data || [];
  document.getElementById('seg-chart-title').textContent = `📊 AUM by ${segTab.replace('_',' ')}`;
  renderCanvas('chart-seg-aum', c => Charts.barChart(c, { labels: seg.map(s=>s.segment), values: seg.map(s=>s.totalDisbursed), color: Charts.COLORS.PALETTE }));
  renderCanvas('chart-seg-dr',  c => Charts.barChart(c, { labels: seg.map(s=>s.segment), values: seg.map(s=>s.defaultRate), color: Charts.COLORS.danger }));
  document.getElementById('seg-table-body').innerHTML = seg.map(s => `<tr>
    <td>${s.segment}</td><td>${fmt.num(s.count)}</td><td>${fmt.cr(s.totalDisbursed)}</td>
    <td><span class="badge ${s.defaultRate>15?'badge-danger':s.defaultRate>8?'badge-warning':'badge-success'}">${s.defaultRate}%</span></td>
    <td><span class="badge badge-grade">${s.avgDefaultProb?.toFixed(1)||'—'}%</span></td>
    <td>${s.avgScore}</td><td>${s.avgPricing}%</td>
  </tr>`).join('');
}

// ── COHORTS ──────────────────────────────────────────────
async function cohorts() {
  const data = await fetchJSON('/api/cohorts');
  if (!data) return;
  const c = data.data || [];
  renderCanvas('chart-cohort-dr', ch => Charts.lineChart(ch,
    [{ values: c.map(x=>x.defaultRate), color: Charts.COLORS.danger, label:'Default Rate' },
     { values: c.map(x=>x.delinqRate),  color: Charts.COLORS.warning, label:'Delinquency' }],
    c.map(x=>x.cohort)
  ));
  renderCanvas('chart-cohort-vc', ch => Charts.barChart(ch, { labels: c.map(x=>x.cohort), values: c.map(x=>x.valueCreated), color: Charts.COLORS.PALETTE }));
  document.getElementById('cohort-table-body').innerHTML = c.map(x => `<tr>
    <td><span class="badge badge-neutral">${x.cohort}</span></td>
    <td>${fmt.num(x.loans)}</td>
    <td><span class="badge ${x.defaultRate>12?'badge-danger':x.defaultRate>7?'badge-warning':'badge-success'}">${x.defaultRate}%</span></td>
    <td><span class="badge ${x.delinqRate>15?'badge-warning':'badge-neutral'}">${x.delinqRate}%</span></td>
    <td>${x.avgPricing}%</td>
    <td><span class="badge badge-grade">${x.avgEWScore}</span></td>
    <td style="color:${x.valueCreated>0?'var(--success)':'var(--danger)'}">${fmt.cr(Math.abs(x.valueCreated))}</td>
  </tr>`).join('');
}

// ── CLUSTERS (K-Means) ────────────────────────────────────
async function clusters() {
  const data = await fetchJSON('/api/clusters');
  if (!data) return;
  const cl = data.data || [];
  const colors = [Charts.COLORS.success, Charts.COLORS.primary, Charts.COLORS.warning, Charts.COLORS.danger];

  document.getElementById('cluster-cards').innerHTML = cl.map((c,i) => `
    <div class="kpi-card" style="--kpi-color:${colors[i]||Charts.COLORS.primary}">
      <div class="kpi-icon" style="font-size:24px">${['🟢','🔵','🟡','🔴'][i]||'⚪'}</div>
      <div class="kpi-label">${c.name}</div>
      <div class="kpi-value">${fmt.num(c.count)} loans</div>
      <div class="kpi-delta">AUM: ${fmt.cr(c.totalAUM)}</div>
      <div class="kpi-delta">Avg Score: ${c.avgCreditScore} · DR: ${c.defaultRate}%</div>
      <div class="kpi-delta">EW Score: ${c.avgEWScore}/100</div>
    </div>`).join('');

  renderCanvas('chart-cluster-aum', c => Charts.barChart(c, { labels: cl.map(x=>x.name), values: cl.map(x=>x.totalAUM), color: colors }));
  renderCanvas('chart-cluster-ew',  c => Charts.barChart(c, { labels: cl.map(x=>x.name), values: cl.map(x=>x.avgEWScore), color: colors }));
}

// ── PRICING ──────────────────────────────────────────────
async function pricing() {
  const [m, data] = await Promise.all([fetchJSON('/api/metrics'), fetchJSON('/api/pricing')]);
  if (!data) return;
  const prods  = data.products || [];
  const grades = data.grades   || [];
  const sorted = [...prods].sort((a,b)=>b.avgPricing-a.avgPricing);
  const best   = prods.reduce((a,b)=>(b.avgPricing-b.defaultRate*2>a.avgPricing-a.defaultRate*2?b:a), prods[0]||{});
  const worst  = prods.reduce((a,b)=>(b.avgPricing-b.defaultRate*2<a.avgPricing-a.defaultRate*2?b:a), prods[0]||{});

  document.getElementById('pr-avg').textContent   = fmt.pct(m?.avgPricing||0);
  document.getElementById('pr-best').textContent  = (best.segment||'—').split(' ')[0];
  document.getElementById('pr-worst').textContent = (worst.segment||'—').split(' ')[0];
  document.getElementById('pr-total').textContent = fmt.cr(Math.abs(m?.totalValueProxy||0));

  const gradeOrder = ['AAA','AA','A','BBB','BB','B','CCC'];
  const gMap = Object.fromEntries(grades.map(g=>[g.segment,g]));
  renderCanvas('chart-pricing-grade', c => Charts.barChart(c, {
    labels: gradeOrder, values: gradeOrder.map(g=>gMap[g]?.avgPricing||0),
    color:  gradeOrder.map(g=>Charts.COLORS.GRADE[g]||Charts.COLORS.primary),
  }));
  renderCanvas('chart-value-dr', c => Charts.scatterPlot(c,
    prods.map((s,i)=>({ x: s.avgScore||680, y: s.avgDefaultProb||s.defaultRate, r:8, color: Charts.COLORS.PALETTE[i] })),
    { xLabel: 'Avg Credit Score →' }
  ));

  document.getElementById('pricing-table-body').innerHTML = sorted.map((s,i) => `<tr>
    <td>${s.segment}</td><td>${fmt.num(s.count)}</td><td>${fmt.cr(s.totalDisbursed)}</td>
    <td><span class="badge ${s.defaultRate>12?'badge-danger':'badge-warning'}">${s.defaultRate}%</span></td>
    <td><span class="badge badge-grade">${s.avgDefaultProb?.toFixed(1)||'—'}%</span></td>
    <td>${s.avgPricing}%</td>
    <td style="color:var(--success)">${fmt.cr(s.count * s.avgPricing * 1000)}</td>
  </tr>`).join('');
}

// ── RECOMMENDATIONS ───────────────────────────────────────
async function recommendations() {
  const data = await fetchJSON('/api/recommendations');
  if (!data) return;
  const recs = data.data || [];
  const colors = { info:'var(--primary)', danger:'var(--danger)', warning:'var(--warning)', success:'var(--accent)' };
  document.getElementById('rec-cards').innerHTML = recs.map((r,i) => `
    <div class="card fade-in-up d${i+1}" style="border-left:3px solid ${colors[r.type]||colors.info}">
      <div class="card-header">
        <span style="font-size:20px">${r.icon}</span>
        <span class="card-title">${r.title}</span>
        <span class="badge badge-grade" style="margin-left:auto">ML-Powered</span>
      </div>
      <div class="card-body" style="font-size:13px;color:var(--text-2);line-height:1.8">${r.body}</div>
    </div>`).join('');
}

// ── HELPERS ───────────────────────────────────────────────
async function fetchJSON(url) {
  try {
    const res = await fetch(API + url);
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

function renderCanvas(id, fn) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const wrap = canvas.parentElement;
  if (wrap) { canvas.style.width='100%'; canvas.style.height = (wrap.clientHeight||200)+'px'; }
  requestAnimationFrame(() => fn(canvas));
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

window.addEventListener('resize', () => { if (currentPage) renderPage(currentPage); });
