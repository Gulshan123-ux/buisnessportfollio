"""
app.py — LendIQ ML Portfolio Optimizer
Flask backend with scikit-learn ML models + OpenAI GPT-4o AI recommendations
"""
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import pandas as pd
import io, json, traceback, os

from ml.data_generator import generate_portfolio, normalize_upload
from ml.models import LendingMLModels
from ml.analyzer import PortfolioAnalyzer

app  = Flask(__name__)
CORS(app)
app.secret_key = 'lendiq-ml-secret-42'



try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    _openai_available = True
    print("✅ OpenAI client initialized successfully")
except Exception as e:
    _openai_client = None
    _openai_available = False
    print(f"⚠️  OpenAI not available: {e}")

# ── Global state (single-user local app) ──────────────────
_df:    pd.DataFrame | None = None
_ml:    LendingMLModels     = LendingMLModels()
_source: str                = 'none'
_last_train_info: dict      = {}

def get_analyzer() -> PortfolioAnalyzer | None:
    if _df is None:
        return None
    return PortfolioAnalyzer(_df, _ml)

# ── Pages ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── Generate synthetic data ───────────────────────────────
@app.route('/api/generate', methods=['POST'])
def generate():
    global _df, _source, _last_train_info
    try:
        n = int(request.json.get('n', 2000))
        _df = generate_portfolio(n)
        train_info = _ml.fit(_df)
        _last_train_info = train_info
        _source = 'demo'
        return jsonify({'status':'ok','rows':len(_df),'source':'demo','ml':train_info})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── Upload CSV ────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload():
    global _df, _source, _last_train_info
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error':'No file uploaded'}), 400

        raw = pd.read_csv(file)
        if len(raw) == 0:
            return jsonify({'error': 'CSV is empty'}), 400

        print(f"[Upload] Raw CSV: {len(raw)} rows, columns: {list(raw.columns)}")
        _df = normalize_upload(raw)
        print(f"[Upload] Normalized: {len(_df)} rows, columns: {list(_df.columns)}")
        print(f"[Upload] credit_quality values: {_df['credit_quality'].unique()}")
        print(f"[Upload] default_indicator sum: {_df['default_indicator'].sum()}")
        print(f"[Upload] loan_status counts: {_df['loan_status'].value_counts().to_dict()}")

        train_info = _ml.fit(_df)
        _last_train_info = train_info
        _source = 'csv'

        return jsonify({
            'status': 'ok',
            'rows': len(_df),
            'columns': list(raw.columns),
            'normalized_columns': list(_df.columns),
            'source': 'csv',
            'ml': train_info,
            'summary': {
                'total_loans': len(_df),
                'default_rate': round(float(_df['default_indicator'].mean() * 100), 1),
                'avg_credit_score': int(_df['credit_score'].mean()),
                'total_aum': int(_df['loan_size'].sum()),
                'grades_found': _df['credit_quality'].unique().tolist(),
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# ── Portfolio Metrics ─────────────────────────────────────
@app.route('/api/metrics')
def metrics():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data loaded'}), 400
    return jsonify({**a.portfolio_metrics(), 'source': _source})

# ── Risk Distribution ─────────────────────────────────────
@app.route('/api/risk')
def risk():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data'}), 400
    return jsonify({
        'distribution': a.risk_distribution(),
        'featureImportance': a.feature_importance(),
        'scatter': a.scatter_data('credit_score', 'default_prob', 'credit_quality'),
    })

# ── Early Warning ─────────────────────────────────────────
@app.route('/api/early-warning')
def early_warning():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data'}), 400
    grade = request.args.get('grade', '')
    signals = a.early_warning_signals(top_n=200)
    if grade:
        signals = [s for s in signals if s.get('credit_quality') == grade]
    return jsonify({
        'signals':  signals[:50],
        'stats':    a.ew_distribution(),
    })

# ── Segments ──────────────────────────────────────────────
@app.route('/api/segments')
def segments():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data'}), 400
    by = request.args.get('by', 'geography')
    return jsonify({'data': a.segment_breakdown(by)})

# ── Cohort Analysis ───────────────────────────────────────
@app.route('/api/cohorts')
def cohorts():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data'}), 400
    return jsonify({'data': a.cohort_analysis()})

# ── Pricing & ROI ─────────────────────────────────────────
@app.route('/api/pricing')
def pricing():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data'}), 400
    prod = a.segment_breakdown('product_type')
    grade = a.segment_breakdown('credit_quality')
    return jsonify({'products': prod, 'grades': grade})

# ── Customer Clusters ─────────────────────────────────────
@app.route('/api/clusters')
def clusters():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data'}), 400
    return jsonify({'data': a.cluster_profiles()})

# ── Recommendations (ML + GPT-4o AI) ─────────────────────
@app.route('/api/recommendations')
def recommendations():
    a = get_analyzer()
    if a is None:
        return jsonify({'error':'No data'}), 400

    M   = a.portfolio_metrics()
    rd  = a.risk_distribution()
    ew  = a.ew_distribution()
    seg = a.segment_breakdown('geography')
    prod_seg = a.segment_breakdown('product_type')

    # Always generate ML-based recommendations from actual data
    ml_recs = _build_ml_recommendations(M, rd, ew, seg, prod_seg)

    # Try to enrich with OpenAI GPT-4o
    if _openai_available:
        try:
            ai_recs = _generate_openai_recommendations(M, rd, ew, seg, prod_seg, _source)
            # Combine: ML recs first, then AI narrative
            return jsonify({'data': ml_recs + ai_recs})
        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            # Fall back to ML-only recommendations
            return jsonify({'data': ml_recs, 'ai_error': str(e)})

    return jsonify({'data': ml_recs})


def _build_ml_recommendations(M, rd, ew, seg, prod_seg):
    """Build data-driven recommendations from actual portfolio metrics."""
    ccc = next((r for r in rd if r['grade'] == 'CCC'), {})
    bb  = next((r for r in rd if r['grade'] == 'BB'),  {})
    b   = next((r for r in rd if r['grade'] == 'B'),   {})
    high_risk = ew.get('high', 0)
    total = M.get('totalLoans', 0)

    # Find best and worst segments by default rate
    if seg:
        best_geo  = min(seg, key=lambda x: x['defaultRate'])
        worst_geo = max(seg, key=lambda x: x['defaultRate'])
    else:
        best_geo = worst_geo = {}

    if prod_seg:
        best_prod  = min(prod_seg, key=lambda x: x['defaultRate'])
        worst_prod = max(prod_seg, key=lambda x: x['defaultRate'])
    else:
        best_prod = worst_prod = {}

    benchmark = 8.0
    dr_status = '⚠️ ABOVE' if M.get('defaultRate', 0) > benchmark else '✅ BELOW'

    recs = [
        {
            'icon': '🎯', 'title': 'Executive Summary', 'type': 'info',
            'body': (
                f"Portfolio of <strong>{M.get('totalLoans', 0):,} loans</strong> with AUM of "
                f"₹{M.get('totalDisbursed', 0)/1e7:.2f}Cr. "
                f"Actual default rate: <strong>{M.get('defaultRate', 0):.1f}%</strong> — "
                f"{dr_status} the {benchmark}% industry benchmark. "
                f"ML-predicted average default probability: <strong>{M.get('avgDefaultProb', 0):.2f}%</strong>. "
                f"Average Credit Score: <strong>{M.get('avgCreditScore', 0)}</strong>. "
                f"Avg Early Warning Score: <strong>{M.get('avgEWScore', 0)}/100</strong>."
            )
        },
        {
            'icon': '🤖', 'title': 'ML Model — Default Risk Analysis',
            'type': 'danger' if M.get('avgDefaultProb', 0) > 10 else 'warning' if M.get('avgDefaultProb', 0) > 5 else 'info',
            'body': (
                f"Random Forest + Gradient Boosting ensemble trained on <strong>{total:,} loans</strong>. "
                f"CCC-grade segment: <strong>{ccc.get('count', 0)} loans</strong>, "
                f"ML-estimated default probability: <strong>{ccc.get('avgDefaultProb', 0):.1f}%</strong>. "
                f"BB-grade: {bb.get('count', 0)} loans at {bb.get('avgDefaultProb', 0):.1f}% ML prob. "
                f"B-grade: {b.get('count', 0)} loans at {b.get('avgDefaultProb', 0):.1f}% ML prob. "
                + (f"<br><strong>Action:</strong> Halt new CCC originations. Tighten underwriting for BB/B-grade loans (DTI > 0.4 or LTV > 1.5)." if ccc.get('count', 0) > 0 else "")
            )
        },
        {
            'icon': '⚡', 'title': 'Early Warning — Isolation Forest Signals',
            'type': 'warning',
            'body': (
                f"Isolation Forest anomaly detection: <strong>{high_risk} active loans</strong> flagged as high-risk (score > 60). "
                f"<strong>{ew.get('medium', 0)}</strong> loans in medium-risk zone (40–60). "
                f"<strong>{ew.get('shocks', 0)}</strong> loans show spending shock signals. "
                f"Risk distribution — High: {high_risk} | Medium: {ew.get('medium', 0)} | Low: {ew.get('low', 0)}. "
                f"<br><strong>Action:</strong> Deploy automated nudges at DPD-7, personal outreach at DPD-15."
            )
        },
        {
            'icon': '🌏', 'title': 'Geographic Risk Analysis',
            'type': 'success' if best_geo.get('defaultRate', 0) < 5 else 'info',
            'body': (
                f"Best performing geography: <strong>{best_geo.get('segment', 'N/A')}</strong> "
                f"(default rate: {best_geo.get('defaultRate', 0):.1f}%, "
                f"AUM: ₹{best_geo.get('totalDisbursed', 0)/1e5:.1f}L). "
                f"Highest risk geography: <strong>{worst_geo.get('segment', 'N/A')}</strong> "
                f"(default rate: {worst_geo.get('defaultRate', 0):.1f}%, "
                f"ML prob: {worst_geo.get('avgDefaultProb', 0):.1f}%). "
                f"<br><strong>Action:</strong> Expand originations in {best_geo.get('segment', 'N/A')}; tighten criteria in {worst_geo.get('segment', 'N/A')}."
            )
        },
        {
            'icon': '📦', 'title': 'Product Portfolio Optimization',
            'type': 'success',
            'body': (
                f"Best performing product: <strong>{best_prod.get('segment', 'N/A')}</strong> "
                f"(default rate: {best_prod.get('defaultRate', 0):.1f}%, avg pricing: {best_prod.get('avgPricing', 0):.1f}%). "
                f"Highest-risk product: <strong>{worst_prod.get('segment', 'N/A')}</strong> "
                f"(default rate: {worst_prod.get('defaultRate', 0):.1f}%). "
                f"<br><strong>Action:</strong> Increase allocation to {best_prod.get('segment', 'N/A')}. "
                f"Review pricing model for {worst_prod.get('segment', 'N/A')}."
            )
        },
        {
            'icon': '💰', 'title': 'Pricing & ROI Recommendation',
            'type': 'info',
            'body': (
                f"Current avg pricing: <strong>{M.get('avgPricing', 0):.2f}%</strong>. "
                f"At-risk AUM: ₹{M.get('atRiskAUM', 0)/1e7:.2f}Cr. "
                f"Risk-adjusted pricing floor: <code>Rate = Base + (PD × LGD × 100)</code>. "
                f"Total value proxy: ₹{M.get('totalValueProxy', 0)/1e7:.2f}Cr. "
                f"<br><strong>Action:</strong> Re-price BB/B segments upward by 2–3%. "
                f"Prioritise high-margin, low-risk originations."
            )
        },
    ]
    return recs


def _generate_openai_recommendations(M, rd, ew, seg, prod_seg, source):
    """Call OpenAI GPT-4o to generate advanced AI-powered narrative recommendations."""

    # Build a concise data summary for the prompt
    risk_summary = [
        f"{r['grade']}: {r['count']} loans, AUM ₹{r['aum']/1e5:.1f}L, "
        f"actual default {r['defaultRate']:.1f}%, ML prob {r['avgDefaultProb']:.1f}%"
        for r in rd if r['count'] > 0
    ]
    geo_summary = [
        f"{s['segment']}: {s['count']} loans, default {s['defaultRate']:.1f}%, "
        f"ML prob {s['avgDefaultProb']:.1f}%"
        for s in (seg or [])
    ]
    prod_summary = [
        f"{s['segment']}: {s['count']} loans, default {s['defaultRate']:.1f}%, "
        f"pricing {s['avgPricing']:.1f}%"
        for s in (prod_seg or [])
    ]

    prompt = f"""You are a senior credit risk analyst at a leading Indian digital lending NBFC. 
Analyse the following real portfolio data and provide 2 concise, actionable, data-specific recommendations.

PORTFOLIO DATA ({source.upper()} — {M.get('totalLoans', 0)} loans):
- Total AUM: ₹{M.get('totalDisbursed', 0)/1e7:.2f} Crore
- Actual Default Rate: {M.get('defaultRate', 0):.1f}% (Benchmark: 8%)
- ML-Predicted Avg Default Probability: {M.get('avgDefaultProb', 0):.2f}%
- Avg Credit Score: {M.get('avgCreditScore', 0)}
- Avg Pricing Rate: {M.get('avgPricing', 0):.2f}%
- At-Risk AUM: ₹{M.get('atRiskAUM', 0)/1e7:.2f} Crore
- Early Warning — High Risk: {ew.get('high', 0)}, Medium: {ew.get('medium', 0)}, Spending Shocks: {ew.get('shocks', 0)}

RISK GRADE DISTRIBUTION:
{chr(10).join(risk_summary) if risk_summary else 'No grade data'}

GEOGRAPHY BREAKDOWN:
{chr(10).join(geo_summary) if geo_summary else 'No geography data'}

PRODUCT BREAKDOWN:
{chr(10).join(prod_summary) if prod_summary else 'No product data'}

Respond with exactly 2 JSON objects in a JSON array. Each object must have:
- "icon": an emoji
- "title": short title (max 8 words)  
- "type": one of "info", "warning", "danger", "success"
- "body": 3-4 sentences of specific, data-driven insight and action items using the actual numbers above

Return ONLY the JSON array, no other text."""

    response = _openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a credit risk AI analyst. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    ai_recs = json.loads(raw)
    # Tag AI recommendations
    for rec in ai_recs:
        rec['title'] = '🧠 AI: ' + rec.get('title', 'Insight')
    return ai_recs


# ── Predict Single Loan ───────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    if not _ml.fitted:
        return jsonify({'error':'Model not trained yet'}), 400
    loan = request.json or {}
    result = _ml.predict_loan(loan)
    return jsonify(result)

@app.route('/api/sample-csv')
def sample_csv():
    from flask import Response
    csv = (
        "loan_id,loan_amount,cibil_score,risk_grade,region,employment,product,interest_rate,"
        "is_default,dpd,tenure,channel,income,dti\n"
        "LN-000001,150000,720,A,Metro,Salaried,Personal Loan,14.5,0,Current,12,Digital-Direct,55000,0.25\n"
        "LN-000002,500000,680,BBB,Tier-2,Self-Employed,SME Working Capital,16.0,0,Current,24,Partner-Fintech,80000,0.35\n"
        "LN-000003,25000,620,B,Rural,Gig Worker,Micro Loan,22.0,1,Default,6,Agent Network,20000,0.45\n"
        "LN-000004,80000,760,AA,Metro,Salaried,Unsecured Consumer,12.5,0,Current,18,Digital-Direct,70000,0.20\n"
        "LN-000005,15000,580,CCC,Semi-Urban,Informal,BNPL,26.0,1,DPD30+,3,Social Media,15000,0.60\n"
    )
    return Response(csv, mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=lendiq_sample_template.csv'})

# ── Reset State ───────────────────────────────────────────
@app.route('/api/reset', methods=['POST'])
def reset():
    global _df, _source, _ml, _last_train_info
    _df = None
    _source = 'none'
    _ml = LendingMLModels()
    _last_train_info = {}
    return jsonify({'status': 'reset'})

# ── Data Status ───────────────────────────────────────────
@app.route('/api/status')
def status():
    return jsonify({
        'loaded': _df is not None,
        'rows': len(_df) if _df is not None else 0,
        'source': _source,
        'ml_fitted': _ml.fitted,
        'openai_available': _openai_available,
    })

if __name__ == '__main__':
    print("🚀 LendIQ ML Server starting at http://localhost:5050")
    print(f"🤖 OpenAI GPT-4o: {'✅ Enabled' if _openai_available else '❌ Not available'}")
    app.run(debug=True, port=5050, use_reloader=False)
