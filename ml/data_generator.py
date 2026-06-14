"""
data_generator.py — Synthetic Loan Portfolio Generator (NumPy/Pandas)
Generates realistic Indian digital lending data for ML training & demo
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

GEOGRAPHIES   = ['Metro', 'Tier-2', 'Semi-Urban', 'Rural']
EMPLOYMENT    = ['Salaried', 'Self-Employed', 'Gig Worker', 'Small Business', 'Informal']
PRODUCTS      = ['Personal Loan', 'SME Working Capital', 'BNPL', 'Unsecured Consumer', 'Micro Loan']
CHANNELS      = ['Digital-Direct', 'Partner-Fintech', 'Agent Network', 'NBFC-BC', 'Social Media']
RISK_GRADES   = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
GRADE_WEIGHTS = [0.05, 0.10, 0.18, 0.22, 0.20, 0.15, 0.10]

GRADE_SCORE   = {'AAA':820,'AA':780,'A':740,'BBB':700,'BB':660,'B':620,'CCC':560}
GRADE_DEFAULT = {'AAA':0.01,'AA':0.02,'A':0.04,'BBB':0.07,'BB':0.12,'B':0.20,'CCC':0.35}
GRADE_RATE    = {'AAA':10,'AA':12,'A':14,'BBB':16,'BB':19,'B':22,'CCC':26}
LOAN_RANGES   = {
    'Personal Loan':       (50_000, 500_000),
    'SME Working Capital': (200_000, 2_000_000),
    'BNPL':                (5_000, 50_000),
    'Unsecured Consumer':  (10_000, 200_000),
    'Micro Loan':          (5_000, 50_000),
}
INCOME_MAP = {'Salaried':55000,'Small Business':80000}

def score_to_grade(score: float) -> str:
    if score >= 800: return 'AAA'
    if score >= 760: return 'AA'
    if score >= 720: return 'A'
    if score >= 680: return 'BBB'
    if score >= 640: return 'BB'
    if score >= 600: return 'B'
    return 'CCC'


def generate_portfolio(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    grades      = rng.choice(RISK_GRADES, size=n, p=GRADE_WEIGHTS)
    geographies = rng.choice(GEOGRAPHIES, size=n)
    employment  = rng.choice(EMPLOYMENT,  size=n)
    products    = rng.choice(PRODUCTS,    size=n)
    channels    = rng.choice(CHANNELS,    size=n)

    base_scores  = np.array([GRADE_SCORE[g] for g in grades], dtype=float)
    credit_scores = np.clip(
        (base_scores + rng.normal(0, 40, n)).round().astype(int), 300, 900
    )

    income_base  = np.where(
        employment == 'Salaried', 55000,
        np.where(employment == 'Small Business', 80000, 35000)
    )
    income       = np.clip(rng.normal(income_base, 20000), 10000, 500000).astype(int)

    # Loan sizes per product
    loan_sizes = np.array([
        int(np.clip(
            rng.normal((LOAN_RANGES[p][0]+LOAN_RANGES[p][1])/2,
                       (LOAN_RANGES[p][1]-LOAN_RANGES[p][0])/6),
            *LOAN_RANGES[p]
        )) for p in products
    ])

    tenures  = rng.choice([6, 12, 18, 24, 36], size=n)
    base_rates = np.array([GRADE_RATE[g] for g in grades], dtype=float)
    pricing  = np.round(base_rates + rng.uniform(-1, 1, n), 2)

    default_probs = np.array([GRADE_DEFAULT[g] for g in grades])
    is_default    = rng.random(n) < default_probs
    is_delinquent = (~is_default) & (rng.random(n) < default_probs * 2.5)

    cashflow_consistency = np.round(1 - rng.uniform(0.1, 0.9, n), 2)
    balance_volatility   = np.round(rng.uniform(0.05, 0.85, n), 2)
    spending_shock       = rng.random(n) < 0.15

    ltv = np.round(loan_sizes / (income * 12 * 0.4), 3).clip(0, 3)
    dti = np.round((loan_sizes / tenures) / income, 4).clip(0, 0.8)

    orig_months = rng.integers(0, 24, n)
    cohorts     = [f"C{m//6+1}" for m in orig_months]

    value_proxy = np.round(
        loan_sizes * pricing / 100 * np.where(is_default, -0.4, 0.8), 2
    )

    loan_status = np.where(is_default, 'Default',
                  np.where(is_delinquent, 'At-Risk', 'Active'))
    delinq_status = np.where(is_default, 'Default',
                    np.where(is_delinquent, 'DPD30+', 'Current'))

    df = pd.DataFrame({
        'loan_id':             [f'LN-{i+1:06d}' for i in range(n)],
        'geography':           geographies,
        'employment_type':     employment,
        'credit_score':        credit_scores,
        'income_proxy':        income,
        'credit_quality':      grades,
        'product_type':        products,
        'loan_size':           loan_sizes,
        'tenure_months':       tenures,
        'pricing_rate':        pricing,
        'ltv_ratio':           ltv,
        'dti_ratio':           dti,
        'delinquency_status':  delinq_status,
        'loan_status':         loan_status,
        'cashflow_consistency':cashflow_consistency,
        'balance_volatility':  balance_volatility,
        'spending_shock':      spending_shock.astype(int),
        'acquisition_channel': channels,
        'origination_month':   orig_months,
        'cohort':              cohorts,
        'default_indicator':   is_default.astype(int),
        'value_proxy':         value_proxy,
    })
    return df


def normalize_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Map common CSV column names to internal schema with robust handling."""
    col_map = {
        'loan_amount':'loan_size','amount':'loan_size','principal':'loan_size',
        'disbursed_amount':'loan_size','sanctioned_amount':'loan_size',
        'cibil_score':'credit_score','cibil':'credit_score','bureau_score':'credit_score',
        'interest_rate':'pricing_rate','rate':'pricing_rate','roi':'pricing_rate','irr':'pricing_rate',
        'risk_grade':'credit_quality','grade':'credit_quality','rating':'credit_quality',
        'region':'geography','location':'geography','tier':'geography',
        'employment':'employment_type','job_type':'employment_type',
        'product':'product_type','loan_type':'product_type','scheme':'product_type',
        'is_default':'default_indicator','default':'default_indicator','npa':'default_indicator','bad_flag':'default_indicator',
        'dpd':'delinquency_status','dpd_bucket':'delinquency_status','days_past_due':'delinquency_status',
        'tenure':'tenure_months','term':'tenure_months','months':'tenure_months',
        'channel':'acquisition_channel','source':'acquisition_channel',
        'income':'income_proxy','salary':'income_proxy','monthly_income':'income_proxy',
        'dti':'dti_ratio','foir':'dti_ratio',
        'ltv':'ltv_ratio',
        'month':'origination_month','disbursement_month':'origination_month',
        'vintage':'cohort',
        'cashflow':'cashflow_consistency','cf_score':'cashflow_consistency',
        'balance_vol':'balance_volatility',
        'shock':'spending_shock',
    }
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # ── Normalize credit_quality grades ──────────────────
    GRADE_ALIASES = {
        'aaa': 'AAA', 'aa': 'AA', 'a': 'A', 'bbb': 'BBB',
        'bb': 'BB', 'b': 'B', 'ccc': 'CCC',
        'investment': 'AAA', 'prime': 'AAA', 'good': 'A', 'fair': 'BBB',
        'poor': 'CCC', 'bad': 'CCC', 'high': 'CCC', 'medium': 'BBB', 'low': 'AA',
    }
    if 'credit_quality' in df.columns:
        df['credit_quality'] = df['credit_quality'].astype(str).str.strip().str.lower().map(
            lambda x: GRADE_ALIASES.get(x, x.upper() if x.upper() in ['AAA','AA','A','BBB','BB','B','CCC'] else None)
        )

    # Derive credit_quality from score if missing/null
    if 'credit_quality' not in df.columns or df['credit_quality'].isnull().any():
        if 'credit_score' in df.columns:
            cq = df['credit_score'].apply(score_to_grade)
            if 'credit_quality' in df.columns:
                df['credit_quality'] = df['credit_quality'].fillna(cq)
            else:
                df['credit_quality'] = cq
        else:
            df['credit_quality'] = df.get('credit_quality', pd.Series(['BBB'] * len(df))).fillna('BBB')

    # ── Normalize employment_type ────────────────────────
    EMP_ALIASES = {
        'self_employed': 'Self-Employed', 'self employed': 'Self-Employed',
        'selfemployed': 'Self-Employed', 'self-employed': 'Self-Employed',
        'salaried': 'Salaried', 'salary': 'Salaried',
        'gig': 'Gig Worker', 'gig_worker': 'Gig Worker', 'freelancer': 'Gig Worker',
        'small_business': 'Small Business', 'small business': 'Small Business',
        'business': 'Small Business', 'sme': 'Small Business',
        'informal': 'Informal', 'daily_wage': 'Informal', 'daily wage': 'Informal',
    }
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].astype(str).str.strip().apply(
            lambda x: EMP_ALIASES.get(x.lower(), x)
        )

    # ── Normalize product_type ───────────────────────────
    PROD_ALIASES = {
        'personal loan': 'Personal Loan', 'personal_loan': 'Personal Loan', 'pl': 'Personal Loan',
        'sme working capital': 'SME Working Capital', 'sme_working_capital': 'SME Working Capital',
        'working capital': 'SME Working Capital', 'sme': 'SME Working Capital',
        'bnpl': 'BNPL', 'buy now pay later': 'BNPL',
        'unsecured consumer': 'Unsecured Consumer', 'consumer loan': 'Unsecured Consumer',
        'micro loan': 'Micro Loan', 'micro_loan': 'Micro Loan', 'microloan': 'Micro Loan',
    }
    if 'product_type' in df.columns:
        df['product_type'] = df['product_type'].astype(str).str.strip().apply(
            lambda x: PROD_ALIASES.get(x.lower(), x)
        )

    # ── Normalize geography ──────────────────────────────
    GEO_ALIASES = {
        'metro': 'Metro', 'urban': 'Metro', 'city': 'Metro',
        'tier-2': 'Tier-2', 'tier2': 'Tier-2', 'tier 2': 'Tier-2', 'tier_2': 'Tier-2',
        'semi-urban': 'Semi-Urban', 'semi_urban': 'Semi-Urban', 'semi urban': 'Semi-Urban',
        'rural': 'Rural', 'village': 'Rural',
    }
    if 'geography' in df.columns:
        df['geography'] = df['geography'].astype(str).str.strip().apply(
            lambda x: GEO_ALIASES.get(x.lower(), x)
        )

    # ── Normalize acquisition_channel ────────────────────
    CHAN_ALIASES = {
        'digital-direct': 'Digital-Direct', 'digital': 'Digital-Direct', 'online': 'Digital-Direct', 'app': 'Digital-Direct',
        'partner-fintech': 'Partner-Fintech', 'fintech': 'Partner-Fintech', 'partner': 'Partner-Fintech',
        'agent network': 'Agent Network', 'agent_network': 'Agent Network', 'agent': 'Agent Network',
        'nbfc-bc': 'NBFC-BC', 'nbfc': 'NBFC-BC', 'bc': 'NBFC-BC',
        'social media': 'Social Media', 'social_media': 'Social Media', 'social': 'Social Media',
    }
    if 'acquisition_channel' in df.columns:
        df['acquisition_channel'] = df['acquisition_channel'].astype(str).str.strip().apply(
            lambda x: CHAN_ALIASES.get(x.lower(), x)
        )

    # ── Parse default_indicator ──────────────────────────
    if 'default_indicator' in df.columns:
        di = df['default_indicator'].astype(str).str.lower().str.strip()
        df['default_indicator'] = di.isin(['1','yes','true','y','default','defaulted','npa']).astype(int)
    else:
        df['default_indicator'] = 0

    # ── Parse delinquency_status ─────────────────────────
    if 'delinquency_status' in df.columns:
        raw_ds = df['delinquency_status'].astype(str).str.lower().str.strip()
        def parse_delinq(row_default, raw):
            if row_default == 1:
                return 'Default'
            if any(k in raw for k in ['dpd30', 'dpd 30', 'dpd+', 'overdue', 'delinquent', 'past due', '30+']):
                return 'DPD30+'
            if 'current' in raw or raw == '0' or raw == 'nan':
                return 'Current'
            # Numeric DPD — treat >0 as overdue
            try:
                dpd_val = float(raw)
                return 'DPD30+' if dpd_val > 0 else 'Current'
            except ValueError:
                return 'Current'
        df['delinquency_status'] = [parse_delinq(d, r) for d, r in zip(df['default_indicator'], raw_ds)]
    else:
        df['delinquency_status'] = np.where(df['default_indicator']==1, 'Default', 'Current')

    df['loan_status'] = np.where(
        df['default_indicator']==1, 'Default',
        np.where(df['delinquency_status']=='DPD30+', 'At-Risk', 'Active')
    )

    # ── Fill missing numeric columns with sensible defaults ──
    defaults = {
        'loan_size':100000, 'credit_score':680, 'pricing_rate':15,
        'income_proxy':50000, 'tenure_months':12, 'ltv_ratio':0.5,
        'dti_ratio':0.3, 'cashflow_consistency':0.7, 'balance_volatility':0.3,
        'spending_shock':0, 'origination_month':0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    # ── Fill missing string columns ──────────────────────
    str_defaults = {
        'geography':'Metro','employment_type':'Salaried','product_type':'Personal Loan',
        'acquisition_channel':'Digital-Direct','credit_quality':'BBB',
    }
    for col, val in str_defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val).astype(str).str.strip()

    if 'loan_id' not in df.columns:
        df['loan_id'] = [f'LN-{i+1:06d}' for i in range(len(df))]

    if 'cohort' not in df.columns:
        df['cohort'] = df['origination_month'].apply(lambda m: f"C{int(m)//6+1}")

    if 'value_proxy' not in df.columns:
        df['value_proxy'] = (
            df['loan_size'] * df['pricing_rate'] / 100 *
            np.where(df['default_indicator']==1, -0.4, 0.8)
        ).round(2)

    return df
