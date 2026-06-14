"""
models.py — Machine Learning Models for LendIQ
- RandomForestClassifier : default prediction
- IsolationForest        : early warning anomaly detection
- KMeans                 : customer segmentation
- LogisticRegression     : risk grade classification
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

CAT_COLS = ['geography', 'employment_type', 'product_type', 'acquisition_channel', 'credit_quality']
NUM_COLS = ['credit_score', 'loan_size', 'pricing_rate', 'ltv_ratio', 'dti_ratio',
            'income_proxy', 'tenure_months', 'cashflow_consistency', 'balance_volatility', 'spending_shock']


class LendingMLModels:
    def __init__(self):
        self.encoders:   dict[str, LabelEncoder] = {}
        self.scaler      = StandardScaler()
        self.rf_default  = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
        self.gb_default  = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
        self.iso_forest  = IsolationForest(contamination=0.12, random_state=42)
        self.kmeans      = None   # created adaptively in fit()
        self.lr_grade    = LogisticRegression(max_iter=500, random_state=42)
        self.fitted      = False
        self.feature_names: list[str] = []
        self.grade_classes: list[str] = []

    # ── Feature Engineering ───────────────────────────────
    def _encode(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        parts = []
        for col in CAT_COLS:
            if col not in df.columns:
                parts.append(np.zeros((len(df), 1)))
                continue
            if fit or col not in self.encoders:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                known = set(le.classes_)
                safe  = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                encoded = le.transform(safe)
            parts.append(encoded.reshape(-1, 1))

        num = df[NUM_COLS].fillna(0).values
        X = np.hstack(parts + [num])

        if fit:
            X = self.scaler.fit_transform(X)
            self.feature_names = [f'cat_{c}' for c in CAT_COLS] + NUM_COLS
        else:
            X = self.scaler.transform(X)
        return X

    # ── Fit All Models ────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> dict:
        n = len(df)
        X = self._encode(df, fit=True)
        y_default = df['default_indicator'].values
        auc = 0.0

        # For very small datasets, cap estimators to avoid overfitting/errors
        n_estimators_rf = max(10, min(120, n * 5))
        n_estimators_gb = max(10, min(80, n * 3))

        # 1) Random Forest — Default Prediction
        try:
            self.rf_default = RandomForestClassifier(
                n_estimators=n_estimators_rf, max_depth=min(8, max(2, n//2)),
                random_state=42, n_jobs=-1,
                min_samples_split=max(2, min(5, n//3)),
                min_samples_leaf=max(1, min(3, n//5)),
            )
            self.rf_default.fit(X, y_default)
        except Exception as e:
            print(f'[RF] warning: {e}')

        # 2) Gradient Boosting — ensemble
        try:
            self.gb_default = GradientBoostingClassifier(
                n_estimators=n_estimators_gb, max_depth=min(4, max(2, n//3)),
                random_state=42,
                min_samples_split=max(2, min(5, n//3)),
            )
            self.gb_default.fit(X, y_default)
        except Exception as e:
            print(f'[GB] warning: {e}')

        # 3) Isolation Forest — Early Warning anomalies
        try:
            if n >= 5:
                contamination = min(0.5, max(0.05, float((y_default == 1).mean() or 0.12)))
                self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
                active_mask = df['loan_status'] == 'Active'
                X_fit = X[active_mask] if active_mask.sum() >= 3 else X
                self.iso_forest.fit(X_fit)
        except Exception as e:
            print(f'[IsoForest] warning: {e}')

        # 4) K-Means — adaptive clusters (never > n_samples)
        try:
            k = max(1, min(4, n))
            self.kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            self.kmeans.fit(X)
        except Exception as e:
            print(f'[KMeans] warning: {e}')
            self.kmeans = None

        # 5) Logistic Regression — Risk Grade (needs >= 2 classes)
        try:
            grades = df['credit_quality'].astype(str)
            unique_grades = grades.nunique()
            if unique_grades >= 2:
                le_g = LabelEncoder()
                y_grade = le_g.fit_transform(grades)
                self.encoders['_grade'] = le_g
                self.grade_classes = list(le_g.classes_)
                lr_c = max(0.01, min(1.0, float(n) / 100))
                self.lr_grade = LogisticRegression(max_iter=500, random_state=42, C=lr_c)
                self.lr_grade.fit(X, y_grade)
            else:
                print(f'[LR] only {unique_grades} grade class(es), skipping.')
        except Exception as e:
            print(f'[LR] warning: {e}')

        self.fitted = True

        # AUC metric — only meaningful if both classes present
        try:
            if len(set(y_default)) == 2:
                auc = roc_auc_score(y_default, self.rf_default.predict_proba(X)[:, 1])
            else:
                auc = 0.0
        except Exception:
            auc = 0.0

        fi = pd.Series(
            self.rf_default.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        return {
            'auc': round(float(auc), 4),
            'feature_importance': fi.head(10).to_dict(),
            'n_samples': n,
        }

    # ── Predict Default Probability ───────────────────────
    def predict_default(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(df))
        X = self._encode(df, fit=False)
        proba = self.rf_default.predict_proba(X)[:, 1]
        # Ensemble with GB
        try:
            proba_gb = self.gb_default.predict_proba(X)[:, 1]
            proba = (proba * 0.6 + proba_gb * 0.4)
        except Exception:
            pass
        return np.round(proba, 4)

    # ── Early Warning Score ───────────────────────────────
    def early_warning_score(self, df: pd.DataFrame) -> np.ndarray:
        """Composite early warning score 0-100 (higher = riskier)."""
        cf_score  = (1 - df['cashflow_consistency'].clip(0, 1)) * 30
        bal_score = df['balance_volatility'].clip(0, 1) * 25
        shock_score = df['spending_shock'].astype(float) * 20
        dti_score = df['dti_ratio'].clip(0, 0.8) / 0.8 * 25

        rule_score = (cf_score + bal_score + shock_score + dti_score).clip(0, 100)

        # Add ML default probability contribution
        if self.fitted:
            ml_prob = self.predict_default(df)
            score = rule_score * 0.6 + ml_prob * 100 * 0.4
        else:
            score = rule_score

        # Isolation forest anomaly bonus
        try:
            X = self._encode(df, fit=False)
            anomaly = self.iso_forest.decision_function(X)  # lower = more anomalous
            anomaly_norm = (-anomaly).clip(0, 1) * 10       # adds up to +10
            score = (score + anomaly_norm).clip(0, 100)
        except Exception:
            pass

        return np.round(score, 1)

    # ── Customer Segments ────────────────────────────────
    def cluster_labels(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.kmeans is None:
            return np.zeros(len(df), dtype=int)
        try:
            X = self._encode(df, fit=False)
            return self.kmeans.predict(X)
        except Exception:
            return np.zeros(len(df), dtype=int)

    # ── Feature Importance ───────────────────────────────
    def feature_importance(self) -> dict:
        if not self.fitted:
            return {}
        fi = pd.Series(
            self.rf_default.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False).head(10)
        return fi.round(4).to_dict()

    # ── Predict Single Loan ──────────────────────────────
    def predict_loan(self, loan_dict: dict) -> dict:
        row = pd.DataFrame([loan_dict])
        for col in NUM_COLS:
            if col not in row.columns:
                row[col] = 0
        for col in CAT_COLS:
            if col not in row.columns:
                row[col] = 'Unknown'
        prob = float(self.predict_default(row)[0])
        ew   = float(self.early_warning_score(row)[0])
        return {
            'default_probability': round(prob, 4),
            'early_warning_score': round(ew, 1),
            'risk_level': 'High' if ew > 60 else 'Medium' if ew > 40 else 'Low',
        }
