import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple, Union
from matplotlib.axes import Axes
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ['bytes', 'runes', 'words', 'lines']
TARGET = 'input_tokens'

# --- Shared defaults (can be overridden in notebooks) ---
SEED: int = 42
N_GRID_DEFAULT = [5, 10, 15, 20]
MAPE_THRESHOLD_DEFAULT = 50
DECISION_MARGIN_DEFAULT = 0.03
CV_SPLITS_A_DEFAULT = 5
CV_SPLITS_DEFAULT = 10
CV_REPEATS_DEFAULT = 5
BOOT_B_DEFAULT = 1000


def corr_heatmap(df: pd.DataFrame, cols: Sequence[str] = (FEATURES + [TARGET])) -> Axes:
    """Plot correlation matrix heatmap for quick multicollinearity sanity check."""
    cm = df[list(cols)].corr(numeric_only=True)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt='.2f',
                     cmap='vlag', square=True, cbar=True)
    plt.title('Correlation heatmap')
    plt.tight_layout()
    plt.show()
    return ax


def save_fig(ax_or_fig: Union[Axes, plt.Figure], name: str) -> Path:
    """Save a Matplotlib Axes or Figure to output/figures/{name}.png with tight layout.
    Returns the written path.
    """
    base = Path(__file__).resolve().parents[1]  # poc_token_approx/
    out_dir = base / 'reports' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    if isinstance(ax_or_fig, plt.Figure):
        fig = ax_or_fig
    else:
        fig = ax_or_fig.get_figure()
    if fig is not None:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    return out_path


def approach_b_ols(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Fit OLS with intercept on [bytes, runes, words, lines]; return test metrics and coefficients."""
    Xtr = train_df[FEATURES].to_numpy()
    ytr = train_df[TARGET].to_numpy()
    Xte = test_df[FEATURES].to_numpy()
    yte = test_df[TARGET].to_numpy()

    ols = LinearRegression(fit_intercept=True)
    ols.fit(Xtr, ytr)
    pte = ols.predict(Xte)
    mae = float(mean_absolute_error(yte, pte))
    bias = float(np.mean(yte - pte))
    r2 = float(r2_score(yte, pte))

    res = {
        'model': 'OLS',
        'test_mae': mae,
        'test_bias': bias,
        'test_r2': r2,
        'intercept': float(ols.intercept_),
        'coefs': dict(zip(FEATURES, map(float, ols.coef_))),
        'yte': yte,
        'pred': pte,
    }
    return res


def approach_b_ridge_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
    alphas: Optional[Sequence[float]] = None,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Ridge with StandardScaler + CV over alpha on the train set; return best alpha and test metrics."""
    if alphas is None:
        alphas = np.logspace(-3, 3, 13)

    X = train_df[FEATURES].to_numpy()
    y = train_df[TARGET].to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_alpha = None
    best_cv_mae = np.inf

    for a in alphas:
        fold_maes = []
        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=a, fit_intercept=True)),
            ])
            pipe.fit(Xtr, ytr)
            p = pipe.predict(Xva)
            fold_maes.append(mean_absolute_error(yva, p))
        cv_mae = float(np.mean(fold_maes))
        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_alpha = float(a)

    # Refit on full train with best alpha
    Xtr = X
    ytr = y
    Xte = test_df[FEATURES].to_numpy()
    yte = test_df[TARGET].to_numpy()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha, fit_intercept=True)),
    ])
    pipe.fit(Xtr, ytr)
    pte = pipe.predict(Xte)

    # Recover coefficients in input space: scaler handles standardization; coefficients are w on standardized features.
    scaler: StandardScaler = pipe.named_steps['scaler']
    ridge: Ridge = pipe.named_steps['ridge']

    # Convert standardized coefs to original feature scale: w_orig = w_std / scale_, intercept_orig = intercept_std - sum(w_orig * mean_)
    w_std = ridge.coef_.astype(float)
    scale = scaler.scale_.astype(float)
    mean = scaler.mean_.astype(float)
    w_orig = w_std / scale
    intercept_orig = float(ridge.intercept_ - np.sum(w_orig * mean))

    mae = float(mean_absolute_error(yte, pte))
    bias = float(np.mean(yte - pte))
    r2 = float(r2_score(yte, pte))

    res = {
        'model': 'Ridge',
        'alpha': best_alpha,
        'cv_mae': float(best_cv_mae),
        'test_mae': mae,
        'test_bias': bias,
        'test_r2': r2,
        'intercept': intercept_orig,
        'coefs': dict(zip(FEATURES, map(float, w_orig))),
        'yte': yte,
        'pred': pte,
    }
    return res


def approach_b_elasticnet_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
    l1_ratio_grid: Optional[Sequence[float]] = None,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Elastic Net with StandardScaler + CV over alpha and l1_ratio on the train set; return best params and test metrics.

    Uses sklearn's ElasticNetCV inside a Pipeline with StandardScaler. Coefficients are recovered to original feature scale.
    """
    if l1_ratio_grid is None:
        l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

    Xtr = train_df[FEATURES].to_numpy()
    ytr = train_df[TARGET].to_numpy()
    Xte = test_df[FEATURES].to_numpy()
    yte = test_df[TARGET].to_numpy()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNetCV(l1_ratio=l1_ratio_grid, alphas=None,
         cv=n_splits, random_state=seed, max_iter=100000)),
    ])
    pipe.fit(Xtr, ytr)
    pte = pipe.predict(Xte)

    # Recover original-scale coefficients
    scaler: StandardScaler = pipe.named_steps['scaler']
    enet: ElasticNetCV = pipe.named_steps['enet']
    w_std = enet.coef_.astype(float)
    scale = scaler.scale_.astype(float)
    mean = scaler.mean_.astype(float)
    w_orig = w_std / scale
    intercept_orig = float(enet.intercept_ - np.sum(w_orig * mean))

    mae = float(mean_absolute_error(yte, pte))
    bias = float(np.mean(yte - pte))
    r2 = float(r2_score(yte, pte))

    res = {
        'model': 'ElasticNetCV',
        'alpha': float(enet.alpha_),
        'l1_ratio': float(enet.l1_ratio_),
        'cv_mae': None,
        'test_mae': mae,
        'test_bias': bias,
        'test_r2': r2,
        'intercept': intercept_orig,
        'coefs': dict(zip(FEATURES, map(float, w_orig))),
        'yte': yte,
        'pred': pte,
    }
    return res


def summarize_and_decide(
    res_a: Dict[str, Any],
    res_b_ols: Optional[Dict[str, Any]] = None,
    res_b_ridge: Optional[Dict[str, Any]] = None,
    res_b_enet: Optional[Dict[str, Any]] = None,
    prefer_margin: float = 0.03,
) -> Dict[str, Any]:
    """Compare Approach A test MAE vs best of B (OLS/Ridge/ElasticNet) and recommend the simpler model unless B beats A by >≈margin.
    Returns a dict with the decision and metrics.
    """
    a_mae = float(res_a.get('test_mae'))
    a_bias = float(res_a.get('test_bias'))
    yte_a = res_a.get('yte')
    mean_y = float(np.mean(yte_a)) if yte_a is not None else np.nan
    bias_pct = float(
        abs(a_bias) / mean_y) if mean_y and not np.isnan(mean_y) else np.nan

    candidates = []
    if res_b_ols is not None:
        candidates.append(res_b_ols)
    if res_b_ridge is not None:
        candidates.append(res_b_ridge)
    if res_b_enet is not None:
        candidates.append(res_b_enet)

    best_b = None
    if candidates:
        best_b = min(candidates, key=lambda r: r['test_mae'])

    decision = 'A'  # default to simpler
    reason = 'Approach A selected by default.'

    if best_b is not None:
        b_mae = float(best_b['test_mae'])
        rel_improve = (a_mae - b_mae) / a_mae
        if rel_improve > prefer_margin:
            decision = 'B'
            reason = f"Approach B ({best_b['model']}) beats A by {rel_improve*100:.1f}% MAE (> {prefer_margin*100:.0f}% margin)."
        else:
            reason = f"Approach A retained; B improves MAE by {rel_improve*100:.1f}% (≤ {prefer_margin*100:.0f}% margin)."

    # Bias threshold note per plan (2–3% of mean(y))
    bias_flag = bool(not np.isnan(bias_pct) and bias_pct >
                     0.03 and not bool(res_a.get('fit_intercept', True)))

    summary = {
        'decision': decision,
        'reason': reason,
        'A_test_mae': a_mae,
        'A_test_bias': a_bias,
        'A_bias_pct_of_mean_y': bias_pct,
        'A_bias_flag_gt3pct_without_intercept': bias_flag,
        'B_best_model': None if best_b is None else best_b['model'],
        'B_test_mae': None if best_b is None else float(best_b['test_mae']),
        'B_test_bias': None if best_b is None else float(best_b['test_bias']),
        'B_test_r2': None if best_b is None else float(best_b['test_r2']),
        'B_details': best_b,
    }
    return summary


# --- Additions: exporters, bootstrap CI, and coef sanity checks ---

def export_linear_multifeature(
    intercept: float,
    coefs: Dict[str, float],
    model: str,
    scope: str,
    out_path: Optional[Path] = None,
    estimator: Optional[str] = None,
    feature_set: Sequence[str] = tuple(FEATURES),
) -> Path:
    """Export a multivariate linear model to JSON in a single standardized file name.
    Payload: {type: 'linear', w0, w_bytes, ...}. Returns the path written.
    """
    import json
    from pathlib import Path

    payload: Dict[str, Any] = {
        'type': 'linear',
        'w0': float(intercept),
        'model': model,
        'scope': scope,
    }
    if estimator is not None:
        payload['estimator'] = estimator
    if feature_set is not None:
        payload['feature_set'] = list(feature_set)

    # Ensure keys ordered and deterministic
    for k in FEATURES:
        if k in coefs:
            payload[f'w_{k}'] = float(coefs[k])

    if out_path is None:
        base = Path(__file__).resolve().parents[1]  # poc_token_approx/
        # Standardize to the same filename used by single-feature export
        out = base / 'models' / 'model_coefs.json'
    else:
        out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f, separators=(',', ':'), ensure_ascii=False)
    return out


def bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Nonparametric bootstrap CI for MAE on paired (y_true, y_pred)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        maes.append(float(np.mean(np.abs(y_true[idx] - y_pred[idx]))))
    lo = float(np.quantile(maes, alpha / 2))
    hi = float(np.quantile(maes, 1 - alpha / 2))
    return lo, hi


# --- Learning curve utilities ---

def repeated_cv_mae(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_splits: int = 10,
    n_repeats: int = 5,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run repeated K-fold CV, return (MAE, y_true_all, y_pred_all)."""
    rkf = RepeatedKFold(n_splits=n_splits,
                        n_repeats=n_repeats, random_state=seed)
    y_true_all = []
    y_pred_all = []
    for tr_idx, te_idx in rkf.split(X):
        m = model_factory()
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict(X[te_idx])
        y_true_all.append(y[te_idx])
        y_pred_all.append(p)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    return mae, y_true_all, y_pred_all


def learning_curve(
    df: pd.DataFrame,
    n_grid: Sequence[int],
    feature_A: Sequence[str] = ('bytes',),
    feature_B: Sequence[str] = tuple(FEATURES),
    seed: int = 42,
    cv_splits: int = 10,
    cv_repeats: int = 5,
    boot_B: int = 500,
    fit_intercept_A: bool = True,
) -> pd.DataFrame:
    """Compute learning curve for Approach A and B.

    - Approach A: LinearRegression on feature_A with configurable intercept (fit_intercept_A).
    - Approach B: OLS (LinearRegression with intercept) on feature_B.

    Returns a tidy DataFrame with n, A_mae/A_lo/A_hi, B_mae/B_lo/B_hi, and relative CI half-widths.
    """
    N = len(df)
    n_grid = [int(n) for n in n_grid if n <= N]
    if not n_grid:
        print({'learning_curves': 'skipped', 'reason': 'n_grid > len(df)'})
        return pd.DataFrame([])

    mean_y = float(df[TARGET].mean())
    rows = []
    for n in n_grid:
        sub = df.sample(n=n, random_state=seed)
        y = sub[TARGET].to_numpy()

        # A: selected single feature + chosen intercept
        XA = sub[list(feature_A)].to_numpy()
        maeA, ytA, ypA = repeated_cv_mae(XA, y, model_factory=lambda: LinearRegression(fit_intercept=fit_intercept_A),
                                         n_splits=cv_splits, n_repeats=cv_repeats, seed=seed)
        loA, hiA = bootstrap_mae_ci(ytA, ypA, B=boot_B, seed=seed)

        # B: OLS on provided features
        XB = sub[list(feature_B)].to_numpy()
        maeB, ytB, ypB = repeated_cv_mae(XB, y, model_factory=lambda: LinearRegression(fit_intercept=True),
                                         n_splits=cv_splits, n_repeats=cv_repeats, seed=seed)
        loB, hiB = bootstrap_mae_ci(ytB, ypB, B=boot_B, seed=seed)

        rows.append({
            'n': int(n),
            'A_mae': float(maeA), 'A_lo': float(loA), 'A_hi': float(hiA),
            'B_mae': float(maeB), 'B_lo': float(loB), 'B_hi': float(hiB),
            'A_rel_hw': float(((hiA - loA) / 2) / mean_y),
            'B_rel_hw': float(((hiB - loB) / 2) / mean_y),
        })

    return pd.DataFrame(rows)


def plot_learning_curve(lc_df: pd.DataFrame) -> Axes:
    """Plot learning curve MAE with bootstrap CIs for A and B."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    xs = lc_df['n'].to_numpy()
    for prefix, label in [('A', 'A (bytes + intercept)'), ('B', 'B (OLS)')]:
        y = lc_df[f'{prefix}_mae'].to_numpy()
        lo = lc_df[f'{prefix}_lo'].to_numpy()
        hi = lc_df[f'{prefix}_hi'].to_numpy()
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(xs, y, yerr=yerr, fmt='-o', label=label)
    ax.set_xlabel('n (samples)')
    ax.set_ylabel('MAE (tokens)')
    ax.set_title('Learning curves')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax


def coef_sanity_checks(coefs: Dict[str, float]) -> Dict[str, Any]:
    """Basic sanity checks for linear model coefficients.
    Flags negative weights for bytes/runes and reports magnitude hints.
    """
    flags: Dict[str, Any] = {
        'negative_bytes': False,
        'negative_runes': False,
        'notes': [],
    }
    w_bytes = coefs.get('bytes')
    w_runes = coefs.get('runes')
    if w_bytes is not None and w_bytes < 0:
        flags['negative_bytes'] = True
        flags['notes'].append('bytes coef negative')
    if w_runes is not None and w_runes < 0:
        flags['negative_runes'] = True
        flags['notes'].append('runes coef negative')
    return flags


# --- Additional utilities extracted from notebook ---

def load_records(records_path: Path) -> pd.DataFrame:
    """Load records.jsonl and normalize nested 'features' into flat columns.

    Expects each line to be a JSON object with a 'features' field containing
    bytes/runes/words/lines. Returns a DataFrame with required analysis columns.
    Raises FileNotFoundError or ValueError on missing input or malformed schema.
    """
    import json
    if not Path(records_path).exists():
        raise FileNotFoundError(f"records file not found: {records_path}")
    rows = []
    with open(records_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    raw = pd.DataFrame(rows)
    if 'features' not in raw.columns:
        raise ValueError("records missing 'features' column")
    feat = pd.json_normalize(raw['features'])
    required_nested = ['bytes', 'runes', 'words', 'lines']
    missing_nested = [k for k in required_nested if k not in feat.columns]
    if missing_nested:
        raise ValueError(f"features missing required keys: {missing_nested}")
    df = pd.concat([raw.drop(columns=['features']), feat], axis=1)
    cols = ['bytes', 'runes', 'words', 'lines',
            'input_tokens', 'model', 'source_path']
    return df[cols]


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean records DataFrame in-place style.

    - Ensures required columns exist.
    - Drops rows with missing target and non-numeric feature values.
    - Attaches a concise summary under df.attrs['validation_info'].
    """
    required = ['bytes', 'runes', 'words', 'lines', 'input_tokens']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"records missing required columns: {missing_cols}")
    total_before = len(df)
    info = {'total_before': total_before, 'drops': {}, 'total_after': None}
    # Drop rows missing target
    before = len(df)
    df = df.dropna(subset=['input_tokens'])
    info['drops']['input_tokens_na'] = before - len(df)
    # Ensure numeric features
    for c in ['bytes', 'runes', 'words', 'lines']:
        before_c = len(df)
        df = df[pd.to_numeric(df[c], errors='coerce').notna()]
        info['drops'][f'{c}_non_numeric_or_na'] = before_c - len(df)
    info['total_after'] = len(df)
    # Attach concise validation info without printing
    df.attrs['validation_info'] = info
    return df


def cv_single_feature(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[float, float]:
    """Cross-validate a single-feature LinearRegression.

    Returns (mean_MAE, mean_bias) across folds, where bias = mean(y - yhat).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes = []
    biases = []
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        lr = LinearRegression(fit_intercept=fit_intercept)
        lr.fit(Xtr, ytr)
        p = lr.predict(Xva)
        maes.append(mean_absolute_error(yva, p))
        biases.append(float(np.mean(yva - p)))
    return float(np.mean(maes)), float(np.mean(biases))


def select_best(cv_df: pd.DataFrame, prefer_margin: float = 0.03) -> pd.Series:
    """Pick best row by CV MAE, preferring b=0 if within margin for same feature."""
    best_row = cv_df.sort_values('cv_mae', ascending=True).iloc[0]
    if best_row['fit_intercept'] and ((cv_df['feature'] == best_row['feature']) & (~cv_df['fit_intercept'])).any():
        zero_row = cv_df[(cv_df['feature'] == best_row['feature'])
                         & (~cv_df['fit_intercept'])].iloc[0]
        if (zero_row['cv_mae'] - best_row['cv_mae']) / best_row['cv_mae'] <= prefer_margin:
            return zero_row
    return best_row


def fit_single_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature: str,
    fit_intercept: bool,
) -> Dict[str, Any]:
    """Fit/test LinearRegression on one feature; return metrics and params."""
    Xtr = train_df[[feature]].to_numpy()
    ytr = train_df[TARGET].to_numpy()
    Xte = test_df[[feature]].to_numpy()
    yte = test_df[TARGET].to_numpy()
    lr = LinearRegression(fit_intercept=fit_intercept)
    lr.fit(Xtr, ytr)
    pte = lr.predict(Xte)
    return {
        'feature': feature,
        'fit_intercept': bool(fit_intercept),
        'coef_a': float(lr.coef_[0]),
        'intercept_b': float(lr.intercept_) if fit_intercept else 0.0,
        'test_mae': float(mean_absolute_error(yte, pte)),
        'test_bias': float(np.mean(yte - pte)),
        'pred': pte,
        'yte': yte,
    }


def fit_eval(cols: Sequence[str], train_df: pd.DataFrame, test_df: pd.DataFrame, ytr: np.ndarray, yte: np.ndarray) -> Dict[str, Any]:
    """Fit OLS with intercept on selected columns and evaluate on test set."""
    Xtr = train_df[list(cols)].to_numpy()
    Xte = test_df[list(cols)].to_numpy()
    lr = LinearRegression(fit_intercept=True)
    lr.fit(Xtr, ytr)
    p = lr.predict(Xte)
    mae = mean_absolute_error(yte, p)
    bias = float(np.mean(yte - p))
    return {
        'cols': list(cols),
        'mae': float(mae),
        'bias': bias,
        'intercept': float(lr.intercept_),
        'coefs': dict(zip(cols, map(float, lr.coef_))),
    }


def ols_ablations(train_df: pd.DataFrame, test_df: pd.DataFrame, candidates: Sequence[Sequence[str]]) -> Sequence[Dict[str, Any]]:
    """Run OLS fit_eval over candidate column sets; return list of result dicts."""
    ytr = train_df[TARGET].to_numpy()
    yte = test_df[TARGET].to_numpy()
    return [fit_eval(cols, train_df, test_df, ytr, yte) for cols in candidates]


def plot_residuals_vs(yhat: np.ndarray, resid: np.ndarray, x_series: Optional[pd.Series] = None, x_label: str = '') -> Union[Axes, np.ndarray]:
    """Plot residuals vs predictions, and optionally residuals vs an x_series."""
    if x_series is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(yhat, resid)
        ax.axhline(0, color='k', lw=1)
        ax.set_xlabel('ŷ')
        ax.set_ylabel('y-ŷ')
        ax.set_title('Residuals vs Predictions')
        plt.tight_layout()
        plt.show()
        return ax
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].scatter(yhat, resid)
        axes[0].axhline(0, color='k', lw=1)
        axes[0].set_title('Residuals vs Predictions')
        axes[0].set_xlabel('ŷ')
        axes[0].set_ylabel('y-ŷ')
        axes[1].scatter(x_series, resid)
        axes[1].axhline(0, color='k', lw=1)
        axes[1].set_title(f'Residuals vs {x_label}')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('y-ŷ')
        plt.tight_layout()
        plt.show()
        return axes


def export_single_feature(
    feature_type: str,
    a: float,
    b: float = 0.0,
    model: str = 'claude-3-7-sonnet-latest',
    scope: str = 'en-long-one-turn',
    out_path: Optional[Path] = None,
    estimator: Optional[str] = 'LinearRegression',
    fit_intercept: Optional[bool] = True,
) -> Path:
    """Export a single-feature linear model to JSON at output/model_coefs.json.

    Payload keys: type (feature name), a, b, model, scope. Returns path written.
    """
    import json
    if out_path is None:
        out_path = Path(__file__).resolve(
        ).parents[1] / 'models' / 'model_coefs.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        'type': feature_type,
        'a': float(a),
        'b': float(b),
        'model': model,
        'scope': scope,
    }
    if estimator is not None:
        payload['estimator'] = estimator
    if fit_intercept is not None:
        payload['fit_intercept'] = bool(fit_intercept)
    with open(out_path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'), ensure_ascii=False)
    return out_path
