"""
CASP: Covariance-Aware Simplex Projection for Portfolio Optimization

Implements the CASP repair operator and runs experiments on S&P 500 data.
Includes ablation study, out-of-sample validation, and statistical testing.
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.stats import wilcoxon, ttest_rel, spearmanr
from scipy.special import gamma
import matplotlib.pyplot as plt
import warnings
import time
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# --- Configuration ---

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    n_assets: int = 100
    cardinality_K: int = 15
    weight_lb: float = 0.02
    weight_ub: float = 0.15
    risk_free_rate: float = 0.045
    transaction_cost_bps: float = 10  # basis points
    # RA-CASP parameters
    lambda_ret: float = 0.8
    gamma_ret: float = 0.25
    # Optimization parameters
    mogwo_pop_size: int = 50
    mogwo_max_iter: int = 100
    mogwo_archive_size: int = 30
    # Experiment parameters
    n_direct_samples: int = 500
    n_optimization_runs: int = 15
    random_seed: int = 42


# --- Results Export ---

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def summarize_ablation(results: Dict, baseline: str = 'Euclidean') -> pd.DataFrame:
    """Create a compact ablation summary table."""
    methods = [k for k in results.keys() if isinstance(results.get(k), dict) and 'vars' in results[k]]
    rows = []
    base_var = float(np.mean(results[baseline]['vars']))
    base_sh = float(np.mean(results[baseline]['sharpes']))
    for name in methods:
        mv = float(np.mean(results[name]['vars']))
        ms = float(np.mean(results[name]['sharpes']))
        rows.append({
            'method': name,
            'mean_variance': mv,
            'std_variance': float(np.std(results[name]['vars'])),
            'mean_sharpe': ms,
            'std_sharpe': float(np.std(results[name]['sharpes'])),
            'var_reduction_pct_vs_' + baseline: (base_var - mv) / base_var * 100.0,
            'sharpe_improvement_pct_vs_' + baseline: (ms - base_sh) / (abs(base_sh) + 1e-12) * 100.0,
            'n': int(len(results[name]['vars'])),
        })
    df = pd.DataFrame(rows)
    # Preferred ordering for paper readability
    preferred = [
        'Euclidean', 'VolNorm+Euc', 'MinVar+Euc', 'Sharpe+Euc',
        'CASP-Basic', 'CASP-RetSel', 'RA-CASP'
    ]
    if set(preferred).issubset(set(df['method'])):
        df['__order'] = df['method'].apply(lambda m: preferred.index(m))
        df = df.sort_values('__order').drop(columns='__order')
    else:
        df = df.sort_values('method')
    return df.reset_index(drop=True)


def export_ablation_results(results: Dict, output_dir: str, baseline: str = 'Euclidean') -> None:
    """Write ablation summary + raw distributions for plotting."""
    _ensure_dir(output_dir)
    summary = summarize_ablation(results, baseline=baseline)
    summary.to_csv(os.path.join(output_dir, 'ablation_summary.csv'), index=False)

    # Save raw values as CSV (human-readable; avoids binary artifact files)
    raw_rows = []
    for method in summary['method'].tolist():
        vars_arr = np.asarray(results[method]['vars'])
        sharpes_arr = np.asarray(results[method]['sharpes'])
        rets_arr = np.asarray(results[method].get('rets', []))
        has_rets = len(rets_arr) == len(vars_arr) and len(vars_arr) > 0

        for i in range(len(vars_arr)):
            row = {
                'method': method,
                'sample_idx': i,
                'variance': float(vars_arr[i]),
                'sharpe': float(sharpes_arr[i]),
            }
            if has_rets:
                row['return'] = float(rets_arr[i])
            raw_rows.append(row)

    pd.DataFrame(raw_rows).to_csv(os.path.join(output_dir, 'ablation_raw.csv'), index=False)


def summarize_oos(results: Dict) -> pd.DataFrame:
    """Create a compact out-of-sample summary table."""
    rows = []
    for method, d in results.items():
        ins = np.asarray(d['in_sample'])
        outs = np.asarray(d['out_sample'])
        corr, _ = spearmanr(ins, outs)
        rows.append({
            'method': method,
            'mean_in_sample_sharpe': float(np.mean(ins)),
            'std_in_sample_sharpe': float(np.std(ins)),
            'mean_out_sample_sharpe': float(np.mean(outs)),
            'std_out_sample_sharpe': float(np.std(outs)),
            'spearman_corr_in_vs_out': float(corr),
            'n': int(len(ins)),
        })
    df = pd.DataFrame(rows)
    preferred = ['Euclidean', 'VolNorm+Euc', 'CASP-Basic', 'RA-CASP', 'Sharpe+Euc']
    if set(preferred).issubset(set(df['method'])):
        df['__order'] = df['method'].apply(lambda m: preferred.index(m))
        df = df.sort_values('__order').drop(columns='__order')
    else:
        df = df.sort_values('method')
    return df.reset_index(drop=True)


def export_oos_results(results: Dict, output_dir: str) -> None:
    """Write OOS summary + raw in/out pairs for plotting."""
    _ensure_dir(output_dir)
    summary = summarize_oos(results)
    summary.to_csv(os.path.join(output_dir, 'oos_summary.csv'), index=False)

    raw_rows = []
    for method in summary['method'].tolist():
        ins = np.asarray(results[method]['in_sample'])
        outs = np.asarray(results[method]['out_sample'])
        for i in range(len(ins)):
            raw_rows.append({
                'method': method,
                'sample_idx': i,
                'in_sample_sharpe': float(ins[i]),
                'out_sample_sharpe': float(outs[i]),
            })

    pd.DataFrame(raw_rows).to_csv(os.path.join(output_dir, 'oos_raw.csv'), index=False)


# --- Data Loading and Processing ---

def load_sp500_data(filepath: str) -> pd.DataFrame:
    """Load S&P 500 price data from CSV file."""
    print("Loading S&P 500 price data...")
    
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(data)}")
    print(f"  Assets: {len(data.columns)}")
    
    data = data.dropna(axis=1)
    print(f"  Assets after cleaning: {len(data.columns)}")
    
    return data


def split_data_temporal(prices: pd.DataFrame, 
                        train_end: str = '2023-12-31') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally for out-of-sample testing."""
    train_data = prices[prices.index <= train_end]
    test_data = prices[prices.index > train_end]
    
    print(f"\nTemporal Split:")
    print(f"  Training: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} days)")
    print(f"  Testing:  {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data)} days)")
    
    return train_data, test_data


def compute_financial_metrics(prices: pd.DataFrame, 
                               shrinkage: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute expected returns and covariance matrix with Ledoit-Wolf shrinkage."""
    returns = np.log(prices / prices.shift(1)).dropna()
    
    mu = returns.mean().values * 252
    sample_cov = returns.cov().values * 252
    n_samples, n_assets = returns.shape
    
    # Ledoit-Wolf shrinkage
    trace = np.trace(sample_cov)
    mu_target = trace / n_assets
    delta = np.eye(n_assets) * mu_target
    
    Omega = (1 - shrinkage) * sample_cov + shrinkage * delta
    
    min_eig = np.min(np.linalg.eigvalsh(Omega))
    if min_eig < 1e-8:
        Omega += (1e-8 - min_eig + 1e-6) * np.eye(n_assets)
    
    print(f"\nFinancial Statistics:")
    print(f"  Annualized return range: [{mu.min()*100:.1f}%, {mu.max()*100:.1f}%]")
    print(f"  Annualized volatility range: [{np.sqrt(np.diag(Omega)).min()*100:.1f}%, {np.sqrt(np.diag(Omega)).max()*100:.1f}%]")
    print(f"  Covariance shrinkage: {shrinkage:.2f}")
    print(f"  Condition number: {np.linalg.cond(Omega):.1f}")
    
    return mu, Omega, list(returns.columns)


def compute_out_of_sample_metrics(weights: np.ndarray, 
                                   test_prices: pd.DataFrame,
                                   rf: float = 0.045) -> Dict[str, float]:
    """Compute realized out-of-sample performance metrics."""
    test_returns = np.log(test_prices / test_prices.shift(1)).dropna()
    
    # Portfolio daily returns
    portfolio_returns = test_returns.values @ weights
    
    # Annualized metrics
    realized_return = np.mean(portfolio_returns) * 252
    realized_vol = np.std(portfolio_returns) * np.sqrt(252)
    realized_sharpe = (realized_return - rf) / (realized_vol + 1e-10)
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = np.max(drawdown)
    
    return {
        'realized_return': realized_return,
        'realized_volatility': realized_vol,
        'realized_sharpe': realized_sharpe,
        'max_drawdown': max_drawdown,
        'n_days': len(portfolio_returns)
    }


def load_real_esg_scores(tickers: List[str], esg_file: str) -> Optional[np.ndarray]:
    """
    Load real ESG scores from file if available.
    Returns None if file doesn't exist or data is incomplete.
    """
    try:
        esg_df = pd.read_csv(esg_file, index_col=0)
        
        # Check if we have enough coverage
        available = [t for t in tickers if t in esg_df.index]
        coverage = len(available) / len(tickers)
        
        if coverage < 0.5:
            print(f"  Real ESG data coverage too low ({coverage*100:.1f}%), using synthetic")
            return None
        
        # Build ESG array, using mean for missing values
        esg_scores = np.zeros(len(tickers))
        mean_esg = esg_df['ESG_Score'].mean()
        
        for i, ticker in enumerate(tickers):
            if ticker in esg_df.index:
                esg_scores[i] = esg_df.loc[ticker, 'ESG_Score']
            else:
                esg_scores[i] = mean_esg
        
        print(f"  Loaded real ESG scores from {esg_file}")
        print(f"  Coverage: {len(available)}/{len(tickers)} ({coverage*100:.1f}%)")
        print(f"  ESG score range: [{esg_scores.min():.1f}, {esg_scores.max():.1f}]")
        
        return esg_scores
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  Error loading ESG data: {e}")
        return None


def generate_esg_scores(tickers: List[str], mu: np.ndarray, 
                         esg_file: str = None) -> np.ndarray:
    """
    Get ESG scores - first try to load real data, fall back to synthetic.
    """
    # Try to load real ESG data first
    if esg_file:
        real_esg = load_real_esg_scores(tickers, esg_file)
        if real_esg is not None:
            return real_esg
    
    # Fall back to synthetic ESG scores based on sector patterns
    print("  Generating synthetic ESG scores based on sector patterns...")
    np.random.seed(123)
    base_esg = 65 + np.random.randn(len(tickers)) * 12
    
    # Sector adjustments (simplified)
    energy_tickers = ['XOM', 'CVX', 'SLB', 'EOG', 'OXY', 'COP', 'VLO', 'MPC', 'PSX']
    healthcare_tickers = ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'AMGN', 'GILD', 'VRTX', 'REGN', 'BMY']
    tech_tickers = ['MSFT', 'AAPL', 'GOOGL', 'CRM', 'NOW', 'ADBE', 'NVDA', 'AMD', 'INTC']
    tobacco_tickers = ['MO', 'PM']
    defense_tickers = ['BA', 'RTX', 'LMT', 'NOC', 'GD']
    utilities_tickers = ['NEE', 'DUK', 'SO', 'AEP', 'D']
    
    for i, ticker in enumerate(tickers):
        if ticker in energy_tickers:
            base_esg[i] -= 15
        elif ticker in healthcare_tickers:
            base_esg[i] += 8
        elif ticker in tech_tickers:
            base_esg[i] += 5
        elif ticker in tobacco_tickers:
            base_esg[i] -= 20
        elif ticker in defense_tickers:
            base_esg[i] -= 10
        elif ticker in utilities_tickers:
            base_esg[i] += 10
    
    esg = np.clip(base_esg, 25, 95)
    print(f"  ESG score range: [{esg.min():.0f}, {esg.max():.0f}]")
    print(f"  Note: ESG scores are synthetic (sector-based estimates)")
    
    return esg


# --- Projection Operators ---

def select_assets_by_magnitude(z: np.ndarray, K: int) -> np.ndarray:
    """Baseline: Select K assets with largest |z_i| (standard approach)."""
    return np.argpartition(np.abs(z), -K)[-K:]


def select_assets_volatility_normalized(z: np.ndarray, Omega: np.ndarray, K: int) -> np.ndarray:
    """Select K assets with largest |z_i| / sqrt(Σ_ii) (volatility-normalized)."""
    vol = np.sqrt(np.diag(Omega) + 1e-10)
    scores = np.abs(z) / vol
    return np.argpartition(scores, -K)[-K:]


def select_assets_return_aware(z: np.ndarray, Omega: np.ndarray, mu: np.ndarray, 
                                K: int, lambda_ret: float = 0.8) -> np.ndarray:
    """Select K assets with return-boosted volatility-normalized scores."""
    vol = np.sqrt(np.diag(Omega) + 1e-10)
    
    # Normalize returns to [0, 1]
    mu_min, mu_max = mu.min(), mu.max()
    mu_norm = (mu - mu_min) / (mu_max - mu_min + 1e-10)
    
    # Score = |z_i| * (1 + λ * μ_norm) / σ_i
    return_boost = 1.0 + lambda_ret * mu_norm
    scores = np.abs(z) * return_boost / vol
    
    return np.argpartition(scores, -K)[-K:]


def select_assets_min_variance(z: np.ndarray, Omega: np.ndarray, K: int) -> np.ndarray:
    """Select K assets with lowest individual variance (diversification-focused)."""
    variances = np.diag(Omega)
    # Among positive z values, prefer low variance
    scores = np.abs(z) / (variances + 1e-10)
    return np.argpartition(scores, -K)[-K:]


def select_assets_sharpe_based(z: np.ndarray, Omega: np.ndarray, mu: np.ndarray, 
                                K: int, rf: float = 0.045) -> np.ndarray:
    """Select K assets with highest individual Sharpe ratios weighted by |z|."""
    vol = np.sqrt(np.diag(Omega) + 1e-10)
    individual_sharpe = (mu - rf) / vol
    # Combine with signal strength
    scores = np.abs(z) * np.maximum(individual_sharpe, 0.01)
    return np.argpartition(scores, -K)[-K:]


def project_euclidean(z_sel: np.ndarray, K: int, lb: float, ub: float) -> np.ndarray:
    """Euclidean projection onto constrained simplex."""
    z_clip = np.clip(z_sel, lb, ub)
    
    tau_lo = np.min(z_clip) - ub - 1
    tau_hi = np.max(z_clip) - lb + 1
    
    for _ in range(60):
        tau = (tau_lo + tau_hi) / 2
        w = np.clip(z_sel - tau, lb, ub)
        s = np.sum(w)
        if abs(s - 1.0) < 1e-10:
            break
        elif s > 1.0:
            tau_lo = tau
        else:
            tau_hi = tau
    
    w = np.clip(z_sel - tau, lb, ub)
    w = w / w.sum()  # Ensure exact sum = 1
    return w


def volnorm_selection_euclidean_projection(z: np.ndarray, Omega: np.ndarray,
                                           K: int, lb: float, ub: float,
                                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Volatility-normalized selection + Euclidean projection.
    This baseline isolates the effect of CASP's Stage-1 selection rule
    from the covariance-aware (Σ-metric) projection geometry.
    """
    n = len(z)
    indices = select_assets_volatility_normalized(z, Omega, K)
    z_sel = z[indices]
    w_sel = project_euclidean(z_sel, K, lb, ub)

    w_full = np.zeros(n)
    w_full[indices] = w_sel
    return w_full, indices


def project_mahalanobis(z_sel: np.ndarray, Omega_sel: np.ndarray, 
                         K: int, lb: float, ub: float) -> np.ndarray:
    """
    Σ-metric projection onto the constrained simplex (no return term).

    Note: We intentionally use the covariance-induced quadratic form
    (w - z)^T Σ (w - z), which equals tracking-error variance, rather than the
    classical statistical Mahalanobis distance that uses Σ^{-1}.
    """
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(Omega_sel))
    if min_eig < 1e-8:
        Omega_sel = Omega_sel + (1e-8 - min_eig) * np.eye(K)
    
    def objective(w):
        diff = w - z_sel
        return 0.5 * diff @ Omega_sel @ diff
    
    def gradient(w):
        return Omega_sel @ (w - z_sel)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0,
                   'jac': lambda w: np.ones(K)}
    bounds = [(lb, ub) for _ in range(K)]
    w0 = np.ones(K) / K
    
    result = minimize(objective, w0, method='SLSQP', jac=gradient,
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-12, 'maxiter': 200})
    
    return result.x


def project_mahalanobis_return_regularized(z_sel: np.ndarray, Omega_sel: np.ndarray,
                                            mu_sel: np.ndarray, K: int, 
                                            lb: float, ub: float,
                                            gamma_ret: float = 0.25) -> np.ndarray:
    """
    Σ-metric projection with a linear return regularizer.

    See `project_mahalanobis` docstring for the Σ (vs Σ^{-1}) clarification.
    """
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(Omega_sel))
    if min_eig < 1e-8:
        Omega_sel = Omega_sel + (1e-8 - min_eig) * np.eye(K)
    
    # Normalize returns for stability
    mu_norm = mu_sel / (np.abs(mu_sel).max() + 1e-10)
    
    def objective(w):
        diff = w - z_sel
        variance_term = 0.5 * diff @ Omega_sel @ diff
        return_term = gamma_ret * mu_norm @ w
        return variance_term - return_term
    
    def gradient(w):
        return Omega_sel @ (w - z_sel) - gamma_ret * mu_norm
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0,
                   'jac': lambda w: np.ones(K)}
    bounds = [(lb, ub) for _ in range(K)]
    w0 = np.ones(K) / K
    
    result = minimize(objective, w0, method='SLSQP', jac=gradient,
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-12, 'maxiter': 200})
    
    return result.x


# --- Composite Projection Operators ---

def euclidean_projection(z: np.ndarray, K: int, lb: float, ub: float,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Euclidean projection (baseline)."""
    n = len(z)
    indices = select_assets_by_magnitude(z, K)
    z_sel = z[indices]
    w_sel = project_euclidean(z_sel, K, lb, ub)
    
    w_full = np.zeros(n)
    w_full[indices] = w_sel
    return w_full, indices


def casp_basic_projection(z: np.ndarray, Omega: np.ndarray, K: int, 
                           lb: float, ub: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic CASP: Volatility-normalized selection + Mahalanobis projection.
    No return terms - pure variance focus.
    """
    n = len(z)
    indices = select_assets_volatility_normalized(z, Omega, K)
    z_sel = z[indices]
    Omega_sel = Omega[np.ix_(indices, indices)]
    
    w_sel = project_mahalanobis(z_sel, Omega_sel, K, lb, ub)
    
    w_full = np.zeros(n)
    w_full[indices] = w_sel
    return w_full, indices


def casp_return_selection_projection(z: np.ndarray, Omega: np.ndarray, 
                                       mu: np.ndarray, K: int, lb: float, ub: float,
                                       lambda_ret: float = 0.8, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return-aware selection + Mahalanobis projection (no return in projection).
    Isolates the effect of return-aware selection.
    """
    n = len(z)
    indices = select_assets_return_aware(z, Omega, mu, K, lambda_ret)
    z_sel = z[indices]
    Omega_sel = Omega[np.ix_(indices, indices)]
    
    w_sel = project_mahalanobis(z_sel, Omega_sel, K, lb, ub)
    
    w_full = np.zeros(n)
    w_full[indices] = w_sel
    return w_full, indices


def casp_full_projection(z: np.ndarray, Omega: np.ndarray, mu: np.ndarray,
                          K: int, lb: float, ub: float,
                          lambda_ret: float = 0.8, 
                          gamma_ret: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full RA-CASP: Return-aware selection + Return-regularized Mahalanobis projection.
    """
    n = len(z)
    indices = select_assets_return_aware(z, Omega, mu, K, lambda_ret)
    z_sel = z[indices]
    Omega_sel = Omega[np.ix_(indices, indices)]
    mu_sel = mu[indices]
    
    w_sel = project_mahalanobis_return_regularized(z_sel, Omega_sel, mu_sel, K, lb, ub, gamma_ret)
    
    w_full = np.zeros(n)
    w_full[indices] = w_sel
    return w_full, indices


def minvar_selection_euclidean_projection(z: np.ndarray, Omega: np.ndarray,
                                           K: int, lb: float, ub: float,
                                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """MinVar selection + Euclidean projection (baseline comparison)."""
    n = len(z)
    indices = select_assets_min_variance(z, Omega, K)
    z_sel = z[indices]
    w_sel = project_euclidean(z_sel, K, lb, ub)
    
    w_full = np.zeros(n)
    w_full[indices] = w_sel
    return w_full, indices


def sharpe_selection_euclidean_projection(z: np.ndarray, Omega: np.ndarray,
                                           mu: np.ndarray, K: int, 
                                           lb: float, ub: float,
                                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Sharpe-based selection + Euclidean projection (return-aware baseline)."""
    n = len(z)
    indices = select_assets_sharpe_based(z, Omega, mu, K)
    z_sel = z[indices]
    w_sel = project_euclidean(z_sel, K, lb, ub)
    
    w_full = np.zeros(n)
    w_full[indices] = w_sel
    return w_full, indices


# --- Portfolio Metrics ---

def portfolio_variance(w: np.ndarray, Omega: np.ndarray) -> float:
    return w @ Omega @ w


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return mu @ w


def portfolio_esg(w: np.ndarray, esg: np.ndarray) -> float:
    return esg @ w


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, Omega: np.ndarray, 
                  rf: float = 0.045) -> float:
    ret = portfolio_return(w, mu)
    vol = np.sqrt(portfolio_variance(w, Omega))
    return (ret - rf) / (vol + 1e-10)


def compute_turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    """Compute portfolio turnover (one-way)."""
    return 0.5 * np.sum(np.abs(w_new - w_old))


def compute_transaction_cost(turnover: float, cost_bps: float = 10) -> float:
    """Compute transaction cost from turnover."""
    return turnover * cost_bps / 10000


# --- Multi-Objective Grey Wolf Optimizer ---

def dominates(obj1: Tuple, obj2: Tuple) -> bool:
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))


def non_dominated_indices(objectives: List[Tuple]) -> List[int]:
    n = len(objectives)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i != j and not dominated[j] and dominates(objectives[j], objectives[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def hypervolume_2d(objectives: List[Tuple], ref: Tuple) -> float:
    points = sorted([o for o in objectives if o[0] < ref[0] and o[1] < ref[1]], key=lambda x: x[0])
    hv = 0.0
    prev_y = ref[1]
    for p in points:
        hv += (ref[0] - p[0]) * (prev_y - p[1])
        prev_y = min(prev_y, p[1])
    return hv


class MOGWO:
    """Multi-Objective Grey Wolf Optimizer with pluggable repair operator."""
    
    def __init__(self, mu: np.ndarray, Omega: np.ndarray, esg: np.ndarray,
                 K: int, lb: float, ub: float,
                 pop_size: int = 50, max_iter: int = 100, archive_size: int = 30,
                 projection_method: str = 'casp_full',
                 lambda_ret: float = 0.8, gamma_ret: float = 0.25):
        self.mu = mu
        self.Omega = Omega
        self.esg = esg
        self.n = len(mu)
        self.K = K
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.projection_method = projection_method
        self.lambda_ret = lambda_ret
        self.gamma_ret = gamma_ret
        self.hv_history = []
    
    def repair(self, z: np.ndarray) -> np.ndarray:
        if self.projection_method == 'euclidean':
            w, _ = euclidean_projection(z, self.K, self.lb, self.ub)
        elif self.projection_method == 'casp_basic':
            w, _ = casp_basic_projection(z, self.Omega, self.K, self.lb, self.ub)
        elif self.projection_method == 'casp_return_selection':
            w, _ = casp_return_selection_projection(z, self.Omega, self.mu, self.K, 
                                                     self.lb, self.ub, self.lambda_ret)
        elif self.projection_method == 'casp_full':
            w, _ = casp_full_projection(z, self.Omega, self.mu, self.K, 
                                         self.lb, self.ub, self.lambda_ret, self.gamma_ret)
        elif self.projection_method == 'minvar_euc':
            w, _ = minvar_selection_euclidean_projection(z, self.Omega, self.K, self.lb, self.ub)
        elif self.projection_method == 'sharpe_euc':
            w, _ = sharpe_selection_euclidean_projection(z, self.Omega, self.mu, self.K, self.lb, self.ub)
        else:
            w, _ = euclidean_projection(z, self.K, self.lb, self.ub)
        return w
    
    def evaluate(self, w: np.ndarray) -> Tuple[float, float, float]:
        var = portfolio_variance(w, self.Omega)
        neg_ret = -portfolio_return(w, self.mu)
        neg_esg = -portfolio_esg(w, self.esg)
        return (var, neg_ret, neg_esg)
    
    def run(self) -> List[Tuple[np.ndarray, Tuple]]:
        pop = np.random.randn(self.pop_size, self.n) * 0.3
        portfolios = [self.repair(p) for p in pop]
        objectives = [self.evaluate(w) for w in portfolios]
        
        nd_idx = non_dominated_indices(objectives)
        archive = [(portfolios[i], objectives[i]) for i in nd_idx[:self.archive_size]]
        
        for t in range(self.max_iter):
            a = 2.0 * (1 - t / self.max_iter)
            
            if len(archive) >= 3:
                lidx = np.random.choice(len(archive), 3, replace=False)
                alpha, beta, delta = [archive[i][0] for i in lidx]
            elif len(archive) > 0:
                alpha = beta = delta = archive[0][0]
            else:
                alpha = beta = delta = portfolios[0]
            
            new_portfolios = []
            new_objectives = []
            
            for i in range(self.pop_size):
                A1 = 2 * a * np.random.rand() - a
                A2 = 2 * a * np.random.rand() - a
                A3 = 2 * a * np.random.rand() - a
                C1, C2, C3 = 2 * np.random.rand(3)
                
                X1 = alpha - A1 * np.abs(C1 * alpha - pop[i])
                X2 = beta - A2 * np.abs(C2 * beta - pop[i])
                X3 = delta - A3 * np.abs(C3 * delta - pop[i])
                
                pop[i] = (X1 + X2 + X3) / 3
                w = self.repair(pop[i])
                new_portfolios.append(w)
                new_objectives.append(self.evaluate(w))
            
            combined = archive + list(zip(new_portfolios, new_objectives))
            combined_obj = [x[1] for x in combined]
            nd_idx = non_dominated_indices(combined_obj)
            
            if len(nd_idx) > self.archive_size:
                nd_idx = nd_idx[:self.archive_size]
            
            archive = [combined[i] for i in nd_idx]
            
            if archive:
                arch_obj = [x[1] for x in archive]
                obj_2d = [(o[0], o[1]) for o in arch_obj]
                max_var = max(o[0] for o in arch_obj) * 1.2
                hv = hypervolume_2d(obj_2d, (max_var, 0.0))
                self.hv_history.append(hv)
        
        return archive


# --- Experiments ---

def run_direct_comparison_ablation(mu: np.ndarray, Omega: np.ndarray, 
                                    K: int, lb: float, ub: float,
                                    n_samples: int = 500,
                                    lambda_ret: float = 0.8,
                                    gamma_ret: float = 0.25) -> Dict:
    """
    Direct pairwise comparison with ablation study.
    Compare 6 methods to isolate contributions.
    """
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: Direct Comparison ({n_samples} samples)")
    print(f"Parameters: λ={lambda_ret}, γ={gamma_ret}")
    print('='*70)
    
    n = len(mu)
    
    methods = {
        'Euclidean': lambda z: euclidean_projection(z, K, lb, ub),
        'VolNorm+Euc': lambda z: volnorm_selection_euclidean_projection(z, Omega, K, lb, ub),
        'MinVar+Euc': lambda z: minvar_selection_euclidean_projection(z, Omega, K, lb, ub),
        'Sharpe+Euc': lambda z: sharpe_selection_euclidean_projection(z, Omega, mu, K, lb, ub),
        'CASP-Basic': lambda z: casp_basic_projection(z, Omega, K, lb, ub),
        'CASP-RetSel': lambda z: casp_return_selection_projection(z, Omega, mu, K, lb, ub, lambda_ret),
        'RA-CASP': lambda z: casp_full_projection(z, Omega, mu, K, lb, ub, lambda_ret, gamma_ret),
    }
    
    results = {name: {'vars': [], 'sharpes': [], 'rets': []} for name in methods}
    
    for i in range(n_samples):
        np.random.seed(i + 1000)
        z = np.random.randn(n) * 0.4
        
        for name, proj_func in methods.items():
            w, _ = proj_func(z)
            results[name]['vars'].append(portfolio_variance(w, Omega))
            results[name]['sharpes'].append(sharpe_ratio(w, mu, Omega))
            results[name]['rets'].append(portfolio_return(w, mu))
    
    # Convert to arrays
    for name in methods:
        for key in results[name]:
            results[name][key] = np.array(results[name][key])
    
    # Print results table
    print(f"\n{'Method':<15} {'Mean Var':<12} {'Mean Sharpe':<12} {'Mean Ret':<12}")
    print("-" * 55)
    for name in methods:
        mv = results[name]['vars'].mean()
        ms = results[name]['sharpes'].mean()
        mr = results[name]['rets'].mean()
        print(f"{name:<15} {mv:.6f}     {ms:.4f}       {mr:.4f}")
    
    # Statistical tests vs Euclidean baseline
    baseline = 'Euclidean'
    print(f"\n\nStatistical Tests vs {baseline}:")
    print("-" * 70)
    
    for name in methods:
        if name == baseline:
            continue
        
        # Variance comparison
        var_wins = np.mean(results[name]['vars'] < results[baseline]['vars']) * 100
        _, p_var = wilcoxon(results[name]['vars'], results[baseline]['vars'], alternative='less')
        
        # Sharpe comparison
        sharpe_wins = np.mean(results[name]['sharpes'] > results[baseline]['sharpes']) * 100
        _, p_sharpe = wilcoxon(results[name]['sharpes'], results[baseline]['sharpes'], alternative='greater')
        
        var_reduction = (results[baseline]['vars'].mean() - results[name]['vars'].mean()) / results[baseline]['vars'].mean() * 100
        sharpe_improvement = (results[name]['sharpes'].mean() - results[baseline]['sharpes'].mean()) / results[baseline]['sharpes'].mean() * 100
        
        print(f"{name:<15}: VarRed={var_reduction:+.1f}% (p={p_var:.2e}), SharpeImp={sharpe_improvement:+.1f}% (p={p_sharpe:.2e})")
    
    return results


def run_out_of_sample_validation(train_prices: pd.DataFrame, 
                                  test_prices: pd.DataFrame,
                                  K: int, lb: float, ub: float,
                                  n_samples: int = 200,
                                  lambda_ret: float = 0.8,
                                  gamma_ret: float = 0.25,
                                  rf: float = 0.045) -> Dict:
    """
    Out-of-sample validation: Train on historical data, test on held-out period.
    """
    print(f"\n{'='*70}")
    print("OUT-OF-SAMPLE VALIDATION")
    print('='*70)
    
    # Compute in-sample statistics
    mu_train, Omega_train, tickers = compute_financial_metrics(train_prices)
    n = len(mu_train)
    
    methods = {
        'Euclidean': lambda z: euclidean_projection(z, K, lb, ub),
        'VolNorm+Euc': lambda z: volnorm_selection_euclidean_projection(z, Omega_train, K, lb, ub),
        'CASP-Basic': lambda z: casp_basic_projection(z, Omega_train, K, lb, ub),
        'RA-CASP': lambda z: casp_full_projection(z, Omega_train, mu_train, K, lb, ub, lambda_ret, gamma_ret),
        'Sharpe+Euc': lambda z: sharpe_selection_euclidean_projection(z, Omega_train, mu_train, K, lb, ub),
    }
    
    results = {name: {'in_sample': [], 'out_sample': []} for name in methods}
    
    print(f"\nGenerating {n_samples} portfolios and testing out-of-sample...")
    
    for i in range(n_samples):
        np.random.seed(i + 2000)
        z = np.random.randn(n) * 0.4
        
        for name, proj_func in methods.items():
            w, _ = proj_func(z)
            
            # In-sample metrics (using training data estimates)
            in_sample_sharpe = sharpe_ratio(w, mu_train, Omega_train, rf)
            results[name]['in_sample'].append(in_sample_sharpe)
            
            # Out-of-sample metrics (realized on test data)
            oos_metrics = compute_out_of_sample_metrics(w, test_prices, rf)
            results[name]['out_sample'].append(oos_metrics['realized_sharpe'])
    
    # Convert to arrays
    for name in methods:
        results[name]['in_sample'] = np.array(results[name]['in_sample'])
        results[name]['out_sample'] = np.array(results[name]['out_sample'])
    
    # Print results
    print(f"\n{'Method':<15} {'In-Sample Sharpe':<18} {'Out-Sample Sharpe':<18} {'Correlation':<12}")
    print("-" * 65)
    
    for name in methods:
        is_mean = results[name]['in_sample'].mean()
        oos_mean = results[name]['out_sample'].mean()
        corr, _ = spearmanr(results[name]['in_sample'], results[name]['out_sample'])
        print(f"{name:<15} {is_mean:.4f}             {oos_mean:.4f}              {corr:.3f}")
    
    # Statistical tests for out-of-sample performance
    print(f"\nOut-of-Sample Statistical Tests vs Euclidean:")
    baseline = 'Euclidean'
    for name in methods:
        if name == baseline:
            continue
        wins = np.mean(results[name]['out_sample'] > results[baseline]['out_sample']) * 100
        _, p_val = wilcoxon(results[name]['out_sample'], results[baseline]['out_sample'], alternative='greater')
        improvement = (results[name]['out_sample'].mean() - results[baseline]['out_sample'].mean()) / (abs(results[baseline]['out_sample'].mean()) + 1e-6) * 100
        print(f"  {name}: Wins {wins:.1f}%, Improvement {improvement:+.1f}%, p={p_val:.4f}")
    
    return results


def run_walk_forward_validation(prices: pd.DataFrame,
                                train_end_dates: List[str],
                                K: int, lb: float, ub: float,
                                n_samples: int = 200,
                                lambda_ret: float = 0.8,
                                gamma_ret: float = 0.25,
                                rf: float = 0.045,
                                shrinkage: float = 0.1,
                                seed_base: int = 2000) -> pd.DataFrame:
    """
    Walk-forward (expanding window) validation across multiple train/test splits.

    For each `train_end` date:
      - training = prices[<=train_end]
      - test = prices[(>train_end) & (<= end_of_next_year)]
    """
    rows = []
    for train_end in train_end_dates:
        train = prices[prices.index <= train_end]
        if train.empty:
            continue

        # Define test window as the next calendar year (or whatever is available)
        year = pd.Timestamp(train_end).year
        test_end = f'{year+1}-12-31'
        test = prices[(prices.index > train_end) & (prices.index <= test_end)]
        if test.empty:
            continue

        mu_train, Omega_train, _ = compute_financial_metrics(train, shrinkage=shrinkage)
        n = len(mu_train)

        methods = {
            'Euclidean': lambda z: euclidean_projection(z, K, lb, ub),
            'VolNorm+Euc': lambda z: volnorm_selection_euclidean_projection(z, Omega_train, K, lb, ub),
            'CASP-Basic': lambda z: casp_basic_projection(z, Omega_train, K, lb, ub),
            'RA-CASP': lambda z: casp_full_projection(z, Omega_train, mu_train, K, lb, ub, lambda_ret, gamma_ret),
            'Sharpe+Euc': lambda z: sharpe_selection_euclidean_projection(z, Omega_train, mu_train, K, lb, ub),
        }

        for i in range(n_samples):
            # Use a fixed seed schedule so splits are directly comparable.
            # (Also makes the 2024 split match `run_out_of_sample_validation` by default.)
            np.random.seed(seed_base + i)
            z = np.random.randn(n) * 0.4
            for name, proj_func in methods.items():
                w, _ = proj_func(z)
                oos = compute_out_of_sample_metrics(w, test, rf=rf)['realized_sharpe']
                rows.append({
                    'train_end': train_end,
                    'train_start': str(train.index[0].date()),
                    'test_start': str(test.index[0].date()),
                    'test_end': str(test.index[-1].date()),
                    'test_n_days': int(len(test)),
                    'method': name,
                    'oos_sharpe': float(oos),
                })

    df = pd.DataFrame(rows)
    return df


def run_turnover_analysis(mu: np.ndarray, Omega: np.ndarray,
                           K: int, lb: float, ub: float,
                           n_rebalances: int = 50,
                           lambda_ret: float = 0.8,
                           gamma_ret: float = 0.25,
                           transaction_cost_bps: float = 10) -> Dict:
    """
    Analyze turnover and transaction costs across rebalancing events.
    """
    print(f"\n{'='*70}")
    print("TURNOVER AND TRANSACTION COST ANALYSIS")
    print(f"Transaction cost: {transaction_cost_bps} bps")
    print('='*70)
    
    n = len(mu)
    
    methods = {
        'Euclidean': lambda z: euclidean_projection(z, K, lb, ub),
        'CASP-Basic': lambda z: casp_basic_projection(z, Omega, K, lb, ub),
        'RA-CASP': lambda z: casp_full_projection(z, Omega, mu, K, lb, ub, lambda_ret, gamma_ret),
    }
    
    results = {name: {'turnovers': [], 'costs': [], 'net_sharpes': []} for name in methods}
    
    # Simulate rebalancing
    for name, proj_func in methods.items():
        np.random.seed(3000)
        
        # Initial portfolio
        z0 = np.random.randn(n) * 0.4
        w_prev, _ = proj_func(z0)
        
        for i in range(n_rebalances):
            # New signal with some persistence
            z_new = 0.7 * z0 + 0.3 * np.random.randn(n) * 0.4
            z0 = z_new
            
            w_new, _ = proj_func(z_new)
            
            turnover = compute_turnover(w_new, w_prev)
            cost = compute_transaction_cost(turnover, transaction_cost_bps)
            
            gross_sharpe = sharpe_ratio(w_new, mu, Omega)
            # Approximate net Sharpe (subtract annualized cost assuming 12 rebalances/year)
            net_sharpe = gross_sharpe - cost * 12
            
            results[name]['turnovers'].append(turnover)
            results[name]['costs'].append(cost)
            results[name]['net_sharpes'].append(net_sharpe)
            
            w_prev = w_new
    
    # Convert to arrays and print
    print(f"\n{'Method':<15} {'Avg Turnover':<14} {'Avg Cost (bps)':<16} {'Avg Net Sharpe':<14}")
    print("-" * 60)
    
    for name in methods:
        results[name]['turnovers'] = np.array(results[name]['turnovers'])
        results[name]['costs'] = np.array(results[name]['costs'])
        results[name]['net_sharpes'] = np.array(results[name]['net_sharpes'])
        
        avg_turn = results[name]['turnovers'].mean()
        avg_cost = results[name]['costs'].mean() * 10000  # Convert to bps
        avg_net = results[name]['net_sharpes'].mean()
        
        print(f"{name:<15} {avg_turn:.4f}         {avg_cost:.2f}             {avg_net:.4f}")
    
    return results


def run_optimization_comparison(mu: np.ndarray, Omega: np.ndarray, 
                                  esg: np.ndarray, K: int, lb: float, ub: float,
                                  n_runs: int = 15,
                                  lambda_ret: float = 0.8,
                                  gamma_ret: float = 0.25,
                                  pop_size: int = 50,
                                  max_iter: int = 100) -> Dict:
    """Compare projection methods within MOGWO optimization."""
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPARISON: {n_runs} independent runs")
    print('='*70)
    
    projection_methods = ['euclidean', 'casp_basic', 'casp_full', 'sharpe_euc']
    method_names = ['Euclidean', 'CASP-Basic', 'RA-CASP', 'Sharpe+Euc']
    
    results = {name: {'hv': [], 'best_sharpe': [], 'best_ret': [], 'avg_var': []} 
               for name in method_names}
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)
        
        for proj, name in zip(projection_methods, method_names):
            np.random.seed(run * 100 + hash(proj) % 1000)
            
            opt = MOGWO(mu, Omega, esg, K, lb, ub,
                        pop_size=pop_size, max_iter=max_iter, archive_size=30,
                        projection_method=proj, lambda_ret=lambda_ret, gamma_ret=gamma_ret)
            archive = opt.run()
            
            variances = [portfolio_variance(w, Omega) for w, _ in archive]
            returns = [portfolio_return(w, mu) for w, _ in archive]
            sharpes = [sharpe_ratio(w, mu, Omega) for w, _ in archive]
            
            results[name]['hv'].append(opt.hv_history[-1] if opt.hv_history else 0)
            results[name]['best_sharpe'].append(np.max(sharpes))
            results[name]['best_ret'].append(np.max(returns))
            results[name]['avg_var'].append(np.mean(variances))
        
        print("done")
    
    # Convert to arrays
    for name in method_names:
        for key in results[name]:
            results[name][key] = np.array(results[name][key])
    
    # Print summary
    print(f"\n{'Method':<15} {'Best Sharpe':<14} {'Best Return':<14} {'Hypervolume':<14}")
    print("-" * 60)
    
    for name in method_names:
        bs = results[name]['best_sharpe'].mean()
        br = results[name]['best_ret'].mean()
        hv = results[name]['hv'].mean()
        print(f"{name:<15} {bs:.4f}         {br:.4f}         {hv:.6f}")
    
    # Statistical tests
    print(f"\nStatistical Tests (Wilcoxon signed-rank):")
    baseline = 'Euclidean'
    for name in method_names:
        if name == baseline:
            continue
        _, p_sharpe = wilcoxon(results[name]['best_sharpe'], results[baseline]['best_sharpe'])
        _, p_hv = wilcoxon(results[name]['hv'], results[baseline]['hv'])
        sharpe_imp = (results[name]['best_sharpe'].mean() - results[baseline]['best_sharpe'].mean()) / results[baseline]['best_sharpe'].mean() * 100
        hv_imp = (results[name]['hv'].mean() - results[baseline]['hv'].mean()) / (results[baseline]['hv'].mean() + 1e-10) * 100
        print(f"  {name}: Sharpe +{sharpe_imp:.1f}% (p={p_sharpe:.4f}), HV +{hv_imp:.1f}% (p={p_hv:.4f})")
    
    return results


def tune_parameters(mu: np.ndarray, Omega: np.ndarray, 
                     K: int, lb: float, ub: float,
                     n_samples: int = 300) -> Tuple[float, float]:
    """Find optimal lambda and gamma parameters."""
    print(f"\n{'='*70}")
    print("PARAMETER TUNING")
    print('='*70)
    
    n = len(mu)
    
    # Generate fixed random candidates
    np.random.seed(999)
    candidates = [np.random.randn(n) * 0.4 for _ in range(n_samples)]
    
    # Compute Euclidean baseline
    euc_sharpes = []
    euc_vars = []
    for z in candidates:
        we, _ = euclidean_projection(z, K, lb, ub)
        euc_sharpes.append(sharpe_ratio(we, mu, Omega))
        euc_vars.append(portfolio_variance(we, Omega))
    euc_sharpe_mean = np.mean(euc_sharpes)
    euc_var_mean = np.mean(euc_vars)
    
    print(f"  Euclidean baseline: Sharpe={euc_sharpe_mean:.4f}, Var={euc_var_mean:.6f}")
    
    best_score = -np.inf
    best_params = (0.8, 0.25)
    results_grid = []
    
    # Parameter grid
    for lambda_ret in [0.4, 0.6, 0.8, 1.0, 1.2]:
        for gamma_ret in [0.15, 0.20, 0.25, 0.30, 0.35]:
            casp_sharpes = []
            casp_vars = []
            
            for z in candidates:
                wc, _ = casp_full_projection(z, Omega, mu, K, lb, ub, lambda_ret, gamma_ret)
                casp_sharpes.append(sharpe_ratio(wc, mu, Omega))
                casp_vars.append(portfolio_variance(wc, Omega))
            
            casp_sharpe_mean = np.mean(casp_sharpes)
            casp_var_mean = np.mean(casp_vars)
            
            var_improvement = (euc_var_mean - casp_var_mean) / euc_var_mean * 100
            sharpe_improvement = (casp_sharpe_mean - euc_sharpe_mean) / euc_sharpe_mean * 100
            
            # Score: balance variance reduction and Sharpe improvement
            if var_improvement > 0:
                score = var_improvement * 0.6 + sharpe_improvement * 0.4
            else:
                score = -100
            
            results_grid.append({
                'lambda': lambda_ret, 'gamma': gamma_ret,
                'var_imp': var_improvement, 'sharpe_imp': sharpe_improvement,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = (lambda_ret, gamma_ret)
    
    print(f"\n  Top 5 parameter combinations:")
    results_grid.sort(key=lambda x: x['score'], reverse=True)
    for r in results_grid[:5]:
        print(f"    λ={r['lambda']:.1f}, γ={r['gamma']:.2f}: VarRed={r['var_imp']:.1f}%, SharpeImp={r['sharpe_imp']:+.1f}%")
    
    print(f"\n  Selected: λ={best_params[0]}, γ={best_params[1]}")
    
    return best_params


# --- Main ---

if __name__ == '__main__':
    print("="*70)
    print("RA-CASP: Covariance-Aware Simplex Projection")
    print("Enhanced Analysis with Out-of-Sample Validation & Ablation Study")
    print("="*70)
    
    # Configuration
    config = ExperimentConfig()
    
    # Load real S&P 500 data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'sp500_prices.csv')
    try:
        prices = load_sp500_data(data_path)
    except FileNotFoundError:
        print(f"\nS&P 500 data not found at {data_path}")
        print("Please run download_data.py first to download the data:")
        print("  python download_data.py")
        print("\nFalling back to synthetic data for demonstration...")
        np.random.seed(42)
        n_assets = config.n_assets
        n_days = 1237
        
        # Generate realistic synthetic returns
        dates = pd.date_range('2020-01-02', periods=n_days, freq='B')
        tickers = [f'STOCK{i:03d}' for i in range(n_assets)]
        
        # Factor model for returns
        market_factor = np.random.randn(n_days) * 0.01
        sector_factors = np.random.randn(n_days, 5) * 0.005
        
        returns = np.zeros((n_days, n_assets))
        for i in range(n_assets):
            sector = i % 5
            beta = 0.5 + np.random.rand() * 1.0
            returns[:, i] = beta * market_factor + sector_factors[:, sector] + np.random.randn(n_days) * 0.015
        
        # Convert to prices
        prices_arr = 100 * np.exp(np.cumsum(returns, axis=0))
        prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)
        
        print(f"  Generated synthetic data: {n_assets} assets, {n_days} days")
    
    # Split data for out-of-sample testing
    train_prices, test_prices = split_data_temporal(prices, train_end='2023-12-31')
    
    # Compute financial metrics on training data
    mu, Omega, tickers = compute_financial_metrics(train_prices)
    
    # Try to load real ESG data, fall back to synthetic
    esg_file = os.path.join(base_dir, '..', 'data', 'sp500_esg.csv')
    esg = generate_esg_scores(tickers, mu, esg_file=esg_file)
    
    n_assets = len(mu)
    K = config.cardinality_K
    lb, ub = config.weight_lb, config.weight_ub
    
    print(f"\nExperiment Configuration:")
    print(f"  Number of assets (N): {n_assets}")
    print(f"  Cardinality constraint (K): {K}")
    print(f"  Weight bounds: [{lb*100:.0f}%, {ub*100:.0f}%]")
    
    # Parameter tuning
    best_lambda, best_gamma = tune_parameters(mu, Omega, K, lb, ub, n_samples=200)
    
    # Run experiments
    ablation_results = run_direct_comparison_ablation(
        mu, Omega, K, lb, ub, n_samples=config.n_direct_samples,
        lambda_ret=best_lambda, gamma_ret=best_gamma
    )
    
    oos_results = run_out_of_sample_validation(
        train_prices, test_prices, K, lb, ub, n_samples=200,
        lambda_ret=best_lambda, gamma_ret=best_gamma
    )

    # Export for reproducible figures/tables
    results_dir = os.path.join(base_dir, '..', 'results')
    export_ablation_results(ablation_results, results_dir, baseline='Euclidean')
    export_oos_results(oos_results, results_dir)

    # Walk-forward across multiple years (more concrete OOS evidence)
    wf = run_walk_forward_validation(
        prices,
        train_end_dates=['2021-12-31', '2022-12-31', '2023-12-31'],
        K=K, lb=lb, ub=ub,
        n_samples=200,
        lambda_ret=best_lambda, gamma_ret=best_gamma,
        rf=config.risk_free_rate
    )
    if not wf.empty:
        _ensure_dir(results_dir)
        wf.to_csv(os.path.join(results_dir, 'walk_forward_oos_raw.csv'), index=False)
        wf_summary = (wf.groupby(['train_end', 'method'])['oos_sharpe']
                        .agg(['mean', 'std', 'count'])
                        .reset_index()
                        .rename(columns={'mean': 'mean_oos_sharpe', 'std': 'std_oos_sharpe', 'count': 'n'}))
        wf_summary.to_csv(os.path.join(results_dir, 'walk_forward_oos_summary.csv'), index=False)
    
    turnover_results = run_turnover_analysis(
        mu, Omega, K, lb, ub, n_rebalances=50,
        lambda_ret=best_lambda, gamma_ret=best_gamma,
        transaction_cost_bps=config.transaction_cost_bps
    )
    
    opt_results = run_optimization_comparison(
        mu, Omega, esg, K, lb, ub, n_runs=config.n_optimization_runs,
        lambda_ret=best_lambda, gamma_ret=best_gamma,
        pop_size=config.mogwo_pop_size, max_iter=config.mogwo_max_iter
    )
    
    # Note: Publication figures are generated separately using create_figures.py
    
    # Final Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\n1. ABLATION STUDY:")
    euc_var = ablation_results['Euclidean']['vars'].mean()
    casp_var = ablation_results['RA-CASP']['vars'].mean()
    casp_basic_var = ablation_results['CASP-Basic']['vars'].mean()
    volnorm_euc_var = ablation_results['VolNorm+Euc']['vars'].mean() if 'VolNorm+Euc' in ablation_results else np.nan
    print(f"   - Mahalanobis projection (CASP-Basic): {(euc_var - casp_basic_var)/euc_var*100:.1f}% variance reduction")
    print(f"   - Full RA-CASP: {(euc_var - casp_var)/euc_var*100:.1f}% variance reduction")
    if np.isfinite(volnorm_euc_var):
        print(f"   - Volatility-normalized selection alone (VolNorm+Euc): {(euc_var - volnorm_euc_var)/euc_var*100:.1f}% variance reduction")
        print(f"   - Incremental Σ-metric geometry (CASP-Basic vs VolNorm+Euc): {(volnorm_euc_var - casp_basic_var)/euc_var*100:.1f}%")
    print(f"   - Additional contribution from return terms (RA-CASP vs CASP-Basic): {(casp_basic_var - casp_var)/euc_var*100:.1f}%")
    
    print(f"\n2. OUT-OF-SAMPLE VALIDATION:")
    for m in oos_results:
        is_mean = oos_results[m]['in_sample'].mean()
        oos_mean = oos_results[m]['out_sample'].mean()
        corr, _ = spearmanr(oos_results[m]['in_sample'], oos_results[m]['out_sample'])
        print(f"   - {m}: In-Sample={is_mean:.3f}, Out-Sample={oos_mean:.3f}, Correlation={corr:.3f}")
    
    print(f"\n3. TRANSACTION COSTS:")
    for m in turnover_results:
        avg_turn = turnover_results[m]['turnovers'].mean()
        avg_net = turnover_results[m]['net_sharpes'].mean()
        print(f"   - {m}: Turnover={avg_turn:.3f}, Net Sharpe={avg_net:.3f}")
    
    print(f"\n4. OPTIMIZATION:")
    for m in opt_results:
        bs = opt_results[m]['best_sharpe'].mean()
        hv = opt_results[m]['hv'].mean()
        print(f"   - {m}: Best Sharpe={bs:.3f}, Hypervolume={hv:.6f}")
    
    print("\n" + "="*70)
    print("Analysis complete. Figures saved to paper_v2 directory.")
    print("="*70)

