#!/usr/bin/env python3
"""
Generate English Weibull / RUL survival analysis slides.

Covers:
  1. Single-component Weibull (Global vs Vt)
  2. Hazard rate function
  3. TTF vs RUL (conditional survival, conditional median)
  4. E[T] vs Median — recommendation
  5. Cox Proportional Hazards
  6. Chemical & contractor covariate effects
  7. Latent 2-component mixture (Vt)
  8. Summary / RUL priors

Usage:
    python scripts/generate_weibull_slides_en.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from scipy.special import gamma as sc_gamma
from scipy.optimize import brentq

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
from analysis.paths import results_dir

# ─── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.dpi': 150,
})

# ─── Color constants ──────────────────────────────────────────────────────────
# python-pptx RGBColor
NAVY   = RGBColor(0x0D, 0x2A, 0x4A)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT  = RGBColor(0xB0, 0xC4, 0xDE)

# Matplotlib hex
H_GLOBAL  = '#1565C0'
H_VT      = '#C62828'
H_COMP1   = '#E65100'   # early failures
H_COMP2   = '#2E7D32'   # wear-out
H_NAVY    = '#0D2A4A'
H_AMBER   = '#F9A825'
H_PURPLE  = '#6A1B9A'

# ─── Weibull model parameters ─────────────────────────────────────────────────
# 1-component MLE, Кэкспл > 0.5 filter
# Source: docs/methodology/weibull_survival_report.md
WBL = {
    'global': dict(beta=1.059, eta=443, n=1237, label='Global Fleet', color=H_GLOBAL, ls='-'),
    'vt':     dict(beta=1.229, eta=239, n=255,  label='Vt Field',     color=H_VT,    ls='--'),
}

# Bayesian 2-component latent mixture (posterior q50)
# Source: analysis/vt_ya_global_survival_comparison.csv
LATENT = {
    'global': dict(
        comp1=dict(w=0.420, eta=235, beta=0.835),
        comp2=dict(w=0.580, eta=469, beta=1.021),
        B10=25.6, B25=83.8, B50=238.2, B90=947.8,
        n=2124, label='Global Fleet',
    ),
    'vt': dict(
        comp1=dict(w=0.332, eta=123, beta=0.918),
        comp2=dict(w=0.668, eta=251, beta=1.155),
        B10=19.5, B25=57.2, B50=144.7, B90=484.5,
        n=310, label='Vt Field',
    ),
}

# Chemical factors — fleet kekspl medians (days)
# Source: docs/methodology/weibull_external_factors_report.md
CHEM = {
    'Cl load':  dict(low=583, mid=387, high=166, p='≈ 0.000', unit='kg/d'),
    'Ca load':  dict(low=554, mid=403, high=170, p='< 0.001', unit='kg/d'),
    'SO₄ load': dict(low=524, mid=338, high=206, p='< 0.01',  unit='kg/d'),
    'H₂S (0→≥100 mg/L)': dict(low=251, mid=218, high=80,  p='≈ 0.000', unit='binary'),
}

# Contractor — kekspl medians
CONTRACTOR = {
    'Fleet':   dict(Borec=373, Schlumberger=299),
    'Vt Field': dict(Borec=227, Schlumberger=181),
}

# ─── Weibull mathematics ──────────────────────────────────────────────────────

def wbl_S(t, beta, eta):
    return np.exp(-(t / eta) ** beta)

def wbl_h(t, beta, eta):
    return (beta / eta) * (t / eta) ** (beta - 1)

def wbl_H(t, beta, eta):
    return (t / eta) ** beta

def wbl_mean(beta, eta):
    return eta * sc_gamma(1.0 + 1.0 / beta)

def wbl_median(beta, eta):
    return eta * np.log(2.0) ** (1.0 / beta)

def cond_median_rul(t_age, beta, eta):
    """Conditional P50 RUL given survival to age t_age (days)."""
    inner = t_age ** beta + eta ** beta * np.log(2.0)
    return inner ** (1.0 / beta) - t_age

def cond_S(tau, t_age, beta, eta):
    """P(T > t_age + tau | T > t_age)."""
    return np.exp(-((t_age + tau) ** beta - t_age ** beta) / eta ** beta)

def mix_S(t, c1, c2):
    return c1['w'] * wbl_S(t, c1['beta'], c1['eta']) + c2['w'] * wbl_S(t, c2['beta'], c2['eta'])

def mix_mean(c1, c2):
    return c1['w'] * wbl_mean(c1['beta'], c1['eta']) + c2['w'] * wbl_mean(c2['beta'], c2['eta'])

def mix_quantile(pct, c1, c2, t_max=3000):
    target = 1.0 - pct / 100.0
    return brentq(lambda t: mix_S(t, c1, c2) - target, 1e-6, t_max)

# ─── Figure helpers ───────────────────────────────────────────────────────────

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=160)
    buf.seek(0)
    plt.close(fig)
    return buf

def render_eq(latex_str, fontsize=22, figsize=(7.5, 1.0)):
    """Render a mathtext equation and return PNG bytes."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.text(0.5, 0.5, f'${latex_str}$',
            transform=ax.transAxes, fontsize=fontsize,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F4F6FA', ec='#CCCCCC', lw=1.2))
    fig.patch.set_facecolor('#F4F6FA')
    return fig_to_bytes(fig)

def style_table(table, n_rows, header_color=H_NAVY, alt_color='#EBF5FB'):
    """Apply styling to a matplotlib table."""
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor(alt_color)
        else:
            cell.set_facecolor('white')

# ─── Individual figures ───────────────────────────────────────────────────────

def fig_beta_shapes():
    """Survival curves for β = 0.5, 1, 1.5, 3, 8 with the same η — illustrates all regimes."""
    eta = 200
    betas = [0.5, 1.0, 1.5, 3.0, 8.0]
    colors = ['#7B1FA2', '#1565C0', '#2E7D32', '#E65100', '#C62828']
    labels = [
        'β = 0.5  Infant mortality',
        'β = 1.0  Random (exponential)',
        'β = 1.5  Mild wear-out',
        'β = 3.0  Wear-out',
        'β = 8.0  Near-deterministic',
    ]
    t = np.linspace(0.01, 600, 600)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for b, c, lbl in zip(betas, colors, labels):
        ax.plot(t, wbl_S(t, b, eta), color=c, lw=2.1, label=lbl)

    ax.axvline(eta, color='#999', lw=1, ls='--', alpha=0.7)
    ax.text(eta + 4, 0.97, f'η = {eta} d\n(63.2 % fail)', color='#888', fontsize=8, va='top')
    ax.axhline(np.exp(-1), color='#999', lw=0.8, ls=':', alpha=0.5)
    ax.text(5, np.exp(-1) + 0.02, 'S = 0.368 = e⁻¹', color='#888', fontsize=7.5)
    ax.set_xlim(0, 580)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Time (days)', fontsize=10)
    ax.set_ylabel('S(t)', fontsize=10)
    ax.set_title(f'Shape β — all with η = {eta} d', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_survival_comparison():
    """Survival curves for Global and Vt with mean & median marked."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    t = np.linspace(0.5, 950, 600)

    label_offset = {'global': (8, 0.565, 0.425), 'vt': (8, 0.395, 0.245)}

    for key, p in WBL.items():
        s    = wbl_S(t, p['beta'], p['eta'])
        med  = wbl_median(p['beta'], p['eta'])
        mea  = wbl_mean(p['beta'], p['eta'])
        ax.plot(t, s, color=p['color'], lw=2.4, ls=p['ls'],
                label=f"{p['label']}  (n={p['n']}, β={p['beta']:.3f}, η={p['eta']} d)")
        dx, y_med, y_mea = label_offset[key]
        # Median marker
        ax.axvline(med, color=p['color'], lw=1.4, ls=':', alpha=0.75)
        ax.text(med + dx, y_med,
                f'Median\n{med:.0f} d', color=p['color'], fontsize=9,
                va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=p['color'], alpha=0.8))
        # Mean marker
        ax.axvline(mea, color=p['color'], lw=1.4, ls='-.', alpha=0.75)
        ax.text(mea + dx, y_mea,
                f'Mean\n{mea:.0f} d', color=p['color'], fontsize=9,
                va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=p['color'], alpha=0.8))

    ax.axhline(0.5, color='#666', lw=0.9, ls='--', alpha=0.45)
    ax.text(890, 0.515, 'P50', color='#666', fontsize=8.5)

    legend_extras = [
        Line2D([0], [0], color='k', lw=1.5, ls=':',  label='Median (P50)'),
        Line2D([0], [0], color='k', lw=1.5, ls='-.', label='Mean E[T]'),
    ]
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_extras, labels=labels_ + ['Median (P50)', 'Mean E[T]'],
              fontsize=9.5, loc='upper right', framealpha=0.9)

    ax.set_xlim(0, 950)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Survival probability  S(t)', fontsize=12)
    ax.set_title('Single-Component Weibull — Survival Functions\n'
                 'Кэкспл > 0.5 filtered fleet (stable runs only)', fontsize=12, fontweight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_slide3_km_weibull():
    """KM + Weibull for raw (unfiltered) Global-excl-Vt and Vt.
    Mean point gets both a vertical drop-line and a horizontal run to the y-axis.
    """
    import sqlite3
    import pandas as pd
    from lifelines import KaplanMeierFitter, WeibullFitter

    # ── load raw data ──────────────────────────────────────────────────────────
    db = Path(REPO_ROOT) / 'data' / 'warehouse' / 'pump2.db'
    con = sqlite3.connect(str(db))
    df = pd.read_sql(
        "SELECT field, ttf_true_best_days AS ttf, event FROM mart__vt_freq55 "
        "WHERE ttf_true_best_days IS NOT NULL AND ttf_true_best_days > 0",
        con,
    )
    con.close()

    groups = {
        'Global excl. Vt': dict(
            mask=df['field'] != 'Vt', color=H_GLOBAL, ls_km='-', ls_wbl='--',
        ),
        'Vt Field': dict(
            mask=df['field'] == 'Vt', color=H_VT, ls_km='-', ls_wbl='--',
        ),
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    fit_params = {}
    for label, cfg in groups.items():
        sub = df[cfg['mask']]
        T, E = sub['ttf'], sub['event']

        # Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(T, E, label='_nolegend_')
        km_t = kmf.survival_function_.index.values
        km_s = kmf.survival_function_.iloc[:, 0].values
        ax.step(km_t, km_s, color=cfg['color'], lw=1.5, alpha=0.55,
                where='post', label=f'{label} — KM  (n={len(T)}, events={E.sum()})')

        # Weibull MLE
        wf = WeibullFitter()
        wf.fit(T, E)
        beta = wf.rho_
        eta  = wf.lambda_
        med  = wbl_median(beta, eta)
        mea  = wbl_mean(beta, eta)
        fit_params[label] = dict(beta=beta, eta=eta, med=med, mea=mea, n=len(T), ev=int(E.sum()))

        t_plot = np.linspace(0.5, km_t.max() * 1.05, 600)
        ax.plot(t_plot, wbl_S(t_plot, beta, eta),
                color=cfg['color'], lw=2.4, ls='--',
                label=f'{label} — Weibull  (β={beta:.3f}, η={eta:.0f} d)')

        # ── Median: vertical dotted line ──────────────────────────────────────
        s_at_med = 0.5
        ax.plot([med, med], [0, s_at_med], color=cfg['color'], lw=1.3, ls=':', alpha=0.7)
        ax.plot(med, s_at_med, 'o', color=cfg['color'], ms=6, zorder=5)

        # ── Mean: vertical + horizontal line to y-axis ────────────────────────
        s_at_mea = wbl_S(mea, beta, eta)
        ax.plot([mea, mea], [0, s_at_mea], color=cfg['color'], lw=1.3, ls='-.', alpha=0.8)
        ax.plot([0, mea],   [s_at_mea, s_at_mea],
                color=cfg['color'], lw=1.2, ls='-.', alpha=0.7)
        ax.plot(0, s_at_mea, '<', color=cfg['color'], ms=6, zorder=5)  # arrow tip on y-axis
        ax.plot(mea, s_at_mea, 'D', color=cfg['color'], ms=6, zorder=5)

    # ── annotations (outside loop, placed once per group) ─────────────────────
    gp = fit_params['Global excl. Vt']
    vp = fit_params['Vt Field']

    gmed, gmea, gmea_s = gp['med'], gp['mea'], wbl_S(gp['mea'], gp['beta'], gp['eta'])
    vmed, vmea, vmea_s = vp['med'], vp['mea'], wbl_S(vp['mea'], vp['beta'], vp['eta'])

    ax.text(gmed + 8, 0.55, f'Median\n{gmed:.0f} d', color=H_GLOBAL, fontsize=8.5, va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_GLOBAL, alpha=0.85))
    ax.text(gmea + 8, 0.35, f'Mean\n{gmea:.0f} d',   color=H_GLOBAL, fontsize=8.5, va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_GLOBAL, alpha=0.85))
    ax.text(2,       gmea_s + 0.025, f'S({gmea:.0f}) = {gmea_s:.2f}',
            color=H_GLOBAL, fontsize=7.5, va='bottom')

    ax.text(vmed + 8, 0.38, f'Median\n{vmed:.0f} d', color=H_VT, fontsize=8.5, va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_VT, alpha=0.85))
    ax.text(vmea + 8, 0.20, f'Mean\n{vmea:.0f} d',   color=H_VT, fontsize=8.5, va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_VT, alpha=0.85))
    ax.text(2,       vmea_s + 0.025, f'S({vmea:.0f}) = {vmea_s:.2f}',
            color=H_VT, fontsize=7.5, va='bottom')

    ax.axhline(0.5, color='#777', lw=0.8, ls='--', alpha=0.4)
    ax.text(ax.get_xlim()[1] * 0.97 if ax.get_xlim()[1] > 0 else 900,
            0.515, 'P50', color='#777', fontsize=8.5, ha='right')

    # custom legend: KM step + Weibull line + marker legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg_handles = [
        Line2D([0], [0], color=H_GLOBAL, lw=1.5, alpha=0.6, label='Global excl. Vt — KM'),
        Line2D([0], [0], color=H_GLOBAL, lw=2.2, ls='--', label='Global excl. Vt — Weibull'),
        Line2D([0], [0], color=H_VT,     lw=1.5, alpha=0.6, label='Vt Field — KM'),
        Line2D([0], [0], color=H_VT,     lw=2.2, ls='--', label='Vt Field — Weibull'),
        Line2D([0], [0], color='k',      lw=1.3, ls=':',  label='Median (P50)  ●'),
        Line2D([0], [0], color='k',      lw=1.3, ls='-.',  label='Mean E[T]  ◆ → y-axis'),
    ]
    ax.legend(handles=leg_handles, fontsize=8.8, loc='upper right', framealpha=0.92, ncol=1)

    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Survival probability  S(t)', fontsize=12)
    ax.set_title('Kaplan-Meier + Weibull MLE — Raw (unfiltered) data',
                 fontsize=12, fontweight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    return fig_to_bytes(fig), fit_params


def fig_hazard_rate():
    """Hazard rate and cumulative hazard for Global vs Vt."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    t = np.linspace(1, 850, 600)

    for key, p in WBL.items():
        h_vals = wbl_h(t, p['beta'], p['eta']) * 1000  # ×10⁻³/day
        H_vals = wbl_H(t, p['beta'], p['eta'])
        ax1.plot(t, h_vals, color=p['color'], lw=2.2, ls=p['ls'], label=p['label'])
        ax2.plot(t, H_vals, color=p['color'], lw=2.2, ls=p['ls'], label=p['label'])

    ax1.set_xlabel('Time (days)', fontsize=11)
    ax1.set_ylabel('Hazard rate h(t)  [× 10⁻³ / day]', fontsize=11)
    ax1.set_title('Instantaneous Hazard Rate h(t)', fontweight='bold')
    ax1.legend(fontsize=9.5)
    ax1.set_xlim(0, 850)

    ax2.set_xlabel('Time (days)', fontsize=11)
    ax2.set_ylabel('Cumulative hazard  H(t) = –ln S(t)', fontsize=11)
    ax2.set_title('Cumulative Hazard H(t)', fontweight='bold')
    ax2.legend(fontsize=9.5)

    # Annotation for β interpretation
    ax1.text(700, ax1.get_ylim()[1] * 0.87,
             'β > 1 → increasing h(t)\n(wear-out regime)',
             fontsize=9, color='#444',
             bbox=dict(boxstyle='round', fc='#FFFDE7', ec='#F9A825', alpha=0.9))

    for ax in (ax1, ax2):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_rul_illustration():
    """Conditional survival and RUL bar chart for Vt field."""
    beta, eta = WBL['vt']['beta'], WBL['vt']['eta']
    tau  = np.linspace(0, 580, 500)
    ages = [0, 30, 90]
    cols = ['#1565C0', '#2E7D32', '#C62828']
    meds = [cond_median_rul(a, beta, eta) for a in ages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))

    # Left: conditional survival curves
    for t_age, c, rul in zip(ages, cols, meds):
        cs = cond_S(tau, t_age, beta, eta)
        ax1.plot(tau, cs, color=c, lw=2.3,
                 label=f'Age t = {t_age} d  →  RUL(P50) = {rul:.0f} d')
        ax1.axvline(rul, color=c, lw=1.2, ls=':', alpha=0.85)
        ax1.plot(rul, 0.5, 'o', color=c, ms=7, zorder=5)

    ax1.axhline(0.5, color='#888', lw=0.8, ls='--', alpha=0.45)
    ax1.text(490, 0.515, 'P50', color='#888', fontsize=9)
    ax1.set_xlabel('Remaining time τ (days)', fontsize=11)
    ax1.set_ylabel('Conditional survival  P(T > t + τ | T > t)', fontsize=11)
    ax1.set_title('Conditional Survival — Vt Field\n(β = 1.229,  η = 239 d)',
                  fontweight='bold', fontsize=10.5)
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.set_xlim(0, 560)
    ax1.set_ylim(0, 1.03)

    # Right: RUL bar chart
    bars = ax2.bar([f't = {a} d' for a in ages], meds,
                   color=cols, alpha=0.85, edgecolor='white', linewidth=1.8, width=0.5)
    ttf = wbl_median(beta, eta)
    ax2.axhline(ttf, color='#888', lw=1.3, ls='--', alpha=0.6)
    ax2.text(2.32, ttf + 2, f'TTF (t = 0) = {ttf:.0f} d', color='#666', fontsize=9.5)

    for bar, val in zip(bars, meds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f'{val:.0f} d', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax2.set_ylim(0, 230)
    ax2.set_ylabel('Conditional Median RUL (days)', fontsize=11)
    ax2.set_title('RUL Decreases with Age\n(wear-out: β > 1 → aging effect)',
                  fontweight='bold', fontsize=10.5)

    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_mean_vs_median():
    """Mean vs Median comparison with recommendation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    t = np.linspace(0.5, 1000, 600)

    for key, p in WBL.items():
        s   = wbl_S(t, p['beta'], p['eta'])
        med = wbl_median(p['beta'], p['eta'])
        mea = wbl_mean(p['beta'], p['eta'])
        ax1.plot(t, s, color=p['color'], lw=2.2, ls=p['ls'],
                 label=p['label'])
        ax1.axvline(med, color=p['color'], lw=1.4, ls=':')
        ax1.axvline(mea, color=p['color'], lw=1.4, ls='-.')

    ax1.axhline(0.5, color='#888', lw=0.8, ls='--', alpha=0.4)
    ax1.text(870, 0.515, 'P50', color='#888', fontsize=8.5)

    leg = [
        Line2D([0], [0], color=H_GLOBAL, lw=2.2,        label='Global  (β=1.059)'),
        Line2D([0], [0], color=H_VT,     lw=2.2, ls='--', label='Vt  (β=1.229)'),
        Line2D([0], [0], color='k',      lw=1.5, ls=':',  label='Median (P50)'),
        Line2D([0], [0], color='k',      lw=1.5, ls='-.', label='Mean E[T]'),
    ]
    ax1.legend(handles=leg, fontsize=9, loc='upper right', framealpha=0.9)
    ax1.set_xlabel('Time (days)', fontsize=11)
    ax1.set_ylabel('Survival  S(t)', fontsize=11)
    ax1.set_title('For β > 1:  Mean > Median\n(right-skewed survival, wear-out)', fontweight='bold', fontsize=10.5)
    ax1.set_xlim(0, 990)

    # Right: table
    ax2.axis('off')
    rows = []
    for key, p in WBL.items():
        med = wbl_median(p['beta'], p['eta'])
        mea = wbl_mean(p['beta'], p['eta'])
        rows.append([p['label'], f'β = {p["beta"]:.3f}',
                     f'{med:.0f} d', f'{mea:.0f} d', f'+{mea - med:.0f} d'])

    cols = ['Group', 'β', 'Median (P50)', 'Mean E[T]', 'Mean − Median']
    tbl  = ax2.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    style_table(tbl, len(rows))
    tbl.scale(1.1, 2.2)

    ax2.text(0.5, 0.08,
             'Recommendation: Median (P50)\n'
             '→ "50 % probability of failure within X days"\n'
             '→ Robust to heavy tails, directly actionable',
             transform=ax2.transAxes, ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=H_NAVY,
             bbox=dict(boxstyle='round,pad=0.5', fc='#EBF5FB', ec=H_GLOBAL, lw=1.5))

    ax2.set_title('  Summary — Mean vs. Median', loc='left', fontweight='bold', fontsize=10.5)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_chemical_effects():
    """Bar chart of chemical factor tertile effects on median TTF."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    factors = list(CHEM.keys())
    x = np.arange(len(factors))
    w = 0.24
    bin_cols  = ['#1976D2', '#43A047', '#E53935']
    bin_lbls  = ['Low bin', 'Mid bin', 'High bin']

    for i, (cat, c) in enumerate(zip(['low', 'mid', 'high'], bin_cols)):
        vals = [CHEM[f][cat] for f in factors]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=bin_lbls[i],
                      color=c, alpha=0.87, edgecolor='white', linewidth=1.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
                    str(v), ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # Approximate hazard ratio annotation
    for i, f in enumerate(factors):
        ratio = CHEM[f]['low'] / CHEM[f]['high']
        ax.text(i, CHEM[f]['low'] + 22,
                f'HR ≈ {ratio:.1f}×', ha='center',
                fontsize=9, color='#B71C1C',
                bbox=dict(boxstyle='round,pad=0.25', fc='#FFEBEE', ec='#B71C1C', alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(factors, fontsize=11)
    ax.set_ylabel('Median TTF (days)', fontsize=11)
    ax.set_title('Covariate Effects on Survival — Fleet Кэкспл > 0.5\n'
                 'All log-rank p < 0.01 · HR ≈ high-load vs. low-load hazard ratio',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 700)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_contractor():
    """Contractor comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=False)
    palette = [H_GLOBAL, '#00838F']

    for ax, (scope, data) in zip(axes, CONTRACTOR.items()):
        names = list(data.keys())
        vals  = list(data.values())
        bars  = ax.bar(names, vals, color=palette, alpha=0.87,
                       edgecolor='white', linewidth=1.5, width=0.45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                    f'{v} d', ha='center', va='bottom', fontsize=13, fontweight='bold')
        ratio = vals[1] / vals[0]
        pval  = '0.002' if scope == 'Fleet' else '0.07 (n.s.)'
        ax.text(0.5, 0.90, f'Ratio Sch/Bor = {ratio:.2f}  (p = {pval})',
                transform=ax.transAxes, ha='center', fontsize=10, color='#444',
                bbox=dict(boxstyle='round', fc='#F5F5F5', ec='#AAAAAA'))
        ax.set_title(f'{scope} — kekspl filtered', fontweight='bold', fontsize=10.5)
        ax.set_ylabel('Median TTF (days)', fontsize=11)
        ax.set_ylim(0, max(vals) * 1.3)

    fig.suptitle('Contractor Effect — Borec vs. Schlumberger\n'
                 '(New Technologies excluded: median ~89 d fleet, ~2.5–3× worse than Borec)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_latent_vt():
    """Latent 2-component mixture — Vt field results."""
    lp = LATENT['vt']
    c1, c2 = lp['comp1'], lp['comp2']
    t = np.linspace(0.5, 900, 600)

    s_mix = mix_S(t, c1, c2)
    s1w   = c1['w'] * wbl_S(t, c1['beta'], c1['eta'])
    s2w   = c2['w'] * wbl_S(t, c2['beta'], c2['eta'])

    med_mix  = lp['B50']
    mean_mix = mix_mean(c1, c2)
    med_c1   = wbl_median(c1['beta'], c1['eta'])
    mean_c1  = wbl_mean(c1['beta'], c1['eta'])
    med_c2   = wbl_median(c2['beta'], c2['eta'])
    mean_c2  = wbl_mean(c2['beta'], c2['eta'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))

    # ── Left: survival curves ──────────────────────────────────────────────────
    ax1.plot(t, s_mix, color=H_VT, lw=2.8,
             label=f'Mixture  (n={lp["n"]})')
    ax1.plot(t, s1w,   color=H_COMP1, lw=1.8, ls='--',
             label=f'Comp 1 (early, w={c1["w"]:.2f}, β={c1["beta"]:.2f}, η={c1["eta"]} d)')
    ax1.plot(t, s2w,   color=H_COMP2, lw=1.8, ls=':',
             label=f'Comp 2 (wear-out, w={c2["w"]:.2f}, β={c2["beta"]:.2f}, η={c2["eta"]} d)')

    ax1.axvline(med_mix,  color=H_VT,    lw=1.5, ls=':',  alpha=0.85)
    ax1.axvline(mean_mix, color=H_PURPLE, lw=1.5, ls='-.', alpha=0.85)
    ax1.text(med_mix  + 6, 0.58, f'Median\n{med_mix:.0f} d',  color=H_VT,    fontsize=9,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_VT,    alpha=0.85))
    ax1.text(mean_mix + 6, 0.43, f'Mean\n{mean_mix:.0f} d',   color=H_PURPLE, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_PURPLE, alpha=0.85))

    ax1.axhline(0.5, color='#888', lw=0.8, ls='--', alpha=0.4)
    ax1.text(820, 0.515, 'P50', color='#888', fontsize=8.5)
    ax1.set_xlabel('Time (days)', fontsize=11)
    ax1.set_ylabel('Survival  S(t)', fontsize=11)
    ax1.set_title('Latent 2-Component Weibull — Vt Field\n'
                  '(Bayesian posterior, all Vt runs)', fontweight='bold', fontsize=10.5)
    ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax1.set_xlim(0, 880)
    ax1.set_ylim(0, 1.03)

    # ── Right: parameter table ──────────────────────────────────────────────────
    ax2.axis('off')

    rows = [
        ['Comp 1\n(early)', f'{c1["w"]:.2f}', f'{c1["beta"]:.3f}',
         f'{c1["eta"]} d', f'{med_c1:.0f} d', f'{mean_c1:.0f} d'],
        ['Comp 2\n(wear-out)', f'{c2["w"]:.2f}', f'{c2["beta"]:.3f}',
         f'{c2["eta"]} d', f'{med_c2:.0f} d', f'{mean_c2:.0f} d'],
        ['Mixture', '1.00', '—', '—',
         f'{med_mix:.0f} d', f'{mean_mix:.0f} d'],
    ]
    col_lbl = ['', 'Weight π', 'Shape β', 'Scale η', 'Median\n(P50)', 'Mean\nE[T]']
    tbl = ax2.table(cellText=rows, colLabels=col_lbl, loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.05, 2.1)
    row_colors = [H_NAVY, H_COMP1, H_COMP2, H_VT]
    bg_colors  = ['white', '#FFF3E0', '#E8F5E9', '#FFEBEE']
    for (r, c_), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(H_NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        elif r <= 3:
            cell.set_facecolor(bg_colors[r])

    # Quantile boxes
    quantiles = [('B10', lp['B10']), ('B25', lp['B25']), ('B50', lp['B50']), ('B90', lp['B90'])]
    q_cols = ['#FFCCBC', '#FFF9C4', '#C8E6C9', '#BBDEFB']
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    for j, ((lbl, val), qc) in enumerate(zip(quantiles, q_cols)):
        x0 = 0.04 + j * 0.24
        ax2.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.03), 0.20, 0.16,
            boxstyle='round,pad=0.01', fc=qc, ec='#999999', lw=1.2))
        ax2.text(x0 + 0.10, 0.14, lbl, ha='center', va='center',
                 fontsize=9, fontweight='bold', color='#333')
        ax2.text(x0 + 0.10, 0.06, f'{val:.0f} d', ha='center', va='center',
                 fontsize=10, color='#333')

    ax2.set_title('  Vt Mixture — Parameters & Fleet Quantiles',
                  loc='left', fontweight='bold', fontsize=10.5)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_cox_cl_km():
    """KM + Weibull for actual Cl tertile bins (queried from DB)."""
    import sqlite3
    import pandas as pd
    from lifelines import KaplanMeierFitter, WeibullFitter

    db = Path(REPO_ROOT) / 'data' / 'warehouse' / 'pump2.db'
    con = sqlite3.connect(str(db))
    df = pd.read_sql(
        "SELECT ttf_true_best_days AS ttf, event, cum_chloride_load_kg, run_days "
        "FROM mart__vt_freq55 "
        "WHERE ttf_true_best_days > 0 AND cum_chloride_load_kg IS NOT NULL",
        con,
    )
    con.close()
    df['cl_day'] = df['cum_chloride_load_kg'] / df['run_days']
    q33 = df['cl_day'].quantile(0.333)
    q67 = df['cl_day'].quantile(0.667)
    df['bin'] = pd.cut(df['cl_day'], [-np.inf, q33, q67, np.inf],
                       labels=['Low', 'Mid', 'High'])

    bin_cfg = {
        'Low':  ('#1565C0', f'< {q33/1000:.0f}k kg/d'),
        'Mid':  ('#43A047', f'{q33/1000:.0f}k–{q67/1000:.0f}k kg/d'),
        'High': ('#C62828', f'> {q67/1000:.0f}k kg/d'),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))

    hr_rows = []
    fitted = {}
    for bname, (c, rng) in bin_cfg.items():
        sub = df[df['bin'] == bname]
        T, E = sub['ttf'], sub['event']

        kmf = KaplanMeierFitter()
        kmf.fit(T, E)
        km_t = kmf.survival_function_.index.values
        km_s = kmf.survival_function_.iloc[:, 0].values
        ax1.step(km_t, km_s, color=c, lw=1.4, alpha=0.50, where='post')

        wf = WeibullFitter(); wf.fit(T, E)
        beta, eta = wf.rho_, wf.lambda_
        med = wbl_median(beta, eta)
        fitted[bname] = dict(beta=beta, eta=eta, med=med, n=len(T), ev=int(E.sum()))

        t_pl = np.linspace(0.5, 1200, 500)
        ax1.plot(t_pl, wbl_S(t_pl, beta, eta), color=c, lw=2.2,
                 label=f'Cl {bname} ({rng})\nn={len(T)}, med={med:.0f} d, β={beta:.2f}')
        ax1.axvline(med, color=c, lw=1.0, ls=':', alpha=0.65)

        hr_rows.append([bname, rng, f'{len(T)}', f'{med:.0f} d', f'{beta:.3f}'])

    ax1.axhline(0.5, color='#777', lw=0.8, ls='--', alpha=0.4)
    ax1.set_xlabel('Time (days)', fontsize=11)
    ax1.set_ylabel('Survival  S(t)', fontsize=11)
    ax1.set_title('Kaplan-Meier (step) + Weibull MLE (line)\nChloride daily load tertiles — fleet raw',
                  fontweight='bold', fontsize=10.5)
    ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.92)
    ax1.set_xlim(0, 1200)

    # Right: HR table
    ax2.axis('off')
    hr_med_ratio = fitted['Low']['med'] / fitted['High']['med']
    rows2 = hr_rows + [['HR (Low/High)', '—', '—',
                         f'≈ {hr_med_ratio:.1f}×', '']]
    tbl = ax2.table(
        cellText=rows2,
        colLabels=['Bin', 'Cl range', 'n', 'Median TTF', 'β'],
        loc='center', cellLoc='center',
    )
    style_table(tbl, len(rows2))
    tbl.scale(1.1, 2.1)
    # highlight HR row
    for c_ in range(5):
        tbl[(len(rows2), c_)].set_facecolor('#FFEBEE')

    ax2.set_title('  Cl load → strongest monotone trend\n  HR ≈ median(Low) / median(High)',
                  loc='left', fontweight='bold', fontsize=10.5)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_latent_vt_km():
    """Latent 2-comp Weibull for Vt: KM + mixture + individual (unscaled) components."""
    import sqlite3
    import pandas as pd
    from lifelines import KaplanMeierFitter

    db = Path(REPO_ROOT) / 'data' / 'warehouse' / 'pump2.db'
    con = sqlite3.connect(str(db))
    df = pd.read_sql(
        "SELECT ttf_true_best_days AS ttf, event FROM mart__vt_freq55 "
        "WHERE field='Vt' AND ttf_true_best_days > 0",
        con,
    )
    con.close()

    kmf = KaplanMeierFitter()
    kmf.fit(df['ttf'], df['event'])
    km_t = kmf.survival_function_.index.values
    km_s = kmf.survival_function_.iloc[:, 0].values

    lp = LATENT['vt']
    c1, c2 = lp['comp1'], lp['comp2']
    t = np.linspace(0.5, 900, 600)

    s_mix = mix_S(t, c1, c2)
    s1_raw = wbl_S(t, c1['beta'], c1['eta'])   # unscaled: full pop-1 survival
    s2_raw = wbl_S(t, c2['beta'], c2['eta'])   # unscaled: full pop-2 survival

    med_mix  = lp['B50']
    mean_mix = mix_mean(c1, c2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ── Left: KM + mixture ────────────────────────────────────────────────────
    ax1.step(km_t, km_s, color='#444', lw=1.6, alpha=0.55, where='post',
             label=f'KM — Vt raw  (n={len(df)}, ev={int(df.event.sum())})')
    ax1.plot(t, s_mix, color=H_VT, lw=2.6, label='2-comp mixture  S(t)')

    ax1.axvline(med_mix,  color=H_VT,    lw=1.5, ls=':',  alpha=0.85)
    ax1.axvline(mean_mix, color=H_PURPLE, lw=1.5, ls='-.', alpha=0.85)
    ax1.text(med_mix  + 6, 0.57, f'Median\n{med_mix:.0f} d',  color=H_VT,    fontsize=9,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_VT,    alpha=0.85))
    ax1.text(mean_mix + 6, 0.38, f'Mean\n{mean_mix:.0f} d',   color=H_PURPLE, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=H_PURPLE, alpha=0.85))
    ax1.axhline(0.5, color='#888', lw=0.8, ls='--', alpha=0.4)
    ax1.set_xlabel('Time (days)', fontsize=11)
    ax1.set_ylabel('Survival  S(t)', fontsize=11)
    ax1.set_title('KM vs. Latent Weibull Mixture — Vt Field', fontweight='bold', fontsize=10.5)
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.92)
    ax1.set_xlim(0, 880); ax1.set_ylim(0, 1.03)

    # ── Right: unscaled component survival curves (the actual split) ──────────
    med_c1 = wbl_median(c1['beta'], c1['eta'])
    med_c2 = wbl_median(c2['beta'], c2['eta'])
    ax2.step(km_t, km_s, color='#444', lw=1.4, alpha=0.45, where='post', label='KM (all Vt)')
    ax2.plot(t, s1_raw, color=H_COMP1, lw=2.2, ls='--',
             label=f'Pop-1 S₁(t): {c1["w"]*100:.0f}% of pumps\n'
                   f'β={c1["beta"]:.2f}, η={c1["eta"]} d, med={med_c1:.0f} d')
    ax2.plot(t, s2_raw, color=H_COMP2, lw=2.2, ls=':',
             label=f'Pop-2 S₂(t): {c2["w"]*100:.0f}% of pumps\n'
                   f'β={c2["beta"]:.2f}, η={c2["eta"]} d, med={med_c2:.0f} d')
    ax2.plot(t, s_mix,  color=H_VT,    lw=1.8, alpha=0.7,
             label=f'Mixture = {c1["w"]:.2f}·S₁ + {c2["w"]:.2f}·S₂')

    ax2.axhline(0.5, color='#888', lw=0.8, ls='--', alpha=0.4)
    ax2.set_xlabel('Time (days)', fontsize=11)
    ax2.set_ylabel('Survival  S(t)', fontsize=11)
    ax2.set_title('Latent subpopulation split\n(each curve starts at 1.0 — shows its own failure pattern)',
                  fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8.2, loc='upper right', framealpha=0.92)
    ax2.set_xlim(0, 880); ax2.set_ylim(0, 1.03)

    fig.tight_layout()
    return fig_to_bytes(fig)


# ─── Section 1 extra figures ─────────────────────────────────────────────────

_DB_PATH_S1 = Path(REPO_ROOT) / 'data' / 'warehouse' / 'pump2.db'

_FIELD_CFG = {          # major fields only (n ≥ 50)
    'Ya': ('#1565C0', '-'),
    'Vt': ('#C62828', '--'),
    'Az': ('#2E7D32', '-'),
    'Za': ('#E65100', '--'),
    'Ic': ('#6A1B9A', '-'),
    'Mc': ('#00838F', '--'),
    'Da': ('#795548', ':'),
}

_FAILNODE_EN = {
    'ЭЦН':                   ('ESP Pump',            '#1565C0'),
    'Кабельная линия':        ('Power Cable',         '#C62828'),
    'ПЭД':                   ('ESP Motor',            '#2E7D32'),
    'Гидрозащита':            ('Protector / Seal',    '#E65100'),
    'НКТ':                   ('Production Tubing',    '#6A1B9A'),
    'Газосепаратор':          ('Gas Separator',       '#00838F'),
    'Клапан сливной':         ('Drain Valve',         '#795548'),
    'Диспергатор':            ('Disperser',           '#F57C00'),
    'ТМС':                   ('TMS Sensor',           '#9E9E9E'),
}


def fig_field_km():
    """KM + Weibull by field — raw (left) and utilisation≥0.5 (right)."""
    import sqlite3, pandas as pd
    from lifelines import KaplanMeierFitter, WeibullFitter
    con = sqlite3.connect(str(_DB_PATH_S1))
    df = pd.read_sql(
        "SELECT field, ttf_true_best_days AS ttf, event, run_days "
        "FROM mart__vt_freq55 WHERE ttf_true_best_days > 0 AND run_days > 0",
        con)
    con.close()
    df['kekspl'] = df['ttf'] / df['run_days']

    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    t_max = 900
    t_wbl = np.linspace(0.5, t_max, 500)

    for fld, (col, ls) in _FIELD_CFG.items():
        for ax, filtered in [(ax_r, False), (ax_f, True)]:
            sub = df[df['field'] == fld]
            if filtered:
                sub = sub[sub['kekspl'] >= 0.5]
            if len(sub) < 10:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub['ttf'], event_observed=sub['event'])
            wf = WeibullFitter()
            wf.fit(sub['ttf'], event_observed=sub['event'])
            beta, eta = wf.rho_, wf.lambda_
            med = kmf.median_survival_time_
            med_s = f'{int(med)} d' if np.isfinite(med) else '—'
            lw = 2.1 if fld in ('Vt', 'Ya') else 1.6
            # KM step
            t = kmf.timeline
            s = kmf.survival_function_.iloc[:, 0]
            m = t <= t_max
            ax.plot(t[m], s[m], color=col, lw=lw, ls=ls,
                    label=f'{fld}  n={len(sub)},  β={beta:.2f},  med={med_s}')
            # Weibull overlay (dotted, same colour)
            ax.plot(t_wbl, wbl_S(t_wbl, beta, eta),
                    color=col, lw=1.0, ls=':', alpha=0.75)
            if np.isfinite(med) and med <= t_max:
                ax.axvline(med, color=col, lw=0.7, ls=':', alpha=0.45)

    for ax, title in [(ax_r, 'Raw TTF (all runs)'),
                      (ax_f, 'Stable runs  (utilisation factor ≥ 0.5)')]:
        ax.set_xlim(0, t_max); ax.set_ylim(0, 1.03)
        ax.axhline(0.5, color='#bbb', lw=0.8, ls='--', alpha=0.6)
        ax.set_xlabel('Days', fontsize=10); ax.set_ylabel('Survival  S(t)', fontsize=10)
        ax.legend(fontsize=8.5, loc='upper right', framealpha=0.93)
        ax.set_title(title, fontweight='bold', fontsize=10.5)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_failure_cat_km():
    """Cause-specific KM + Weibull by failure node — 2 panels: Vt and Global excl. Vt."""
    import sqlite3, pandas as pd
    from lifelines import KaplanMeierFitter, WeibullFitter
    con = sqlite3.connect(str(_DB_PATH_S1))
    df = pd.read_sql(
        "SELECT field, failed_node, ttf_true_best_days AS ttf, event "
        "FROM mart__vt_freq55 WHERE ttf_true_best_days > 0",
        con)
    con.close()

    panels = [
        (df['field'] == 'Vt',  'Vt Field',       ['ЭЦН', 'Кабельная линия', 'ПЭД', 'Гидрозащита']),
        (df['field'] != 'Vt',  'Global excl. Vt', ['ЭЦН', 'Кабельная линия', 'ПЭД', 'Гидрозащита',
                                                    'НКТ', 'Газосепаратор', 'Клапан сливной']),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    t_max = 1400
    t_wbl = np.linspace(0.5, t_max, 600)

    for ax, (mask, title, nodes) in zip(axes, panels):
        sub_all = df[mask]
        for node in nodes:
            en, col = _FAILNODE_EN.get(node, (node, '#999'))
            ev_cs = (sub_all['failed_node'] == node).astype(int)
            n_ev = ev_cs.sum()
            if n_ev < 3:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub_all['ttf'], event_observed=ev_cs)
            wf = WeibullFitter()
            wf.fit(sub_all['ttf'], event_observed=ev_cs)
            beta, eta = wf.rho_, wf.lambda_
            med = kmf.median_survival_time_
            med_s = f'{int(med)} d' if np.isfinite(med) else '—'
            t = kmf.timeline
            s = kmf.survival_function_.iloc[:, 0]
            m = t <= t_max
            # KM
            ax.plot(t[m], s[m], color=col, lw=1.9,
                    label=f'{en}  ev={n_ev},  β={beta:.2f},  med={med_s}')
            # Weibull overlay
            ax.plot(t_wbl, wbl_S(t_wbl, beta, eta),
                    color=col, lw=1.0, ls=':', alpha=0.80)
            if np.isfinite(med) and med <= t_max:
                ax.axvline(med, color=col, lw=0.7, ls=':', alpha=0.45)

        ax.set_xlim(0, t_max); ax.set_ylim(0, 1.03)
        ax.axhline(0.5, color='#bbb', lw=0.8, ls='--', alpha=0.6)
        ax.set_xlabel('Days', fontsize=10); ax.set_ylabel('Survival  S(t)', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=10.5)
        ax.legend(fontsize=8.2, loc='upper right', framealpha=0.93)

    fig.suptitle('Cause-Specific KM + Weibull by Failure Node  (all runs, raw TTF)',
                 fontsize=11.5, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_gypsum_km():
    """KM by gypsum proxy tertile — expected null result."""
    import sqlite3, pandas as pd
    from lifelines import KaplanMeierFitter
    con = sqlite3.connect(str(_DB_PATH_S1))
    df = pd.read_sql(
        "SELECT ttf_true_best_days AS ttf, event, run_days, cum_gypsum_scale_proxy "
        "FROM mart__vt_freq55 "
        "WHERE ttf_true_best_days > 0 AND run_days > 0 AND cum_gypsum_scale_proxy IS NOT NULL",
        con)
    con.close()
    df['kekspl'] = df['ttf'] / df['run_days']
    df = df[df['kekspl'] >= 0.5]
    df['rate'] = df['cum_gypsum_scale_proxy'] / df['run_days']

    q33 = df['rate'].quantile(0.333)
    q67 = df['rate'].quantile(0.667)
    df['bin'] = pd.cut(df['rate'], [-np.inf, q33, q67, np.inf],
                       labels=['Low', 'Mid', 'High'])

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for bname, col in [('Low', _C_LOW), ('Mid', _C_MID), ('High', _C_HIGH)]:
        sub = df[df['bin'] == bname]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['ttf'], event_observed=sub['event'])
        med = kmf.median_survival_time_
        med_s = f'{int(med)} d' if np.isfinite(med) else '—'
        t = kmf.timeline
        s = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_survival_function_
        m = t <= 900
        ax.plot(t[m], s[m], color=col, lw=2.1,
                label=f'Gypsum {bname}  n={len(sub)},  med={med_s}')
        ax.fill_between(t[m], ci.iloc[:, 0][m], ci.iloc[:, 1][m], color=col, alpha=0.10)
        if np.isfinite(med) and med <= 900:
            ax.axvline(med, color=col, lw=0.8, ls=':', alpha=0.65)

    _style_ax(ax)
    ax.set_title('KM Survival by Gypsum Proxy tertile  —  Кэкспл ≥ 0.5',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig_to_bytes(fig)


# ─── Frequency analysis figures ──────────────────────────────────────────────

_FREQ_GRP = [
    # (label,                        colour,    lw,  ls)
    ('UHF-60  mean >58 Hz',          '#C62828', 2.6, '-'),
    ('HF  >50% time >55 Hz',         '#E65100', 2.0, '--'),
    ('Base  50–55 Hz',               '#1565C0', 2.2, '-'),
    ('LF  mean <45 Hz',              '#6A1B9A', 1.8, ':'),
    ('Mid  45–50 Hz',                '#9E9E9E', 1.4, '--'),
]


def _assign_freq_groups(df):
    """Assign mutually exclusive frequency groups (priority order matches report)."""
    df = df.dropna(subset=['freq_w_mean']).copy()
    df['grp'] = 'Mid  45–50 Hz'
    df.loc[df['freq_w_mean'] < 45, 'grp'] = 'LF  mean <45 Hz'
    df.loc[
        (df['freq_w_mean'] >= 50) & (df['freq_w_mean'] <= 55) &
        (df['freq_above_55hz_pct'].fillna(0) <= 0.5),
        'grp'
    ] = 'Base  50–55 Hz'
    df.loc[
        (df['freq_above_55hz_pct'].fillna(0) > 0.5) & (df['freq_w_mean'] <= 58),
        'grp'
    ] = 'HF  >50% time >55 Hz'
    df.loc[df['freq_w_mean'] > 58, 'grp'] = 'UHF-60  mean >58 Hz'
    return df


def _plot_freq_panel(ax, df, t_max=900, show_mid=False):
    """KM + Weibull for each freq group on ax; returns dict of medians."""
    from lifelines import KaplanMeierFitter, WeibullFitter
    t_wbl = np.linspace(0.5, t_max, 500)
    medians = {}
    for lbl, col, lw, ls in _FREQ_GRP:
        sub = df[df['grp'] == lbl]
        if len(sub) < 5 or (lbl.startswith('Mid') and not show_mid):
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['ttf'], event_observed=sub['event'])
        wf = WeibullFitter()
        wf.fit(sub['ttf'], event_observed=sub['event'])
        beta, eta = wf.rho_, wf.lambda_
        med = kmf.median_survival_time_
        med_s = f'{int(med)} d' if np.isfinite(med) else '—'
        medians[lbl] = med
        t = kmf.timeline
        s = kmf.survival_function_.iloc[:, 0]
        m = t <= t_max
        ax.plot(t[m], s[m], color=col, lw=lw, ls=ls,
                label=f'{lbl}  n={len(sub)},  β={beta:.2f},  med={med_s}')
        ax.plot(t_wbl, wbl_S(t_wbl, beta, eta), color=col, lw=1.0, ls=':', alpha=0.75)
        if np.isfinite(med) and med <= t_max:
            ax.axvline(med, color=col, lw=0.7, ls=':', alpha=0.45)
    ax.set_xlim(0, t_max); ax.set_ylim(0, 1.03)
    ax.axhline(0.5, color='#bbb', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Days (TTF true)', fontsize=10)
    ax.set_ylabel('Survival  S(t)', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.93)
    return medians


def fig_freq_groups_km():
    """Fleet (left) and Vt (right) KM+Weibull by frequency group."""
    import sqlite3, pandas as pd
    con = sqlite3.connect(str(_DB_PATH_S1))
    df_raw = pd.read_sql(
        "SELECT field, ttf_true_best_days AS ttf, event, "
        "freq_w_mean, freq_above_55hz_pct "
        "FROM mart__vt_freq55 WHERE ttf_true_best_days > 0",
        con)
    con.close()
    df = _assign_freq_groups(df_raw)

    fig, (ax_fl, ax_vt) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _plot_freq_panel(ax_fl, df, t_max=900)
    ax_fl.set_title('Full Fleet  —  all runs, TTF true', fontweight='bold', fontsize=10.5)

    df_vt = df[df['field'] == 'Vt']
    _plot_freq_panel(ax_vt, df_vt, t_max=700)
    ax_vt.set_title('Vt Field  —  all runs, TTF true\n(UHF-60 n=9: indicative only)',
                    fontweight='bold', fontsize=10)

    fig.suptitle('KM + Weibull Survival by Frequency Group', fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_freq_stability_km():
    """UHF-60 vs Base split by utilisation stability (KIP <0.65 vs ≥0.65)."""
    import sqlite3, pandas as pd
    con = sqlite3.connect(str(_DB_PATH_S1))
    df_raw = pd.read_sql(
        "SELECT ttf_true_best_days AS ttf, event, run_days, "
        "freq_w_mean, freq_above_55hz_pct "
        "FROM mart__vt_freq55 WHERE ttf_true_best_days > 0 AND run_days > 0",
        con)
    con.close()
    df = _assign_freq_groups(df_raw)
    df['kip'] = df['ttf'] / df['run_days']

    # keep only Base and UHF-60 for clarity
    df_sub = df[df['grp'].isin(['Base  50–55 Hz', 'UHF-60  mean >58 Hz'])].copy()

    fig, (ax_un, ax_st) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # RMST table values from report (horizon = 817 d)
    rmst_table = {
        'stable':   {'Base': 389, 'UHF-60': 327, 'delta': -16},
        'unstable': {'Base': 308, 'UHF-60': 207, 'delta': -33},
    }

    for ax, kip_mask, title, regime_key in [
        (ax_un, df_sub['kip'] < 0.65,  'Unstable  (utilisation < 0.65)\ncycling / frequent stops', 'unstable'),
        (ax_st, df_sub['kip'] >= 0.65, 'Stable  (utilisation ≥ 0.65)\ncontinuous operation',      'stable'),
    ]:
        from lifelines import KaplanMeierFitter, WeibullFitter
        t_wbl = np.linspace(0.5, 800, 500)
        for lbl, col, lw, ls in [
            ('UHF-60  mean >58 Hz', '#C62828', 2.6, '-'),
            ('Base  50–55 Hz',      '#1565C0', 2.0, '--'),
        ]:
            sub = df_sub[kip_mask & (df_sub['grp'] == lbl)]
            if len(sub) < 5:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub['ttf'], event_observed=sub['event'])
            wf = WeibullFitter()
            wf.fit(sub['ttf'], event_observed=sub['event'])
            beta, eta = wf.rho_, wf.lambda_
            med = kmf.median_survival_time_
            med_s = f'{int(med)} d' if np.isfinite(med) else '—'
            t = kmf.timeline
            s = kmf.survival_function_.iloc[:, 0]
            m = t <= 800
            grp_short = 'UHF-60' if 'UHF' in lbl else 'Base'
            ax.plot(t[m], s[m], color=col, lw=lw, ls=ls,
                    label=f'{grp_short}  n={len(sub)},  β={beta:.2f},  med={med_s}')
            ax.plot(t_wbl, wbl_S(t_wbl, beta, eta), color=col, lw=1.0, ls=':', alpha=0.75)
            if np.isfinite(med) and med <= 800:
                ax.axvline(med, color=col, lw=0.7, ls=':', alpha=0.45)

        r = rmst_table[regime_key]
        ax.text(0.97, 0.42,
                f'RMST (horizon 817 d):\n'
                f'  Base:    {r["Base"]} d\n'
                f'  UHF-60: {r["UHF-60"]} d  ({r["delta"]:+d}%)',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9.5, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='#F4F6FA', ec='#CCCCCC', lw=1.0))

        ax.set_xlim(0, 800); ax.set_ylim(0, 1.03)
        ax.axhline(0.5, color='#bbb', lw=0.8, ls='--', alpha=0.6)
        ax.set_xlabel('Days (TTF true)', fontsize=10)
        ax.set_ylabel('Survival  S(t)', fontsize=10)
        ax.legend(fontsize=9.5, loc='upper right', framealpha=0.93)
        ax.set_title(title, fontweight='bold', fontsize=10.5)

    fig.suptitle('UHF-60 Effect by Operating Stability  (Fleet, TTF true)',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig_to_bytes(fig)


# ─── Section 2: Filtering & Covariate figures ────────────────────────────────

_DB_PATH = Path(REPO_ROOT) / 'data' / 'warehouse' / 'pump2.db'

_CONT_EN = {
    'Борец':            ('Borec',         '#1565C0'),
    'Шлюмберже':        ('Schlumberger',  '#E65100'),
    'Новые технологии': ('NT — New Tech', '#C62828'),
    'Новомет':          ('Novomet',       '#6A1B9A'),
    'ИНК':              ('INK',           '#757575'),
}

_C_LOW  = '#2E7D32'
_C_MID  = '#F57C00'
_C_HIGH = '#C62828'


def _load_mart_s2(extra=''):
    import sqlite3, pandas as pd
    cols = 'ttf_true_best_days AS ttf, event, run_days, field, contractor'
    if extra:
        cols += ', ' + extra
    con = sqlite3.connect(str(_DB_PATH))
    df = pd.read_sql(
        f"SELECT {cols} FROM mart__vt_freq55 "
        "WHERE ttf_true_best_days > 0 AND run_days > 0", con)
    con.close()
    df['kekspl'] = df['ttf'] / df['run_days']
    return df


def _km_curve(ax, df, label, color, t_max=900, lw=2.0):
    """Fit KM on df, plot on ax, return median."""
    from lifelines import KaplanMeierFitter
    if len(df) < 5:
        return np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(df['ttf'], event_observed=df['event'])
    t = kmf.timeline
    s = kmf.survival_function_.iloc[:, 0]
    ci = kmf.confidence_interval_survival_function_
    m = t <= t_max
    ax.plot(t[m], s[m], color=color, lw=lw, label=label)
    ax.fill_between(t[m], ci.iloc[:, 0][m], ci.iloc[:, 1][m], color=color, alpha=0.10)
    med = kmf.median_survival_time_
    if np.isfinite(med) and med <= t_max:
        ax.axvline(med, color=color, lw=0.8, ls=':', alpha=0.65)
    return med


def _style_ax(ax, t_max=900):
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.03)
    ax.axhline(0.5, color='#bbb', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Days', fontsize=10)
    ax.set_ylabel('Survival  S(t)', fontsize=10)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.93)


def fig_contractor_km():
    """KM by contractor — raw TTF, NT highlighted."""
    df = _load_mart_s2()
    fig, (ax_raw, ax_fil) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ru, (en, col) in _CONT_EN.items():
        sub_r = df[df['contractor'] == ru]
        sub_f = df[(df['contractor'] == ru) & (df['kekspl'] >= 0.5)]
        lw = 2.8 if 'NT' in en else 1.9
        for ax, sub, suffix in [(ax_raw, sub_r, ''), (ax_fil, sub_f, '')]:
            if len(sub) < 3:
                continue
            from lifelines import KaplanMeierFitter
            kmf = KaplanMeierFitter()
            kmf.fit(sub['ttf'], event_observed=sub['event'])
            med = kmf.median_survival_time_
            med_s = f'{int(med)} d' if np.isfinite(med) else '—'
            lbl = f'{en}  n={len(sub)},  med={med_s}'
            t = kmf.timeline
            s = kmf.survival_function_.iloc[:, 0]
            ci = kmf.confidence_interval_survival_function_
            m = t <= 900
            ax.plot(t[m], s[m], color=col, lw=lw, label=lbl)
            ax.fill_between(t[m], ci.iloc[:, 0][m], ci.iloc[:, 1][m], color=col, alpha=0.09)
            if np.isfinite(med) and med <= 900:
                ax.axvline(med, color=col, lw=0.8, ls=':', alpha=0.6)
    for ax, ttl in [(ax_raw, 'All runs (raw)'), (ax_fil, 'Stable runs (Кэкспл ≥ 0.5)')]:
        _style_ax(ax)
        ax.set_title(ttl, fontweight='bold', fontsize=10.5)
    fig.suptitle('KM Survival by Contractor', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_kekspl_filter():
    """2-panel: Global (excl. Vt) and Vt — raw vs Кэкспл≥0.5 KM + Weibull fits."""
    from lifelines import KaplanMeierFitter, WeibullFitter
    df = _load_mart_s2()

    configs = [
        (df['field'] != 'Vt', 'Global Fleet (excl. Vt)', H_GLOBAL, '#90CAF9'),
        (df['field'] == 'Vt', 'Vt Field',                H_VT,     '#FFCDD2'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, (mask, title, c_fil, c_raw) in zip(axes, configs):
        df_r = df[mask]
        df_f = df[mask & (df['kekspl'] >= 0.5)]

        for sub, color, lw, ls, lbl_pfx in [
            (df_r, c_raw, 1.6, '--', 'Raw'),
            (df_f, c_fil, 2.2, '-',  'Кэкспл ≥ 0.5'),
        ]:
            kmf = KaplanMeierFitter()
            kmf.fit(sub['ttf'], event_observed=sub['event'])
            wf = WeibullFitter()
            wf.fit(sub['ttf'], event_observed=sub['event'])
            beta = wf.rho_
            med = kmf.median_survival_time_
            med_s = f'{int(med)} d' if np.isfinite(med) else '—'
            lbl = f'{lbl_pfx}  (n={len(sub)},  β={beta:.2f},  med={med_s})'
            t = kmf.timeline
            s = kmf.survival_function_.iloc[:, 0]
            ci = kmf.confidence_interval_survival_function_
            m = t <= 1000
            ax.plot(t[m], s[m], color=color, lw=lw, ls=ls, label=lbl)
            if ls == '-':
                ax.fill_between(t[m], ci.iloc[:, 0][m], ci.iloc[:, 1][m], color=color, alpha=0.12)
                # Weibull fit overlay
                t_wbl = np.linspace(0, 1000, 600)
                ax.plot(t_wbl, wf_S := wbl_S(t_wbl, wf.rho_, wf.lambda_),
                        color=color, lw=1.2, ls=':', alpha=0.8)
            if np.isfinite(med) and med <= 1000:
                ax.axvline(med, color=color, lw=0.8, ls=':', alpha=0.6)

        _style_ax(ax, t_max=1000)
        ax.set_title(title, fontweight='bold', fontsize=10.5)

    fig.suptitle('Effect of Кэкспл ≥ 0.5 filter on survival shape',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_glf_km():
    """KM survival by avg_glf bin — shows U-shaped effect."""
    from lifelines import KaplanMeierFitter
    import sqlite3, pandas as pd
    con = sqlite3.connect(str(_DB_PATH))
    df = pd.read_sql(
        "SELECT ttf_true_best_days AS ttf, event, avg_glf "
        "FROM mart__vt_freq55 WHERE ttf_true_best_days > 0 AND avg_glf IS NOT NULL",
        con)
    con.close()

    cuts  = [-np.inf, 100, 300, 700, np.inf]
    lbls  = ['0–100  (low sep.)', '100–300', '300–700', '700+  (best)']
    cols  = ['#C62828', '#F57C00', '#1565C0', '#2E7D32']
    df['bin'] = pd.cut(df['avg_glf'], cuts, labels=lbls)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for lbl, col in zip(lbls, cols):
        sub = df[df['bin'] == lbl]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['ttf'], event_observed=sub['event'])
        med = kmf.median_survival_time_
        med_s = f'{int(med)} d' if np.isfinite(med) else '—'
        t = kmf.timeline
        s = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_survival_function_
        m = t <= 900
        ax.plot(t[m], s[m], color=col, lw=2.0,
                label=f'GLF {lbl}  n={len(sub)},  med={med_s}')
        ax.fill_between(t[m], ci.iloc[:, 0][m], ci.iloc[:, 1][m], color=col, alpha=0.09)
        if np.isfinite(med) and med <= 900:
            ax.axvline(med, color=col, lw=0.8, ls=':', alpha=0.65)

    _style_ax(ax)
    ax.set_title('KM Survival by Gas-Liquid Factor (GLF) bin  —  raw TTF, all runs',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_h2s_km():
    """KM by H2S concentration bin — threshold effect at ≥100 mg/L."""
    from lifelines import KaplanMeierFitter
    import sqlite3, pandas as pd
    con = sqlite3.connect(str(_DB_PATH))
    df = pd.read_sql(
        "SELECT ttf_true_best_days AS ttf, event, h2s_proxy_mg_l "
        "FROM mart__vt_freq55 WHERE ttf_true_best_days > 0 AND h2s_proxy_mg_l IS NOT NULL",
        con)
    con.close()

    cuts = [-np.inf, 0.001, 10, 100, np.inf]
    lbls = ['= 0  (no H₂S)', '0 – 10 mg/L', '10 – 100 mg/L', '≥ 100 mg/L  ★']
    cols = ['#2E7D32', '#1565C0', '#F57C00', '#C62828']
    df['bin'] = pd.cut(df['h2s_proxy_mg_l'], cuts, labels=lbls)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for lbl, col in zip(lbls, cols):
        lw = 2.8 if '★' in lbl else 1.9
        sub = df[df['bin'] == lbl]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['ttf'], event_observed=sub['event'])
        med = kmf.median_survival_time_
        med_s = f'{int(med)} d' if np.isfinite(med) else '—'
        t = kmf.timeline
        s = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_survival_function_
        m = t <= 900
        ax.plot(t[m], s[m], color=col, lw=lw,
                label=f'H₂S {lbl}  n={len(sub)},  med={med_s}')
        ax.fill_between(t[m], ci.iloc[:, 0][m], ci.iloc[:, 1][m], color=col, alpha=0.09)
        if np.isfinite(med) and med <= 900:
            ax.axvline(med, color=col, lw=0.8, ls=':', alpha=0.65)

    _style_ax(ax)
    ax.set_title('KM Survival by H₂S concentration  —  Кэкспл-filtered runs',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_chem_km(db_col, chem_label, kekspl_filter=True, t_max=900):
    """KM for a chemical covariate (tertile split on daily rate kg/d)."""
    from lifelines import KaplanMeierFitter
    import sqlite3, pandas as pd
    con = sqlite3.connect(str(_DB_PATH))
    df = pd.read_sql(
        f"SELECT ttf_true_best_days AS ttf, event, run_days, {db_col} "
        "FROM mart__vt_freq55 "
        f"WHERE ttf_true_best_days > 0 AND run_days > 0 AND {db_col} IS NOT NULL",
        con)
    con.close()
    df['rate'] = df[db_col] / df['run_days']
    if kekspl_filter:
        df = df[df['ttf'] / df['run_days'] >= 0.5]

    q33 = df['rate'].quantile(0.333)
    q67 = df['rate'].quantile(0.667)
    df['bin'] = pd.cut(df['rate'], [-np.inf, q33, q67, np.inf],
                       labels=['Low', 'Mid', 'High'])

    bin_cfg = {
        'Low':  (_C_LOW,  f'Low  < {q33/1000:.0f}k kg/d'),
        'Mid':  (_C_MID,  f'Mid  {q33/1000:.0f}k–{q67/1000:.0f}k kg/d'),
        'High': (_C_HIGH, f'High  > {q67/1000:.0f}k kg/d'),
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for bname, (col, rng_lbl) in bin_cfg.items():
        sub = df[df['bin'] == bname]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['ttf'], event_observed=sub['event'])
        med = kmf.median_survival_time_
        med_s = f'{int(med)} d' if np.isfinite(med) else '—'
        t = kmf.timeline
        s = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_survival_function_
        m = t <= t_max
        ax.plot(t[m], s[m], color=col, lw=2.1,
                label=f'{rng_lbl}  n={len(sub)},  med={med_s}')
        ax.fill_between(t[m], ci.iloc[:, 0][m], ci.iloc[:, 1][m], color=col, alpha=0.10)
        if np.isfinite(med) and med <= t_max:
            ax.axvline(med, color=col, lw=0.8, ls=':', alpha=0.65)

    filt_note = '  Кэкспл ≥ 0.5 filter' if kekspl_filter else '  raw TTF'
    _style_ax(ax, t_max)
    ax.set_title(f'KM Survival by {chem_label} (daily load tertiles){filt_note}',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig_to_bytes(fig)


def fig_covariate_summary():
    """Horizontal bar chart — median survival ratio (stressed / baseline) per covariate."""
    from lifelines import KaplanMeierFitter
    import sqlite3, pandas as pd

    con = sqlite3.connect(str(_DB_PATH))
    df = pd.read_sql(
        "SELECT ttf_true_best_days AS ttf, event, run_days, field, contractor, "
        "avg_glf, h2s_proxy_mg_l, "
        "cum_calcium_load_kg, cum_chloride_load_kg, cum_sulfate_load_kg "
        "FROM mart__vt_freq55 WHERE ttf_true_best_days > 0 AND run_days > 0",
        con)
    con.close()
    df['kekspl'] = df['ttf'] / df['run_days']
    df_k = df[df['kekspl'] >= 0.5].copy()

    def km_med(sub):
        if len(sub) < 5:
            return np.nan
        kmf = KaplanMeierFitter()
        kmf.fit(sub['ttf'], event_observed=sub['event'])
        return kmf.median_survival_time_

    factors = []

    # Contractor: NT vs Borec (raw)
    m_borec = km_med(df[df['contractor'] == 'Борец'])
    m_nt    = km_med(df[df['contractor'] == 'Новые технологии'])
    factors.append(('Contractor\nNT vs Borec', m_nt / m_borec if np.isfinite(m_nt) else np.nan, 'raw'))

    # Кэкспл: raw All vs kekspl≥0.5 All
    m_raw = km_med(df)
    m_fil = km_med(df_k)
    factors.append(('Кэкспл filter\nraw / kekspl≥0.5', m_raw / m_fil, 'filter'))

    # GLF: 0-100 vs 700+
    df_glf = df.dropna(subset=['avg_glf'])
    m_glf_lo = km_med(df_glf[df_glf['avg_glf'] <= 100])
    m_glf_hi = km_med(df_glf[df_glf['avg_glf'] >= 700])
    factors.append(('GLF\n<100 vs 700+', m_glf_lo / m_glf_hi, 'raw'))

    # H2S: ≥100 vs =0
    df_h = df_k.dropna(subset=['h2s_proxy_mg_l'])
    m_h_lo = km_med(df_h[df_h['h2s_proxy_mg_l'] <= 0.001])
    m_h_hi = km_med(df_h[df_h['h2s_proxy_mg_l'] >= 100])
    factors.append(('H₂S\n≥100 vs =0', m_h_hi / m_h_lo, 'kekspl'))

    for chem_col, lbl in [
        ('cum_sulfate_load_kg',  'SO₄ load\nhigh vs low'),
        ('cum_calcium_load_kg',  'Ca load\nhigh vs low'),
        ('cum_chloride_load_kg', 'Cl load\nhigh vs low'),
    ]:
        tmp = df_k.dropna(subset=[chem_col]).copy()
        tmp['rate'] = tmp[chem_col] / tmp['run_days']
        q33, q67 = tmp['rate'].quantile(0.333), tmp['rate'].quantile(0.667)
        m_lo = km_med(tmp[tmp['rate'] <= q33])
        m_hi = km_med(tmp[tmp['rate'] >= q67])
        factors.append((lbl, m_hi / m_lo, 'kekspl'))

    labels  = [f[0] for f in factors]
    ratios  = [f[1] for f in factors]
    ftypes  = [f[2] for f in factors]
    colors  = ['#C62828' if r < 0.4 else '#F57C00' if r < 0.6 else '#1565C0'
               for r in ratios]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.barh(labels[::-1], [r for r in ratios[::-1]], color=colors[::-1],
                   height=0.55, edgecolor='white', linewidth=0.6)
    ax.axvline(1.0, color='#333', lw=1.2, ls='--', alpha=0.6)
    for bar, ratio in zip(bars, ratios[::-1]):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f'{ratio:.2f}×', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.35)
    ax.set_xlabel('Median survival ratio  (stressed / reference)', fontsize=10.5)
    ax.set_title('Covariate Effect Sizes  —  median ratio < 1 means shorter life under stress',
                 fontweight='bold', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    return fig_to_bytes(fig)


# ─── PPTX builder ─────────────────────────────────────────────────────────────

SL_W = Inches(13.333)
SL_H = Inches(7.5)
HDR_H = Inches(1.08)


def new_prs():
    prs = Presentation()
    prs.slide_width  = SL_W
    prs.slide_height = SL_H
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_rect(slide, left, top, width, height, fill_rgb, line=False):
    shp = slide.shapes.add_shape(1, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_rgb
    if not line:
        shp.line.fill.background()
    return shp


def add_hdr(slide, title, subtitle=None):
    """Colored header band."""
    shp = add_rect(slide, 0, 0, SL_W, HDR_H, NAVY)
    tf  = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = WHITE
    r.font.name = 'Calibri'
    if subtitle:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(11)
        r2.font.color.rgb = LIGHT
        r2.font.name = 'Calibri'
    # thin accent line
    add_rect(slide, 0, HDR_H, SL_W, Inches(0.04), RGBColor(0x15, 0x65, 0xC0))


def add_img(slide, img_bytes, left, top, width, height):
    slide.shapes.add_picture(img_bytes, left, top, width, height)


def txt(slide, text, left, top, width, height,
        size=11, bold=False, color=None, align=PP_ALIGN.LEFT, italic=False):
    """Text box with newline support."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size   = Pt(size)
        run.font.bold   = bold
        run.font.italic = italic
        run.font.name   = 'Calibri'
        if color:
            run.font.color.rgb = color


def bullets(slide, items, left, top, width, height, size=12):
    """
    items: list of (text, indent_level).
    indent_level 0 = body, 1 = sub-bullet.
    """
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (line, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.name = 'Calibri'


# ─── Build all slides ─────────────────────────────────────────────────────────

def build(prs):
    # ── Slide 1: Title ────────────────────────────────────────────────────────
    sl = blank(prs)
    add_rect(sl, 0, 0, SL_W, SL_H, NAVY)
    add_rect(sl, 0, Inches(3.85), SL_W, Inches(0.055), RGBColor(0x15, 0x65, 0xC0))

    txt(sl, 'Remaining Useful Life (RUL)\nEstimation for ESP Fleet',
        Inches(1.0), Inches(1.3), Inches(11.3), Inches(2.0),
        size=36, bold=True, color=WHITE)
    txt(sl, 'State-of-the-Art Weibull Survival Analysis Approach',
        Inches(1.0), Inches(3.95), Inches(11.0), Inches(0.6),
        size=18, color=LIGHT)
    txt(sl, '2,046 runs  ·  1,242 failures  ·  7 fields  ·  Borec + Schlumberger',
        Inches(1.0), Inches(5.0), Inches(11.0), Inches(0.5),
        size=13, color=RGBColor(0xA0, 0xB8, 0xCC))
    txt(sl, 'Data: pump2.db · mart__vt_freq55  ·  TTF = ttf_true_best_days (working days)',
        Inches(1.0), Inches(6.5), Inches(11.0), Inches(0.4),
        size=10.5, color=RGBColor(0x78, 0x98, 0xB0))

    # ── Slide 2: Weibull Survival Function ───────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Weibull Survival Function',
            'Single-component parametric model — the foundation of our RUL framework')

    eq1 = render_eq(
        r'S(t) = P(T > t) = \exp\left[-\left(\frac{t}{\eta}\right)^{\beta}\right]',
        fontsize=26, figsize=(7.0, 1.05))
    add_img(sl, eq1, Inches(0.4), Inches(1.18), Inches(7.0), Inches(1.02))

    items = [
        ('β  — shape parameter (determines failure mode)', 0),
        ('β < 1  →  Infant mortality: early failures dominate; hazard decreases with time', 1),
        ('β = 1  →  Random / memoryless failure (exponential distribution)', 1),
        ('β > 1  →  Wear-out: hazard increases with time — equipment ages', 1),
        ('β >> 1  (e.g. 5–10)  →  Near-deterministic wear-out: failures tightly clustered', 1),
        ('                               around η — like a light bulb burning out at rated life', 1),
        ('', 0),
        ('η  — scale / characteristic life: 63.2 % of units fail by t = η days', 0),
        ('', 0),
        ('Key statistics:', 0),
        ('Median (P50):  t₅₀  =  η · (ln 2)^(1/β)', 1),
        ('Mean E[T]:       η · Γ(1 + 1/β)', 1),
        ('Hazard:           h(t)  =  (β/η) · (t/η)^(β−1)', 1),
    ]
    bullets(sl, items, Inches(0.4), Inches(2.28), Inches(7.7), Inches(4.7), size=12)

    img_beta = fig_beta_shapes()
    add_img(sl, img_beta, Inches(8.1), Inches(1.15), Inches(5.05), Inches(4.15))

    txt(sl,
        'Our ESP fleet (raw):\n'
        'Global excl. Vt  β = 0.91  (infant-mortality mix)\n'
        'Vt field              β = 1.00  (random boundary)\n'
        'After Кэкспл filter:  β rises to 1.06–1.23  (wear-out)',
        Inches(8.1), Inches(5.35), Inches(5.05), Inches(1.55), size=10.5, color=NAVY)

    # ── Slide 3: Survival comparison (KM + Weibull, raw data) ───────────────
    sl = blank(prs)
    add_hdr(sl, 'Kaplan-Meier + Weibull MLE — Global excl. Vt  vs.  Vt Field',
            'Raw unfiltered data  ·  Kaplan-Meier (step) overlaid with parametric Weibull (dashed)')

    img3, fp = fig_slide3_km_weibull()
    add_img(sl, img3, Inches(0.1), Inches(1.15), Inches(9.55), Inches(5.95))

    # Side panel — live fitted parameters
    txt(sl, 'Weibull MLE fit', Inches(9.75), Inches(1.2), Inches(3.4), Inches(0.35),
        size=11, bold=True, color=NAVY)

    col_xs = [9.78, 10.84, 11.38, 11.95, 12.55]
    col_ws = [1.03,  0.52,  0.54,  0.58,  0.58]
    hdrs_s = ['Group', 'β', 'η', 'Median', 'Mean']
    for h_txt, cx, cw in zip(hdrs_s, col_xs, col_ws):
        txt(sl, h_txt, Inches(cx), Inches(1.58), Inches(cw), Inches(0.32),
            size=9.5, bold=True, color=NAVY)

    gp = fp['Global excl. Vt']
    vp = fp['Vt Field']
    ratio_med = vp['med'] / gp['med']
    ratio_mea = vp['mea'] / gp['mea']
    tbl_rows = [
        ('Gl. excl Vt',
         f"{gp['beta']:.3f}", f"{gp['eta']:.0f} d",
         f"{gp['med']:.0f} d", f"{gp['mea']:.0f} d"),
        ('Vt Field',
         f"{vp['beta']:.3f}", f"{vp['eta']:.0f} d",
         f"{vp['med']:.0f} d", f"{vp['mea']:.0f} d"),
        ('Ratio Vt/Gl', '—', '—',
         f'{ratio_med:.2f}×', f'{ratio_mea:.2f}×'),
    ]
    for ri, row in enumerate(tbl_rows):
        for val, cx, cw in zip(row, col_xs, col_ws):
            txt(sl, val, Inches(cx), Inches(1.94 + ri * 0.4), Inches(cw), Inches(0.36),
                size=9, color=(NAVY if ri == 2 else None))

    txt(sl,
        f"Vt: {ratio_med:.0%} shorter median\n"
        f"than rest of fleet (raw data)\n\n"
        f"KM and Weibull align well\n→ Weibull is a good fit\n\n"
        f"Horizontal line at Mean:\nshows S(Mean) on y-axis\n\n"
        f"Mean > Median\n→ right-skewed TTF\n→ use Median for RUL",
        Inches(9.75), Inches(3.16), Inches(3.5), Inches(3.8), size=11)

    # ── Slide 4: Hazard rate ──────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Hazard Rate h(t) and Cumulative Hazard H(t)',
            'Instantaneous failure rate · Accumulated "aging dose" — two views of the same physics')

    eq_h = render_eq(
        r'h(t) = -\frac{d}{dt}\ln S(t) = \frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1}'
        r'\qquad H(t)=-\ln S(t)=\left(\frac{t}{\eta}\right)^{\beta}',
        fontsize=20, figsize=(11, 1.05))
    add_img(sl, eq_h, Inches(1.0), Inches(1.15), Inches(11.0), Inches(1.02))

    img_h = fig_hazard_rate()
    add_img(sl, img_h, Inches(0.1), Inches(2.22), Inches(8.6), Inches(4.55))

    # Physical meaning panel (right side)
    txt(sl, 'Physical meaning', Inches(8.8), Inches(2.28), Inches(4.3), Inches(0.35),
        size=11.5, bold=True, color=NAVY)

    txt(sl,
        'h(t)  — Instantaneous hazard rate\n\n'
        '"What is the probability the pump fails in\n'
        'the next day, given it has survived to today?"\n\n'
        'h(t) · dt  ≈  P(fail in [t, t+dt] | T > t)\n\n'
        'β < 1: risk falls over time — early "lemons"\n'
        '          are weeded out, survivors get safer\n'
        'β = 1: risk constant — failure is memoryless\n'
        'β > 1: risk grows — pump wears out, each\n'
        '          day is more dangerous than the last',
        Inches(8.8), Inches(2.68), Inches(4.3), Inches(2.6), size=11)

    txt(sl, 'H(t)  — Cumulative hazard (accumulated wear)', Inches(8.8), Inches(5.35),
        Inches(4.3), Inches(0.35), size=11.5, bold=True, color=NAVY)

    txt(sl,
        '"Total aging dose absorbed since installation"\n\n'
        'H = 1.0  →  S = e⁻¹ ≈ 37 % still running\n'
        '                   (the pump has "used up" one\n'
        '                   characteristic life unit)\n\n'
        'Proportional hazards: if pump A has H = 2×\n'
        'that of pump B, pump A fails at twice the rate',
        Inches(8.8), Inches(5.72), Inches(4.3), Inches(1.6), size=11)

    # ── Slide 5: TTF vs RUL ───────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Run Life, TTF, and RUL — Three Views of the Same Clock',
            'RUL is updated continuously as the pump accumulates running time')

    txt(sl,
        'Run Life (МРП — межремонтный период)  —  total duration between successive ESP installations/failures\n'
        'TTF (Time to Failure, ННО — наработка на отказ)  —  Run Life of a specific completed run; at t=0 both are equal\n'
        'RUL at age t  —  predicted remaining life for a currently running pump; Run Life = t  +  RUL(t)',
        Inches(0.35), Inches(1.15), Inches(12.7), Inches(0.95), size=12)

    eq_rul = render_eq(
        r'\mathrm{RUL}_{P50}(t)=\left(t^{\beta}+\eta^{\beta}\ln 2\right)^{1/\beta}-t'
        r'\qquad'
        r'P(T>t+\tau\mid T>t)=\frac{S(t+\tau)}{S(t)}',
        fontsize=18, figsize=(11, 1.0))
    add_img(sl, eq_rul, Inches(1.0), Inches(2.14), Inches(11.0), Inches(1.0))

    img_rul = fig_rul_illustration()
    add_img(sl, img_rul, Inches(0.1), Inches(3.18), Inches(12.8), Inches(4.2))

    txt(sl,
        'For a currently running pump (no failure yet):\n'
        '  elapsed age t  +  RUL_P50(t)  =  predicted Run Life',
        Inches(0.3), Inches(7.08), Inches(7.0), Inches(0.38), size=11, color=NAVY)

    # ── Slide 6: Mean vs Median ───────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'E[T] vs. Median — Which Parameter for RUL?',
            'Recommendation: conditional Median (P50) as the operational RUL point estimate')

    img_mm = fig_mean_vs_median()
    add_img(sl, img_mm, Inches(0.1), Inches(1.15), Inches(12.8), Inches(4.95))

    txt(sl,
        '✔  Median (P50): "50 % chance of failure within X days" — directly actionable for maintenance scheduling\n'
        '✘  Mean E[T]: inflated by long-surviving units; no direct probability interpretation; sensitive to heavy tails',
        Inches(0.3), Inches(6.2), Inches(12.7), Inches(0.9), size=12)

    # ── Slide 7: Cox PH Model ─────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Cox Proportional Hazards → Weibull-Cox: RUL under a Hazard Ratio',
            'HR shifts the Weibull scale η while keeping shape β fixed — closed-form RUL follows')

    eq_cox = render_eq(
        r'h(t\mid\mathbf{x}) = h_0(t)\cdot\exp\left(\beta_1 x_1 + \beta_2 x_2 + \cdots\right)'
        r'\qquad HR \equiv \exp\left(\sum_i \beta_i x_i\right)',
        fontsize=19, figsize=(11.5, 1.1))
    add_img(sl, eq_cox, Inches(0.3), Inches(1.15), Inches(11.5), Inches(1.08))

    # Left: model bullets
    items_cox = [
        ('h₀(t) = (β/η₀)(t/η₀)^(β−1)  — Weibull baseline hazard', 0),
        ('HR = exp(Σ βᵢxᵢ)  — time-constant hazard multiplier', 0),
        ('PH assumption: β (shape) unchanged across covariate levels', 0),
        ('β unchanged → only η shifts → closed-form RUL', 0),
        ('', 0),
        ('Under Weibull-Cox, a hazard ratio HR ≡ c implies:', 0),
        ('S(t|HR) = exp[−(t/η₀)^β · c] = exp[−(t/η_x)^β]', 1),
        ('→ same Weibull shape, new scale η_x = η₀ · HR^(−1/β)', 1),
    ]
    bullets(sl, items_cox, Inches(0.35), Inches(2.35), Inches(6.4), Inches(3.5), size=12)

    # Right: the key formulas
    txt(sl, 'HR  →  New scale, median, mean, and RUL',
        Inches(6.85), Inches(1.2), Inches(6.3), Inches(0.38),
        size=12.5, bold=True, color=NAVY)

    eq_eta = render_eq(
        r'\eta_x = \eta_0 \cdot HR^{-1/\beta}', fontsize=20, figsize=(5.0, 0.82))
    add_img(sl, eq_eta, Inches(6.85), Inches(1.60), Inches(5.0), Inches(0.80))

    eq_meds = render_eq(
        r'\mathrm{median}_x = \mathrm{median}_0 \cdot HR^{-1/\beta}'
        r'\quad'
        r'E[T\mid x] = E[T_0] \cdot HR^{-1/\beta}',
        fontsize=17, figsize=(6.3, 0.82))
    add_img(sl, eq_meds, Inches(6.85), Inches(2.44), Inches(6.3), Inches(0.80))

    eq_rul_hr = render_eq(
        r'\mathrm{RUL}_{P50}(t\mid HR)=\left(t^{\beta}+\frac{\eta_0^{\beta}\ln 2}{HR}\right)^{1/\beta}-t',
        fontsize=18, figsize=(6.3, 0.9))
    add_img(sl, eq_rul_hr, Inches(6.85), Inches(3.28), Inches(6.3), Inches(0.88))

    # Key insight box
    txt(sl,
        'Both median and mean scale by the same factor HR^(−1/β)\n'
        '→ HR > 1 reduces all time metrics; β modulates the magnitude\n'
        '   High β: HR has less leverage  ·  Low β: HR has amplified effect',
        Inches(6.85), Inches(4.22), Inches(6.3), Inches(0.78),
        size=11, color=NAVY,
        italic=True)

    # Worked example box (Cl Low→High, HR≈2.9, β≈0.99 from slide 9 data)
    _beta_cl, _eta0_cl, _med0_cl = 0.99, 583.0, 403.0
    _HR = 2.9
    _factor = _HR ** (-1.0 / _beta_cl)
    _med_x  = _med0_cl * _factor
    _t90_base = (90.0**_beta_cl + _eta0_cl**_beta_cl * np.log(2))**(1/_beta_cl) - 90.0
    _t90_hr   = (90.0**_beta_cl + _eta0_cl**_beta_cl * np.log(2) / _HR)**(1/_beta_cl) - 90.0

    txt(sl,
        f'Example — Cl load (High vs Low),  HR = {_HR:.1f},  β = {_beta_cl:.2f}\n'
        f'Scale factor  HR^(−1/β) = {_factor:.3f}\n'
        f'Median:  {_med0_cl:.0f} d  →  {_med_x:.0f} d  ({_factor:.0%} of baseline)\n'
        f'RUL at t=90 d:  {_t90_base:.0f} d  →  {_t90_hr:.0f} d',
        Inches(6.85), Inches(5.07), Inches(6.3), Inches(1.1),
        size=12)

    # ── Slide 8: Chemical covariate effects ───────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Covariate Effects — Chemical Loads',
            'Tertile bins  ·  Fleet kekspl (Кэкспл > 0.5)  ·  Log-rank + Weibull per bin')

    img_chem = fig_chemical_effects()
    add_img(sl, img_chem, Inches(0.1), Inches(1.15), Inches(12.8), Inches(5.35))

    txt(sl,
        'Cl load:  strongest monotone trend across ALL sub-fleets (incl. fleet excl. Vt) — most robust RUL predictor\n'
        'H₂S:  threshold at ≥ 100 mg/L — use as binary feature; applicable primarily to Vt (76 % of ≥100 mg/L runs)',
        Inches(0.3), Inches(6.58), Inches(12.7), Inches(0.82), size=11.5)

    # ── Slide 9: Cox HR illustration — Cl bins with actual KM ─────────────────
    sl = blank(prs)
    add_hdr(sl, 'Covariate Effect — Cl Load Tertiles (KM + Weibull MLE)',
            'Actual Kaplan-Meier curves (step) + Weibull MLE (line) · Fleet raw data · n ≈ 454 per bin')

    img_cl = fig_cox_cl_km()
    add_img(sl, img_cl, Inches(0.1), Inches(1.15), Inches(12.8), Inches(5.5))

    txt(sl,
        'HR ≈ median(Low) / median(High) under proportional hazards  ·  '
        'Log-rank p ≈ 0.000  ·  Monotone trend confirmed in ALL sub-fleets including fleet excl. Vt',
        Inches(0.3), Inches(6.72), Inches(12.7), Inches(0.68), size=11.5)

    # ── Slide 10: Contractor ──────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Contractor Effect — Borec vs. Schlumberger',
            'B + S sub-fleet (n = 1,917); New Technologies excluded (median 89 d, 2.5–3× worse than Borec)')

    img_c = fig_contractor()
    add_img(sl, img_c, Inches(0.3), Inches(1.15), Inches(8.8), Inches(4.9))

    txt(sl,
        'Key findings:\n\n'
        '• Fleet level: Borec significantly better (p = 0.002)\n'
        '  Ratio Sch/Bor = 0.80  →  HR ≈ 1.25×\n\n'
        '• Vt field: difference not significant (p = 0.07)\n'
        '  Harsh conditions dominate; Ratio = 0.80\n\n'
        '• Main contractor gap = New Technologies (НТ)\n'
        '  58 % of НТ runs on Vt (high H₂S, high Cl)\n\n'
        '• Caution: selection bias possible\n'
        '  НТ may be deployed on harder wells',
        Inches(9.15), Inches(1.25), Inches(4.0), Inches(5.2), size=11.5)

    # ── Slide 11: Latent Weibull — model ─────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Latent 2-Component Weibull Mixture',
            'Captures heterogeneous failure modes — two hidden subpopulations within the same fleet')

    eq_mix = render_eq(
        r'S(t) = \pi_1\exp\left[-\left(\frac{t}{\eta_1}\right)^{\beta_1}\right]'
        r'+\,\pi_2\exp\left[-\left(\frac{t}{\eta_2}\right)^{\beta_2}\right]',
        fontsize=22, figsize=(10, 1.1))
    add_img(sl, eq_mix, Inches(1.7), Inches(1.15), Inches(10.0), Inches(1.08))

    items_mix = [
        ('π₁ + π₂ = 1  —  mixing weights: fraction of fleet in each latent subpopulation', 0),
        ('', 0),
        ('Subpopulation 1  (β₁ ≲ 1):  early failures', 0),
        ('Defective equipment, poor installation, gas slugs — high early hazard that quickly falls', 1),
        ('"Bad actors" that would never achieve a long run regardless of conditions', 1),
        ('', 0),
        ('Subpopulation 2  (β₂ > 1):  normal wear-out population', 0),
        ('Correctly installed, good-quality equipment that ages gradually', 1),
        ('Hazard grows with time — standard wear of bearings, impellers, cable insulation', 1),
        ('', 0),
        ('Estimation: Bayesian MCMC (PyMC) with weakly-informative priors', 0),
        ('Model selection: ΔAIC < −4 vs. 1-component → prefer 2-component', 0),
        ('Mean: E[T] = π₁ η₁ Γ(1+1/β₁) + π₂ η₂ Γ(1+1/β₂)  ·  Quantiles Bₙ: numerical solution', 0),
    ]
    bullets(sl, items_mix, Inches(0.45), Inches(2.32), Inches(12.5), Inches(4.5), size=12.5)

    # ── Slide 12: Latent Weibull — Vt results (KM + correct split) ───────────
    sl = blank(prs)
    add_hdr(sl, 'Latent 2-Component Weibull — Vt Field (KM + Individual Subpopulations)',
            'Bayesian mixture · All Vt runs · n = 310, events = 213 · Right panel: each curve starts at 1.0')

    img_lat = fig_latent_vt_km()
    add_img(sl, img_lat, Inches(0.1), Inches(1.15), Inches(12.8), Inches(5.6))

    txt(sl,
        '33 % of Vt runs: early-failure mode (β≈0.92, η≈123 d, med≈83 d) — these pumps fail quickly regardless of age\n'
        '67 % of Vt runs: wear-out mode (β≈1.16, η≈251 d, med≈183 d) — gradual aging pattern  ·  '
        'Mixture B50 = 145 d, Mean ≈ 202 d',
        Inches(0.3), Inches(6.83), Inches(12.7), Inches(0.62), size=11.5)

    # ── Slide 13: Summary / RUL priors ───────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Summary — Weibull Parameters & RUL Priors',
            'Recommended parameter set for Bayesian conditional RUL estimation')

    sum_rows = [
        ('Global Fleet',       '1-comp', '1.059', '443 d', '313 d', '427 d', 'General prior'),
        ('Vt Field',           '1-comp', '1.229', '239 d', '178 d', '220 d', 'Vt-specific prior'),
        ('Vt — early mode',    '2-comp', '0.918', '123 d', '~83 d', '~133 d', '33 % of runs'),
        ('Vt — wear-out mode', '2-comp', '1.155', '251 d', '~183 d','~229 d', '67 % of runs'),
    ]
    fig_sum, ax_s = plt.subplots(figsize=(12, 3.5))
    ax_s.axis('off')
    col_lbl_s = ['Scope', 'Model', 'β', 'η', 'Median P50', 'Mean E[T]', 'Application']
    tbl_s = ax_s.table(cellText=sum_rows, colLabels=col_lbl_s, loc='center', cellLoc='center')
    tbl_s.auto_set_font_size(False)
    tbl_s.set_fontsize(10.5)
    tbl_s.scale(1.0, 2.25)
    row_bg = ['white', '#EBF5FB', '#EBF5FB', '#FFF3E0', '#FFF3E0']
    for (r, c), cell in tbl_s.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(H_NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        elif r <= 5:
            cell.set_facecolor(row_bg[r])
    buf_s = io.BytesIO()
    fig_sum.savefig(buf_s, format='png', bbox_inches='tight', facecolor='white', dpi=160)
    buf_s.seek(0)
    plt.close(fig_sum)
    add_img(sl, buf_s, Inches(0.15), Inches(1.15), Inches(13.0), Inches(3.7))

    items_feat = [
        ('Cl load (continuous)  ·  Ca load (continuous)  ·  SO₄ load (continuous)', 0),
        ('H₂S ≥ 100 mg/L (binary)  ·  GLF bin (categorical, U-shaped effect)  ·  Contractor (binary)', 0),
        ('Gypsum proxy — excluded: confirmed null (p > 0.19 at n > 300 / bin in all sub-fleets)', 0),
        ('', 0),
        ('Conditional Median RUL:   RUL_P50(t) = [ t^β + η^β · ln 2 ]^(1/β) − t', 0),
    ]
    bullets(sl, items_feat, Inches(0.35), Inches(5.05), Inches(12.7), Inches(2.0), size=12)

    # ── Field survival comparison ─────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Survival by Field  —  All Major Fields',
            'Left: raw TTF  ·  Right: utilisation factor ≥ 0.5  ·  Kaplan-Meier estimates')

    img_fld = fig_field_km()
    add_img(sl, img_fld, Inches(0.1), Inches(1.15), Inches(13.1), Inches(5.0))

    txt(sl,
        'Vt is the harshest field (median 145 d raw)  ·  Ya has the largest population and pulls the fleet average down\n'
        'Da outlier: small n with high β — possible data artefact  ·  Mc best survival: moderate chemistry, stable frequency',
        Inches(0.3), Inches(6.23), Inches(12.7), Inches(0.72), size=11.5)

    # ── Failure category survival ─────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Cause-Specific Survival by Failure Node',
            'Each curve: KM treating all other failure modes as censored  ·  raw TTF, all runs')

    img_fc = fig_failure_cat_km()
    add_img(sl, img_fc, Inches(0.1), Inches(1.15), Inches(13.1), Inches(5.05))

    # Failure node descriptions
    txt(sl,
        'ESP Pump — impeller/diffuser wear, sand erosion  ·  Power Cable — insulation degradation, H₂S corrosion\n'
        'ESP Motor — winding failure, bearing wear  ·  Protector/Seal — elastomer degradation, shaft seal failure\n'
        'Production Tubing — corrosion, scale deposition  ·  Gas Separator — flooding at low GLF  ·  Drain Valve — seal failure',
        Inches(0.3), Inches(6.28), Inches(12.7), Inches(0.9), size=11)

    # ── Data issues ───────────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Data Quality Issues  —  Known Limitations',
            'Understanding biases before modelling is essential for reliable RUL estimates')

    txt(sl, 'TTF Calculation: Date Difference vs. True Working Days',
        Inches(0.4), Inches(1.18), Inches(12.5), Inches(0.35),
        size=13.5, bold=True, color=NAVY)

    items_di1 = [
        ('Date-diff TTF  =  stop_date − install_date  =  calendar days including shutdowns and idle periods', 0),
        ('  Can be 2–5× larger than actual pump-on time for cycling / frequently-stopped wells', 1),
        ('  Example: 180 calendar days at 55% utilisation → true TTF ≈ 99 d, date-diff = 180 d', 1),
        ('TTF True (ttf_true_best_days)  =  union(telemetry qliq > 0, tech-reg status "В работе")', 0),
        ('  Best available proxy, but telemetry may lag actual pump state by hours to days', 1),
        ('  Used throughout this analysis; date-diff TTF is NOT used', 1),
    ]
    bullets(sl, items_di1, Inches(0.4), Inches(1.58), Inches(12.5), Inches(2.2), size=12)

    txt(sl, 'Other Potential Data Issues',
        Inches(0.4), Inches(3.85), Inches(12.5), Inches(0.35),
        size=13.5, bold=True, color=NAVY)

    items_di2 = [
        ('H₂S proxy: pad- or well-median concentration, not a direct run measurement → spatial averaging error', 0),
        ('Chemical loads: estimated from spot-sampled produced water; continuous exposure is unknown', 0),
        ('Right-censoring: pumps active at data export → event = 0, failure date unknown (handled by KM / Weibull)', 0),
        ('Failing node classification: based on field engineering reports → subjective, ~5–10 % misclassification likely', 0),
        ('GLF averaging: run-average GLF misses transient slugging events that may drive early failures', 0),
        ('Missing chemical data: ~21 % of runs have no H₂S record, ~33 % no Cl record → imputation or list-wise exclusion', 0),
        ('Contractor reassignment: some wells changed contractor mid-run → ambiguous attribution', 0),
    ]
    bullets(sl, items_di2, Inches(0.4), Inches(4.25), Inches(12.5), Inches(2.85), size=11.5)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Data Quality & Operational Covariates
    # ══════════════════════════════════════════════════════════════════════════

    # ── Section divider ───────────────────────────────────────────────────────
    sl = blank(prs)
    add_rect(sl, 0, 0, SL_W, SL_H, RGBColor(0x0D, 0x2A, 0x4A))
    txt(sl, 'Section II', Inches(0.6), Inches(1.8), Inches(12.0), Inches(0.7),
        size=22, bold=False, color=RGBColor(0xB0, 0xC4, 0xDE),
        align=PP_ALIGN.CENTER)
    txt(sl, 'Data Quality Filters &\nOperational Covariates',
        Inches(0.6), Inches(2.5), Inches(12.0), Inches(1.6),
        size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(sl,
        'Contractor composition  ·  Кэкспл stability filter  ·  GLF  ·  H₂S  ·  Ca  ·  Cl  ·  SO₄',
        Inches(0.6), Inches(4.35), Inches(12.0), Inches(0.5),
        size=16, color=LIGHT, align=PP_ALIGN.CENTER)

    # ── Contractor: all 5, NT highlighted ─────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Contractor Effects — NT (New Tech) as an Outlier',
            'KM survival by contractor  ·  left: raw TTF  ·  right: stable runs (Кэкспл ≥ 0.5)')

    img_cont = fig_contractor_km()
    add_img(sl, img_cont, Inches(0.1), Inches(1.15), Inches(13.1), Inches(5.1))

    txt(sl,
        'NT — New Tech median ≈ 89 d (raw) vs Borec ≈ 281 d  →  ratio 0.32  —  dominates fleet-level statistics\n'
        'After Кэкспл ≥ 0.5 filter: NT n drops sharply (mostly cycling runs); Borec/Schlumberger ratio stabilises at ≈ 0.80',
        Inches(0.3), Inches(6.35), Inches(12.7), Inches(0.72), size=11.5)

    # ── Кэкспл filter: before / after ─────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Кэкспл Stability Filter  —  Effect on Survival Shape',
            'Кэкспл = TTF / run_days  ·  threshold 0.5 removes cycling / frequently-stopped runs')

    img_kek = fig_kekspl_filter()
    add_img(sl, img_kek, Inches(0.1), Inches(1.15), Inches(13.1), Inches(4.95))

    txt(sl,
        'Raw data contains early-censored "cycling" runs that inflate apparent infant mortality (β → 0.9)\n'
        'After filter: β rises toward wear-out for both fields  ·  Global: β 0.90→1.06  ·  Vt: β 1.00→1.23\n'
        'Chemical and frequency analyses are always presented on Кэкспл ≥ 0.5 data',
        Inches(0.3), Inches(6.18), Inches(12.7), Inches(0.95), size=11.5)

    # ── Frequency groups — fleet + Vt ────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Operating Frequency Groups  —  Fleet & Vt KM + Weibull',
            'Groups: UHF-60 (mean >58 Hz)  ·  HF (>50% time >55 Hz)  ·  Base (50–55 Hz)  ·  LF (<45 Hz)')

    img_fq = fig_freq_groups_km()
    add_img(sl, img_fq, Inches(0.1), Inches(1.15), Inches(13.1), Inches(5.05))

    txt(sl,
        'UHF-60 is the only group significantly worse than Base (log-rank p < 0.007 on TTF true)\n'
        'HF ≈ Base — running >50% of time at >55 Hz does not reduce survival if mean stays ≤ 58 Hz\n'
        'Vt UHF-60 n=9: indicative only; all Vt pairs p > 0.5 due to small sample',
        Inches(0.3), Inches(6.28), Inches(12.7), Inches(0.88), size=11.5)

    # ── Frequency stability: UHF-60 stable vs unstable ───────────────────────
    sl = blank(prs)
    add_hdr(sl, 'UHF-60 Effect Depends on Operating Stability',
            'Split by utilisation factor  ·  unstable = cycling / frequent stops  ·  RMST horizon = 817 d')

    img_fqs = fig_freq_stability_km()
    add_img(sl, img_fqs, Inches(0.1), Inches(1.15), Inches(13.1), Inches(5.05))

    txt(sl,
        'Unstable UHF-60 (util < 0.65):  RMST −33% vs Base  ·  median 91 d  →  frequent restarts at 60 Hz accelerate wear\n'
        'Stable UHF-60 (util ≥ 0.65):  RMST −16% vs Base  ·  median 246 d  →  moderate effect; acceptable if well conditions permit\n'
        'Diagnosis: util < 0.65 at >58 Hz is a sign of protective shutdowns / gas slugging — operating point should be revised',
        Inches(0.3), Inches(6.28), Inches(12.7), Inches(0.88), size=11.5)

    # ── GLF ───────────────────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Gas-Liquid Factor (GLF)  —  Non-Monotone U-Shaped Effect',
            'Very low GLF = insufficient gas separation  ·  Very high GLF = stable wells with separators')

    img_glf = fig_glf_km()
    add_img(sl, img_glf, Inches(0.55), Inches(1.15), Inches(11.5), Inches(5.0))

    txt(sl,
        'GLF 0–100: worst survival — likely pump cavitation / gas slugging at low separator load\n'
        'GLF 700+: best survival — typically wells equipped with gas separators, stable operating point\n'
        'Non-monotone → use as categorical / binned feature (not linear) in RUL model',
        Inches(0.3), Inches(6.23), Inches(12.7), Inches(0.95), size=11.5)

    # ── H2S ───────────────────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'H₂S Concentration  —  Threshold Effect at ≥ 100 mg/L',
            'Proxy from pad / well median measurements  ·  Кэкспл ≥ 0.5 filter applied')

    img_h2s = fig_h2s_km()
    add_img(sl, img_h2s, Inches(0.55), Inches(1.15), Inches(11.5), Inches(5.0))

    txt(sl,
        'No monotone trend below 100 mg/L — H₂S 0–100 mg/L has marginal impact on survival\n'
        '≥ 100 mg/L: survival collapses  (median ≈ 80 d vs ≈ 250 d at =0)  →  use as binary flag in Cox model\n'
        'Effect concentrated in Vt field (76 % of ≥100 mg/L runs are Vt)',
        Inches(0.3), Inches(6.23), Inches(12.7), Inches(0.95), size=11.5)

    # ── Cl load ───────────────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Chloride Load  —  Strongest Chemical Predictor',
            'Cumulative Cl load / run_days  ·  tertile split  ·  Кэкспл ≥ 0.5  ·  fleet-wide')

    img_cl = fig_chem_km('cum_chloride_load_kg', 'Cl Load', kekspl_filter=True)
    add_img(sl, img_cl, Inches(0.55), Inches(1.15), Inches(11.5), Inches(5.0))

    txt(sl,
        'Monotone negative: High Cl runs survive ≈ 3× shorter than Low Cl  ·  robust across all sub-fleets incl. B+S\n'
        'Log-rank p ≈ 0.000  ·  Ratio high/low ≈ 0.28  —  largest single covariate effect among chemicals\n'
        'Daily rate (kg/d) is more informative than cumulative load alone (normalises for run duration)',
        Inches(0.3), Inches(6.23), Inches(12.7), Inches(0.95), size=11.5)

    # ── Ca + SO4 ──────────────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Calcium & Sulfate Loads  —  Independent Monotone Predictors',
            'Same tertile approach  ·  Кэкспл ≥ 0.5  ·  fleet-wide')

    img_ca  = fig_chem_km('cum_calcium_load_kg',  'Ca Load',  kekspl_filter=True)
    img_so4 = fig_chem_km('cum_sulfate_load_kg',  'SO₄ Load', kekspl_filter=True)
    add_img(sl, img_ca,  Inches(0.1),  Inches(1.15), Inches(6.55), Inches(5.0))
    add_img(sl, img_so4, Inches(6.7),  Inches(1.15), Inches(6.55), Inches(5.0))

    txt(sl,
        'Ca ratio ≈ 0.31  ·  SO₄ ratio ≈ 0.39  —  both independently significant, effects not fully explained by Cl\n'
        'Gypsum proxy (Ca × SO₄) shows no additional signal (p > 0.77) — confirmed null in both full and B+S fleet',
        Inches(0.3), Inches(6.23), Inches(12.7), Inches(0.72), size=11.5)

    # ── Gypsum proxy ──────────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Gypsum Proxy (CaSO₄)  —  Confirmed Null Result',
            'Tertile KM on daily gypsum proxy rate  ·  utilisation factor ≥ 0.5  ·  fleet-wide')

    img_gyp = fig_gypsum_km()
    add_img(sl, img_gyp, Inches(0.55), Inches(1.15), Inches(11.5), Inches(5.0))

    txt(sl,
        'All three tertile KM curves overlap within confidence bands  ·  log-rank p > 0.77 (n > 300 per bin)\n'
        'Null result confirmed on full fleet and B+S sub-fleet independently  ·  exclude from RUL feature set\n'
        'Ca and SO₄ act through separate corrosion / scaling mechanisms — their product adds no predictive signal',
        Inches(0.3), Inches(6.23), Inches(12.7), Inches(0.95), size=11.5)

    # ── Covariate summary ─────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Covariate Effect Summary  —  Median Survival Ratio',
            'Ratio < 1 = stressed group survives shorter  ·  chemicals on Кэкспл ≥ 0.5  ·  contractor raw')

    img_cov = fig_covariate_summary()
    add_img(sl, img_cov, Inches(0.3), Inches(1.15), Inches(12.7), Inches(5.15))

    txt(sl,
        'Contractor composition (NT) has the single largest effect — must be excluded or controlled before interpreting field trends\n'
        'Cl > Ca > H₂S ≥ SO₄ > GLF in terms of effect magnitude  ·  Кэкспл filter is a data-quality step, not a covariate',
        Inches(0.3), Inches(6.35), Inches(12.7), Inches(0.72), size=11.5)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: UHF Decision Support — Model Architecture & Improvements
    # ══════════════════════════════════════════════════════════════════════════

    # ── Section divider ───────────────────────────────────────────────────────
    sl = blank(prs)
    add_rect(sl, 0, 0, SL_W, SL_H, RGBColor(0x0D, 0x2A, 0x4A))
    txt(sl, 'Section III', Inches(0.6), Inches(1.8), Inches(12.0), Inches(0.7),
        size=22, bold=False, color=RGBColor(0xB0, 0xC4, 0xDE),
        align=PP_ALIGN.CENTER)
    txt(sl, 'UHF Frequency Increase (УВЧ)\nDecision Support Model',
        Inches(0.6), Inches(2.5), Inches(12.0), Inches(1.6),
        size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(sl,
        'Current architecture (Cox PH v5)  ·  Vt field  ·  56 UHF candidate wells  ·  Roadmap for improvement',
        Inches(0.6), Inches(4.35), Inches(12.0), Inches(0.5),
        size=16, color=LIGHT, align=PP_ALIGN.CENTER)

    # ── Current model architecture ────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Current RUL Model for Frequency Increase (УВЧ)  —  Architecture v5',
            '2-component Weibull × Cox PH  ·  Vt field  ·  56 UHF candidate wells  ·  C-index 0.60 (hold-out)')

    # Left column: pipeline
    txt(sl, 'Modelling Pipeline', Inches(0.35), Inches(1.18), Inches(6.7), Inches(0.32),
        size=13, bold=True, color=NAVY)

    items_pipe = [
        ('Step 1 — TTF (true working days)', 0),
        ('Union: telemetry qliq > 0  OR  tech-reg status "In operation"', 1),
        ('NOT calendar days  —  date-diff inflates TTF 2–5× on cycling wells', 1),
        ('Кэкспл = ttf_true / run_days ≥ 0.5 filter removes unstable runs', 1),
        ('', 0),
        ('Step 2 — Latent 2-component Weibull prior (Vt, Кэкспл ≥ 0.5,  n = 255)', 0),
        ('Pop-1  early     :  π = 33%,  β = 0.92,  η = 123 d,  median ≈  83 d', 1),
        ('Pop-2  wear-out  :  π = 67%,  β = 1.16,  η = 251 d,  median ≈ 183 d', 1),
        ('Mixture B50 = 145 d  ·  estimated via Bayesian MCMC', 1),
        ('', 0),
        ('Step 3 — Cox PH hazard adjustment  (Weibull PH baseline)', 0),
        ('h(t|x) = h₀(t) · exp(β₁·H2S_flag + β₂·is_NT + β₃·SO₄_log + β₄·Ca_log)', 1),
        ('H2S ≥ 100 mg/L flag  :  HR = 2.78  (p = 0.002)  —  dominant predictor', 1),
        ('NT contractor flag   :  HR = 1.99  (p = 0.003)', 1),
        ('SO₄ log-load         :  HR = 1.11  (p = 0.080)', 1),
        ('Ca log-load          :  HR = 0.93  (p = 0.130)  —  weak protective signal', 1),
        ('', 0),
        ('Step 4 — Conditional median RUL per well  (Pop-2 × Cox)', 0),
        ('RUL_P50(t | HR) = [ t^β + η₀^β · ln 2 / HR ]^(1/β)  −  t', 1),
        ('Three RUL variants: as-is (rul_med_c) · best-case (rul_best) · post-UHF (rul_med_a)', 1),
    ]
    bullets(sl, items_pipe, Inches(0.35), Inches(1.55), Inches(6.7), Inches(5.25), size=10.5)

    # Right column: Cox table + delta decomposition + portfolio numbers
    txt(sl, 'Delta Decomposition  (56 UHF candidate wells)', Inches(7.25), Inches(1.18), Inches(5.8), Inches(0.32),
        size=13, bold=True, color=NAVY)

    delta_rows = [
        ('d_uvch',   'RUL_after_UHF − RUL_current',       '−21 d avg', 'UHF reliability cost'),
        ('d_contr',  'RUL_best − RUL_current',             '+36 d avg', 'Contractor switch (NT wells)'),
        ('d_h2s',    'RUL_no_H2S − RUL_best',              '+21 d avg', 'H2S mitigation potential'),
        ('d_chem',   'RUL_best_chem − RUL_best',           '+61 d avg', 'Total chemistry penalty'),
    ]
    fig_dt, ax_dt = plt.subplots(figsize=(5.85, 2.2))
    ax_dt.axis('off')
    tbl_dt = ax_dt.table(
        cellText=delta_rows,
        colLabels=['Delta', 'Definition', 'Fleet avg', 'Meaning'],
        loc='center', cellLoc='left')
    tbl_dt.auto_set_font_size(False)
    tbl_dt.set_fontsize(9.0)
    tbl_dt.scale(1.0, 1.9)
    for (r, c), cell in tbl_dt.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor('#0D2A4A')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EBF5FB')
    buf_dt = io.BytesIO()
    fig_dt.savefig(buf_dt, format='png', bbox_inches='tight', facecolor='white', dpi=150)
    buf_dt.seek(0)
    plt.close(fig_dt)
    add_img(sl, buf_dt, Inches(7.25), Inches(1.55), Inches(5.85), Inches(2.45))

    txt(sl, 'Portfolio Summary', Inches(7.25), Inches(4.10), Inches(5.8), Inches(0.30),
        size=12, bold=True, color=NAVY)
    items_port = [
        ('56 Vt wells with scheduled UHF transition (freq increase to УВЧ)', 0),
        ('Median current RUL (rul_med_c) = 117.7 days', 0),
        ('21 wells with P(fail ≤ 90 d) > 55% — high near-term risk', 0),
        ('14 NT-contractor wells: avg contractor gain d_contr = +49 to +104 d', 0),
        ('18 H2S wells: avg H2S penalty d_h2s = +71 d  (H2S-only sub-group)', 0),
        ('UHF value is in oil dQн uplift, NOT in reliability  —  avg cost −21 d', 0),
    ]
    bullets(sl, items_port, Inches(7.25), Inches(4.45), Inches(5.85), Inches(2.45), size=11)

    # ── Current implementation overview (simple) ──────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'What Has Been Done  —  Current Model Summary',
            'Vt field  ·  56 UHF candidate wells  ·  Cox PH v5')

    items_done = [
        'TTF measured as true working days (telemetry + operational logs), not calendar days',
        '2-component Weibull survival model fitted on Vt field historical runs (n = 255, Кэкспл ≥ 0.5)',
        'Cox proportional hazards model with 4 covariates: H2S, contractor (NT), sulfate load, calcium load',
        'Conditional median RUL computed per well for three scenarios: as-is · best-case · post-UHF',
        'UHF reliability cost quantified: average −21 days vs current operating frequency',
        'Contractor and chemistry penalties decomposed as actionable deltas per well',
        '56 UHF candidate wells ranked by current RUL and near-term failure risk',
    ]
    top = Inches(1.8)
    for item in items_done:
        txt(sl, f'—   {item}',
            Inches(1.2), top, Inches(11.1), Inches(0.52),
            size=15, color=NAVY)
        top += Inches(0.62)

    # ── Improvement roadmap ───────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'Model Improvement Roadmap  —  Why Each Change Is Needed',
            'Current gaps: run-average covariates · single baseline · composite failure · fixed-in-time RUL · missing telemetry')

    # Left: improvements with motivation
    txt(sl, 'Model Changes & Motivation', Inches(0.35), Inches(1.18), Inches(6.35), Inches(0.32),
        size=13, bold=True, color=NAVY)

    items_near = [
        ('1.  Extended Cox — time-varying covariates', 0),
        ('WHY: H2S flag is a run-average — a well that turns sour mid-run gets the same', 1),
        ('  coefficient as one sour from day 1; underestimates late-run risk for worsening conditions', 1),
        ('FIX: cumulative load x(t) as a time-dependent covariate in h(t|x(t)) = h₀(t)·exp(β·x(t))', 1),
        ('', 0),
        ('2.  Stratified Weibull / Stratified Cox', 0),
        ('WHY: PH assumption forces a common baseline shape on all cohorts; UHF-60 and NT', 1),
        ('  wells have structurally different curve shapes — not just a shifted hazard level', 1),
        ('FIX: separate h₀_k(t) per {frequency group × contractor} stratum, shared β', 1),
        ('', 0),
        ('3.  Competing Risks Model', 0),
        ('WHY: composite "any failure" event mixes ESP pump wear, motor overheating, and cable', 1),
        ('  insulation failure — causes with different drivers, different interventions', 1),
        ('FIX: cause-specific hazard per failure node; Fine-Gray for cumulative incidence', 1),
        ('', 0),
        ('4.  Bayesian Online RUL Updating', 0),
        ('WHY: RUL is computed once at decision time and frozen; a pump that has survived 300 days', 1),
        ('  dramatically updates its conditional survival, but new chemistry / telemetry is ignored', 1),
        ('FIX: conjugate Weibull-Gamma posterior update as new readings arrive; narrowing credible interval', 1),
        ('', 0),
        ('5.  Calibration & Temporal Validation', 0),
        ('WHY: C-index 0.60 ranks wells but does not confirm that predicted P(fail ≤ 90 d) matches', 1),
        ('  actual failure rates; random hold-out mixes temporal cohorts → data leakage', 1),
        ('FIX: temporal split (pre-2024 train / 2024+ test); Brier score calibration curves', 1),
    ]
    bullets(sl, items_near, Inches(0.35), Inches(1.55), Inches(6.35), Inches(5.6), size=10.0)

    # Right: operational parameters to add
    txt(sl, 'Operational Parameters to Add', Inches(6.9), Inches(1.18), Inches(6.15), Inches(0.32),
        size=13, bold=True, color=NAVY)

    items_ops = [
        ('Downhole telemetry (SCADA / DAS)', 0),
        ('Motor load  (% of nameplate current)  —  each 10% overload cuts insulation life ~20%;', 1),
        ('  operating above nameplate is the strongest controllable failure driver', 1),
        ('Motor temperature  —  persistent ΔT > 15°C above rated → early winding failure indicator', 1),
        ('Pump intake pressure (PIP)  —  low PIP → cavitation / starvation; predicts ESP body failure', 1),
        ('Cable resistance trend  —  rising resistance signals moisture ingress before breakdown', 1),
        ('', 0),
        ('Operational regime', 0),
        ('Start / stop count per run  —  each restart imposes thermal cycling on windings; Кэкспл', 1),
        ('  captures utilisation rate but not cycling intensity', 1),
        ('Frequency ramp profile  (dHz/dt)  —  stepwise jumps vs gradual ramp affect shaft fatigue', 1),
        ('Operating point vs pump curve  (Q / Q_opt)  —  significant off-design → hydraulic wear', 1),
        ('', 0),
        ('Reservoir & production', 0),
        ('Water cut  (% Bw)  —  higher watercut → more corrosive environment; multiplies chemical effects', 1),
        ('Actual GLF at pump intake  —  current avg_glf is a reservoir model estimate, not measured', 1),
        ('Reservoir pressure trend  —  declining P_res lowers PIP and raises pump starvation risk', 1),
        ('', 0),
        ('Equipment design', 0),
        ('Pump series / rated horsepower  —  unmodelled design heterogeneity across the fleet', 1),
        ('Cable age and prior deployment count  —  reinstalled cables carry accumulated electrical fatigue', 1),
    ]
    bullets(sl, items_ops, Inches(6.9), Inches(1.55), Inches(6.15), Inches(5.6), size=10.0)

    # ── ToDo (simple) ─────────────────────────────────────────────────────────
    sl = blank(prs)
    add_hdr(sl, 'What Needs to Be Done  —  Next Steps',
            'Data prerequisites · model improvements · validation')

    items_todo = [
        'Integrate downhole telemetry into the model: motor load, motor temperature, pump intake pressure (PIP)',
        'Add production parameters: water cut, actual gas/liquid ratio at pump intake, reservoir pressure trend',
        'Move from run-average to time-varying covariates (Extended Cox) — H2S exposure is dynamic, not static',
        'Stratify the baseline hazard by frequency group and contractor to remove the proportional hazards assumption',
        'Build separate survival models per failure mode (Motor · Cable · ESP pump) to target interventions',
        'Add dynamic RUL update: posterior refreshed as wells age and new chemistry / telemetry arrives',
        'Validate on a temporal hold-out (pre-2024 train / 2024+ test) and add calibration metrics',
    ]
    top = Inches(1.8)
    for item in items_todo:
        txt(sl, f'□   {item}',
            Inches(1.2), top, Inches(11.1), Inches(0.52),
            size=15, color=NAVY)
        top += Inches(0.62)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir  = results_dir('weibull_slides')
    out_path = out_dir / 'weibull_rul_slides_en.pptx'

    print('Generating figures and building presentation...')
    prs = new_prs()
    build(prs)
    prs.save(str(out_path))
    print(f'Saved ({len(prs.slides)} slides): {out_path}')


if __name__ == '__main__':
    main()
