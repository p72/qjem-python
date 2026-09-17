"""Python port of calc_response_to_exogenous_shocks.prg (Q-JEM 2019),
extended with monetary policy shock simulations.

Writes
  output/responses.csv  : deviation from baseline, 20 quarters
                          (% for levels, percentage points for rates)
  output/SimN.png       : charts in the style of the EViews make_graph
"""
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fetch_boj import ensure_boj_files
from qjem import QJEM

matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

SHOCK_START, SHOCK_END = "0006Q1", "0010Q4"  # caldate(0010Q4, -19)
H = 20

# label, unit ("%" = percent deviation, "pp" = percentage-point difference)
VAR_INFO = {
    "GDP": ("Real GDP", "%"),
    "CP": ("Real Private Consumption", "%"),
    "INV": ("Real Private Non-residential Investment", "%"),
    "EX": ("Real Export", "%"),
    "IM": ("Real Import", "%"),
    "CPIXFOR": ("CPI (all items, less fresh food)", "%"),
    "CALL": ("Call rate (policy rate)", "pp"),
    "IRL": ("10-year JGB yield", "pp"),
    "FXYEN": ("USD/JPY (+ = yen depreciation)", "%"),
    "GAP": ("Output gap", "pp"),
    "IG": ("Real Public Investment", "%"),
    "GDPN": ("Nominal GDP", "%"),
}
BOJ_VARS = ["GDP", "CP", "INV", "EX", "IM", "CPIXFOR"]
MP_VARS = ["CALL", "IRL", "GDP", "GAP", "CP", "INV", "FXYEN", "CPIXFOR"]
FISCAL_VARS = ["GDP", "GAP", "IG", "CP", "INV", "IM", "CALL", "CPIXFOR"]


def forward_guidance_segments(n_peg, bp):
    """Credible announcement at quarter 1: the call rate stays `bp` pp above
    baseline for n_peg quarters. In quarter k of the peg, agents' expected
    call rates for the remaining n_peg-k quarters (ZCALL_V1..ZCALL_V{n_peg-k})
    are pinned at the announced level; beyond that, the model's own
    Taylor-rule expectation recursion takes over from the pinned value."""
    segs = []
    for k in range(1, n_peg + 1):
        pinned = {f"ZCALL_V{j}": bp for j in range(1, n_peg - k + 1)}
        segs.append((1, ["CALL"], ["V_CALL"], pinned))
    segs.append((H - n_peg, [], [], {}))
    return segs


# shocks   : (series, op, value, first quarter, last quarter)  1-based
#            op = "mul" (scale), "add" (level), "gdp%" (add value% of baseline real GDP)
# segments : (number of quarters, endo2exog, exog2endo[, pinned]) in sequence;
#            pinned = {endogenous var: pp added to baseline}, equation dropped
SIMS = {
    "Sim1": dict(
        title="Responses to 1% permanent increase in foreign GDP",
        shocks=[("USGDP", "mul", 1.01, 1, H), ("NUSGDP", "mul", 1.01, 1, H)],
        segments=[(H, ["NUSGDP", "USGDP"], ["V_NUSGAP", "V_USGAP"])],
        plot=BOJ_VARS),
    "Sim2": dict(
        title="Responses to 10% permanent decrease in oil price",
        shocks=[("POIL", "mul", 0.9, 1, H)],
        segments=[(H, ["POIL"], ["V_POIL"])],
        plot=BOJ_VARS),
    "Sim3": dict(
        title="1% increase in foreign GDP and 10% decrease in oil price",
        shocks=[("USGDP", "mul", 1.01, 1, H), ("NUSGDP", "mul", 1.01, 1, H),
                ("POIL", "mul", 0.9, 1, H)],
        segments=[(H, ["NUSGDP", "USGDP", "POIL"], ["V_NUSGAP", "V_USGAP", "V_POIL"])],
        plot=BOJ_VARS),
    "Sim4": dict(
        title="Responses to 10% permanent depreciation of the yen against USD",
        shocks=[("FXYEN", "mul", 1.1, 1, H)],
        segments=[(H, ["FXYEN"], ["V_FXYEN"])],
        plot=BOJ_VARS),
    # --- monetary policy (not in the paper; no published benchmark) ---
    "Sim5": dict(
        title="Monetary policy shock: +100bp policy rate innovation in quarter 1\n"
              "(Taylor rule with smoothing thereafter)",
        shocks=[("V_CALL", "add", 1.0, 1, 1)],
        segments=[(H, [], [])],
        plot=MP_VARS),
    "Sim6": dict(
        title="Call rate held 100bp above baseline for 8 quarters,\n"
              "then returns to the Taylor rule",
        shocks=[("CALL", "add", 1.0, 1, 8)],
        segments=[(8, ["CALL"], ["V_CALL"]), (H - 8, [], [])],
        plot=MP_VARS),
    "Sim7": dict(
        title="Call rate held 100bp above baseline for 8 quarters,\n"
              "with credible forward guidance (expected path pinned)",
        shocks=[("CALL", "add", 1.0, 1, 8)],
        segments=forward_guidance_segments(8, 1.0),
        plot=MP_VARS),
    # --- fiscal policy (not in the paper; no published benchmark) ---
    "Sim8": dict(
        title="Public investment permanently raised by 1% of real GDP\n"
              "(monetary policy follows the Taylor rule)",
        shocks=[("IG", "gdp%", 1.0, 1, H)],
        segments=[(H, ["IG"], ["V_IG"])],
        plot=FISCAL_VARS, multiplier="IG"),
    "Sim9": dict(
        title="Public investment permanently raised by 1% of real GDP,\n"
              "call rate held at baseline for 8 quarters (accommodation)",
        shocks=[("IG", "gdp%", 1.0, 1, H)],
        segments=[(8, ["IG", "CALL"], ["V_IG", "V_CALL"]), (H - 8, ["IG"], ["V_IG"])],
        plot=FISCAL_VARS, multiplier="IG"),
    "Sim10": dict(
        title="Public investment raised by 1% of real GDP for 8 quarters only\n"
              "(temporary stimulus, Taylor rule)",
        shocks=[("IG", "gdp%", 1.0, 1, 8)],
        segments=[(H, ["IG"], ["V_IG"])],
        plot=FISCAL_VARS, multiplier="IG"),
}
COMPARE = [("Sim6", "Sim7", "Without vs with forward guidance (8-quarter +100bp peg)",
            "without FG", "with FG"),
           ("Sim8", "Sim9", "Permanent public investment (+1% of GDP): "
            "monetary policy reaction vs accommodation",
            "Taylor rule", "rate pegged 8Q"),
           ("Sim8", "Sim10", "Public investment (+1% of GDP): permanent vs temporary",
            "permanent", "8 quarters only")]


def run_sim(m, sim):
    t0 = m.t(SHOCK_START)
    X = m.base.copy()
    for var, op, val, q_from, q_to in sim["shocks"]:
        sl = slice(t0 + q_from - 1, t0 + q_to)
        if op == "mul":
            X[m.vidx[var], sl] *= val
        elif op == "gdp%":
            X[m.vidx[var], sl] += val / 100 * m.base[m.vidx["GDP"], sl]
        else:
            X[m.vidx[var], sl] += val
    t = t0
    for n, e2x, x2e, *rest in sim["segments"]:
        pinned = rest[0] if rest else {}
        for var, add in pinned.items():
            X[m.vidx[var], t:t + n] = m.base[m.vidx[var], t:t + n] + add
        m.solve(X, t, t + n - 1, m.block_structure(e2x, x2e, fixed=pinned))
        t += n
    assert t == m.t(SHOCK_END) + 1, "segments must cover the shock window"
    return X


def deviation(m, X, var, t0, t1):
    s, b = X[m.vidx[var], t0:t1 + 1], m.base[m.vidx[var], t0:t1 + 1]
    return s / b * 100 - 100 if VAR_INFO[var][1] == "%" else s - b


def multipliers(m, X, gvar, t0, t1):
    """Real GDP multiplier of a fiscal shock to `gvar`: per-quarter dY/dG and
    the cumulative (sum of dY so far) / (sum of dG so far)."""
    dg = X[m.vidx[gvar], t0:t1 + 1] - m.base[m.vidx[gvar], t0:t1 + 1]
    dy = X[m.vidx["GDP"], t0:t1 + 1] - m.base[m.vidx["GDP"], t0:t1 + 1]
    live = np.abs(dg) > 1e-6
    per = np.where(live, dy / np.where(live, dg, 1.0), np.nan)
    return per, np.cumsum(dy) / np.cumsum(dg)


def plot(sim_name, sim, dev):
    n = len(sim["plot"])
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10, 3 * ((n + 1) // 2)))
    for ax, v in zip(axes.flat, sim["plot"]):
        label, unit = VAR_INFO[v]
        ax.plot(range(1, H + 1), dev[v], color="#1f5fa8", lw=2)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{label} [{unit}]", fontsize=10)
        ax.set_xlabel("quarters after shock")
    fig.suptitle(f"{sim['title']}\n(difference from baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{sim_name}.png"), dpi=120)
    plt.close(fig)


def plot_compare(fname, title, vars_, series):
    n = len(vars_)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10, 3 * ((n + 1) // 2)))
    styles = [dict(color="#1f5fa8", lw=2), dict(color="#d9480f", lw=2, ls="--")]
    for ax, v in zip(axes.flat, vars_):
        label, unit = VAR_INFO[v]
        for (name, devs), st in zip(series, styles):
            ax.plot(range(1, H + 1), devs[v], label=name, **st)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{label} [{unit}]", fontsize=10)
        ax.set_xlabel("quarters after shock")
    axes.flat[0].legend()
    fig.suptitle(f"{title}\n(difference from baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    repl = ensure_boj_files()
    m = QJEM(os.path.join(repl, "qjem_plain.txt"), os.path.join(repl, "BASECASE.wf1"))
    t0, t1 = m.t(SHOCK_START), m.t(SHOCK_END)
    quarters = [m.label(t) for t in range(t0, t1 + 1)]

    # sanity check: solving with no shock must reproduce the baseline
    Xb = m.solve(m.base.copy(), t0, t1, m.block_structure())
    dev = np.nanmax(np.abs(Xb[:, t0:t1 + 1] / m.base[:, t0:t1 + 1] - 1)
                    [np.abs(m.base[:, t0:t1 + 1]) > 1e-6])
    print(f"baseline reproduction: max relative deviation = {dev:.2e}")

    rows, all_devs, mult_rows = [], {}, []
    for name, sim in SIMS.items():
        tic = time.time()
        X = run_sim(m, sim)
        print(f"{name} solved in {time.time() - tic:.1f}s")
        devs = {v: deviation(m, X, v, t0, t1) for v in sim["plot"]}
        all_devs[name] = devs
        if sim.get("multiplier"):
            per, cum = multipliers(m, X, sim["multiplier"], t0, t1)
            mult_rows += [dict(sim=name, h=h + 1, impact=p_, cumulative=c)
                          for h, (p_, c) in enumerate(zip(per, cum))]
        for v, arr in devs.items():
            rows += [dict(sim=name, variable=v, unit=VAR_INFO[v][1], quarter=q, h=h + 1, diff=x)
                     for h, (q, x) in enumerate(zip(quarters, arr))]
        plot(name, sim, devs)

    for a, b, title, la, lb in COMPARE:
        plot_compare(f"{a}_vs_{b}.png", title, SIMS[a]["plot"],
                     [(la, all_devs[a]), (lb, all_devs[b])])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "responses.csv"), index=False)
    summary = df[df.h.isin([1, 4, 8, 12, 20])].pivot_table(
        index=["sim", "variable", "unit"], columns="h", values="diff", sort=False)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_rows", 200)
    print(summary)

    if mult_rows:
        md = pd.DataFrame(mult_rows)
        md.to_csv(os.path.join(OUT, "fiscal_multipliers.csv"), index=False)
        print("\nReal GDP multiplier of public investment (dY/dG)")
        print(md[md.h.isin([1, 4, 8, 12, 20])].pivot_table(
            index=["sim"], columns="h", values=["impact", "cumulative"], sort=False))


if __name__ == "__main__":
    main()
