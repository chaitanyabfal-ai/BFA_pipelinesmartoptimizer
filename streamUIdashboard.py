"""Pipeline Smart Optimizer - 4-step presentation-layer dashboard with Left Pane inputs.

Flow: splash page (logo) -> login page (small logo, username/password) -> dashboard.
Inputs, Geometry, Valve Operation, and Step 3 Optimization Settings live in the Left Sidebar.
Step 2 Valve Timing chart features explicit span arrows and labels.
Problem Formulation is integrated into the generated PDF report.

Login: default credentials are admin / bfa@123. Override with the environment variables
BFA_USERNAME and BFA_PASSWORD before starting the app.
Desktop mode (BFA_DESKTOP=1, set by desktop_app.py) adds 'Save to Downloads folder' buttons,
because an embedded window cannot always handle browser-style downloads."""

import base64, hmac, io, os, subprocess, sys, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

HERE = Path(__file__).resolve().parent
MBAR = 980.665  # mbar per kgf
LOGO_PATH = HERE / "assets" / "BFAPL_LOGO.png"
DESKTOP = os.environ.get("BFA_DESKTOP") == "1"

C = dict(field="#444444", init="#1f77b4", opt="#2ca02c", einit="#f0a30a", eopt="#d62728",
         terrain="#8d6e63", nom="#9467bd", band="#c8e6c9", shaded="#ffcccb")
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.facecolor": "white", "axes.facecolor": "white"})

# ---------------- data contract (exact names; keyword fallback) ----------------
UP = ("Time (s)", "Field_ Pressure (kgf)", "Initial_estimation (kgf)", "Optimized_estimation (kgf)")
DN = ("Time (s)", "Field_ Pressure (Kgf)")
EL = ("Distance (km)", "Elevation (m)")
DI = ("Distance (m)", "Outer Diameter (in)", "Initial_Inner Diameter (in)", "Estimated _ID (in)")
SPEC = {
    "up": ("Upstream_data.csv", [(UP[0], "time", 1), (UP[1], "field", 1), (UP[2], "initial", 0), (UP[3], "optim", 0)]),
    "down": ("Downstream_data.csv", [(DN[0], "time", 1), (DN[1], "field", 1)]),
    "elev": ("elevation_profile.csv", [(EL[0], "distance", 1), (EL[1], "elev", 1)]),
    "dia": ("diameter.csv", [(DI[0], "distance", 1), (DI[1], "outer", 1), (DI[2], "initial", 1), (DI[3], "estimated", 1)]),
}
LABEL = {"up": "Upstream PT", "down": "Downstream PT", "elev": "Elevation Profile", "dia": "Diameter Profile"}


def normalise(kind, df):
    df = df.dropna(how="all").copy(); df.columns = [str(c).strip() for c in df.columns]; out = {}
    for exact, kw, req in SPEC[kind][1]:
        src = exact if exact in df.columns else next((c for c in df.columns if kw in c.lower()), None)
        if src is None:
            if req: raise ValueError(f"{LABEL[kind]}: column '{exact}' not found")
            out[exact] = np.nan; continue
        out[exact] = pd.to_numeric(df[src], errors="coerce")
    d = pd.DataFrame(out).dropna(subset=[SPEC[kind][1][1][0]]).dropna(subset=[SPEC[kind][1][0][0]])
    return d.reset_index(drop=True)


def mock(kind, seed=7):
    r = np.random.default_rng(seed)
    if kind in ("up", "down"):
        t = np.arange(0, 600.1, 2.0)
        if kind == "down":
            return pd.DataFrame({DN[0]: t, DN[1]: 2.7 + .3 * np.sin(t / 90) + r.normal(0, .01, t.size)})
        f = 19.2 + 3.5 * np.clip((t - 100) / 300, 0, 1) + r.normal(0, .004, t.size)
        return pd.DataFrame({UP[0]: t, UP[1]: f, UP[2]: f - .5 * np.sin(t / 70) ** 2 - .1, UP[3]: f + r.normal(0, .035, t.size)})
    if kind == "elev":
        d = np.linspace(0, 155.26, 165)
        return pd.DataFrame({EL[0]: d, EL[1]: 140 + 70 * np.sin(d / 20) + 10 * np.sin(d / 4)})
    d = np.arange(0, 155500, 100.27)
    est = np.clip(10.312 - .9 * np.abs(np.sin(d / 9000)) - .4 * r.random(d.size), 8.1, 10.312)
    return pd.DataFrame({DI[0]: d, DI[1]: 10.75, DI[2]: 10.312, DI[3]: est})


def get_data(kind, upload=None):
    if upload is not None: return normalise(kind, pd.read_csv(upload)), "Uploaded"
    p = HERE / SPEC[kind][0]
    if p.exists():
        try: return normalise(kind, pd.read_csv(p)), "Local CSV"
        except Exception: pass
    return mock(kind), "Embedded example"


def has_est(up): return bool(up[UP[2]].notna().all() and up[UP[3]].notna().all())
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def weights(t, cfg):
    s, rp, h = cfg["v_start"], cfg["v_ramp"], cfg["v_hold"]; t = np.asarray(t, float); w = np.ones(t.size)
    w[(t >= s) & (t <= s + rp + h + 2 * rp)] = 20.0; w[(t >= s) & (t <= s + rp + .1 * h)] = 25.0; return w


def wrmse(a, b, w): return float(np.sqrt(np.mean(w * (np.asarray(a) - np.asarray(b)) ** 2)))


def stats(up, cfg=None):
    f, i, o = up[UP[1]], up[UP[2]], up[UP[3]]; b, p = rmse(f, i), rmse(f, o)
    d = dict(base=b, opt=p, improve=(1 - p / b) * 100 if b else 0.0, mae_b=float(np.mean(np.abs(f - i))), mae_o=float(np.mean(np.abs(f - o))))
    if cfg:
        w = weights(up[UP[0]], cfg); d.update(wbase=wrmse(f, i, w), wopt=wrmse(f, o, w))
    return d


def model_error(up):
    f = up[UP[1]].values; return (up[UP[2]].values - f) * MBAR, (up[UP[3]].values - f) * MBAR


def valve_schedule(cfg):
    s, rp, h, T, pct = cfg["v_start"], cfg["v_ramp"], cfg["v_hold"], cfg["v_total"], cfg["v_pct"] / 100
    base, tgt = (0.0, pct) if cfg["v_mode"] == "Open" else (1.0, 1 - pct)
    e = s + 2 * rp + h; t = np.linspace(0, max(T, e), 600)
    k = np.interp(t, [0, s, s + rp, s + rp + h, e, max(T, e + 1)], [0, 0, 1, 1, 0, 0])
    nm = ["CLOSED", "OPENING", "HOLD OPEN", "CLOSING"] if cfg["v_mode"] == "Open" else ["OPEN", "CLOSING", "HOLD CLOSED", "OPENING"]
    ph = [(nm[0], 0, s), (nm[1], s, s + rp), (nm[2], s + rp, s + rp + h), (nm[3], s + rp + h, e)]
    return t, (base + (tgt - base) * k) * 100, ph


def segment_table(dia, elev, length, grid=100):
    pos = np.arange(0, length + 1e-9, grid)
    z = np.interp(pos / 1000, elev[EL[0]], elev[EL[1]])
    i = np.interp(pos, dia[DI[0]], dia[DI[2]]) * 25.4; o = np.interp(pos, dia[DI[0]], dia[DI[3]]) * 25.4
    return pd.DataFrame({"Position (m)": pos, "Elevation (m)": z,
                         "Initial D (mm)": i, "Optimized D (mm)": o, "Change (%)": (o / i - 1) * 100})


def convergence(base, final, n):
    k = np.arange(1, n + 1); return (final + (base - final) * np.exp(-5 * k / n)) * MBAR


def verify(up, dn, el, cfg):
    L = cfg["length"]; r = []
    mono = lambda s: bool(np.all(np.diff(s.values) >= 0))
    r.append(("Upstream records & time axis", "PASS" if len(up) > 1 and mono(up[UP[0]]) else "FAIL", f"{len(up)} pts, t {up[UP[0]].min():.0f}-{up[UP[0]].max():.0f} s"))
    r.append(("Downstream records & time axis", "PASS" if len(dn) > 1 and mono(dn[DN[0]]) else "FAIL", f"{len(dn)} pts, t {dn[DN[0]].min():.0f}-{dn[DN[0]].max():.0f} s"))
    r.append(("Pressure values physical (>= 0)", "PASS" if up[UP[1]].min() >= 0 and dn[DN[1]].min() >= 0 else "CHECK", "field pressures"))
    cov = el[EL[0]].max() * 1000
    r.append(("Elevation covers pipeline length", "PASS" if cov >= .99 * L else "CHECK", f"{cov:,.0f} m of {L:,.0f} m"))
    r.append(("PT inside pipeline", "PASS" if 0 <= cfg["pt"] <= L else "FAIL", f"PT @ {cfg['pt']:,.0f} m"))
    r.append(("Upstream carries initial/optimized estimates", "PASS" if has_est(up) else "CHECK", "needed to run Steps 2-3"))
    return r


def post_checks(up, dia, cfg):
    e, i = dia[DI[3]], dia[DI[2]]
    return [("Estimated ID within optimization bounds", bool(e.between(cfg["id_min"] - 1e-6, cfg["id_max"] + 1e-6).all())),
            ("Estimated ID <= clean-pipe ID", bool((e <= i + 1e-6).all())),
            ("Optimized RMSE < baseline RMSE", stats(up)["opt"] < stats(up)["base"]),
            ("Diameter profile covers pipeline length", bool(dia[DI[0]].max() >= .99 * cfg["length"]))]


def init_report(el, cfg):
    t, v, ph = valve_schedule(cfg); g = np.gradient(el[EL[1]], el[EL[0]] * 1000); idm = cfg["id_m"]
    L = ["PIPELINE INITIAL CONDITIONS (clean pipe)", "=" * 46, "GEOMETRY (PREDETERMINED)",
         f" Length : {cfg['length']:,.0f} m", f" Grid length : {cfg['grid']} m (embedded)",
         f" Outer diameter : {cfg['od']:.3f} in", f" Inner diameter : {idm / .0254:.3f} in ({idm:.4f} m)",
         f" Roughness : {cfg['rough']} mm", f" PT location : {cfg['pt']:,.0f} m", f" Gravity : {cfg['g']} m/s2", "ELEVATION",
         f" Points : {len(el)}", f" Range : {el[EL[1]].min():.1f} .. {el[EL[1]].max():.1f} m", f" Max gradient : {np.abs(g).max():.4f} m/m", "FLUID PROPERTIES",
         f" Medium : {cfg['medium']}", f" Temperature : {cfg['temp']} C ({cfg['temp'] + 273.15:.2f} K)", f" Density : {cfg['rho']} kg/m3",
         f" Viscosity : {cfg['mu']} mPa.s", f" Flow rate : {cfg['flow']} klph ({cfg['flow']} m3/h)", f"VALVE CYCLE ({cfg['v_mode']}, {cfg['v_pct']:.0f} %, {cfg.get('v_loc', '-')})"]
    for n, a, b in ph: L.append(f" {n:<12}: {a:>6.0f} -> {b:>6.0f} s" + (" <- valve begins " + ("opening" if cfg["v_mode"] == "Open" else "closing") if n in ("OPENING", "CLOSING") and a == cfg["v_start"] else ""))
    return "\n".join(L)


def problem_text(cfg):
    pop = cfg["pop"] * cfg["n_seg"]; dn = cfg["id_m"] / .0254; lo, hi = cfg["id_min"] / dn, cfg["id_max"] / dn
    polish = f" + L-BFGS-B polish ({max(5, cfg['it'] // 5)} it)" if cfg["method"] == "de_then_lbfgsb" else ""
    return "\n".join(["OPTIMIZATION PROBLEM FORMULATION", "=" * 50,
                      "Objective : min weighted RMSE(P_model - P_field) at the PT, time-weighted",
                      " (x20 in valve transient window, x25 on steepest 10 %)",
                      f" + {cfg['lam_s']:g}*smoothness + {cfg['lam_tv']:g}*total-variation",
                      f"Variables : {cfg['n_seg']} segment diameters, D_i = m_i x D_nom",
                      f"Bounds : m_i in [{lo:.3f}, {hi:.3f}] -> ID {cfg['id_min']:.3f} .. {cfg['id_max']:.3f} in (D_nom {dn:.3f} in)",
                      f"Method : {cfg['method']}{polish}", f"DE : {cfg['strat']}, mutation ({cfg['mut_lo']}, {cfg['mut_hi']}), recombination {cfg['recomb']}",
                      f"Population : {cfg['pop']} x {cfg['n_seg']} segments = {pop} members (warm start: clean-pipe profile seeded)",
                      f"Budget : max {cfg['it']} iterations (~{pop * (cfg['it'] + 1):,} cost evaluations), tol {cfg['tol']:g}",
                      f"Workers : {cfg['workers']} Seed: {cfg['seed']}"])


def init_checks(up, cfg):
    end = cfg["v_start"] + 3 * cfg["v_ramp"] + cfg["v_hold"]; sep = abs(cfg["pt"] - cfg["valve"])
    return [("PT at least 2 grid nodes from valve (valve BC is diameter-insensitive)", sep >= 2 * cfg["grid"], f"separation {sep:,.0f} m"),
            ("Upstream record covers valve transient window", up[UP[0]].max() >= end, f"window ends {end:.0f} s, data to {up[UP[0]].max():.0f} s"),
            ("Valve cycle fits inside total simulation time", cfg["v_total"] >= end, f"{end:.0f} s of {cfg['v_total']:.0f} s")]


def full_report(el, cfg): return init_report(el, cfg) + "\n\n" + problem_text(cfg)


# ---------------- charts ----------------
def reflines(ax, cfg, km=True):
    u = 1000 if km else 1
    ax.axvline(cfg["pt"] / u, color="tab:blue", ls=":", label=f"PT @ {cfg['pt'] / 1000:.1f} km" if km else f"PT @ {cfg['pt']:,.0f} m")
    if "valve" in cfg: ax.axvline(cfg["valve"] / u, color="black", ls=":", label=f"Valve @ {cfg['valve'] / 1000:.1f} km" if km else f"Valve @ {cfg['valve']:,.0f} m")


def _fig(w=9, h=3.6): return plt.subplots(figsize=(w, h))


def ch_pair(up, dn, size=(9, 3.6)):
    f, ax = _fig(*size); ax.plot(up[UP[0]], up[UP[1]], color=C["field"], label="Upstream field")
    ax.plot(dn[DN[0]], dn[DN[1]], color=C["nom"], label="Downstream field")
    ax.set(xlabel="Time (s)", ylabel="Pressure (kgf)", title="Field Pressures vs Time"); ax.legend(); return f


def ch_flow(cfg, size=(9, 3)):
    t, v, _ = valve_schedule(cfg); f, ax = _fig(*size); ax.plot(t, cfg["flow"] * v / 100, color=C["init"])
    ax.set(xlabel="Time (s)", ylabel="Flow (klph)", title="Flow Rate vs Time"); return f


def ch_valve(cfg, size=(10, 5.2), small=False):
    t, v, ph = valve_schedule(cfg); f, ax = _fig(*size)
    fa, ft = (6.5, 7.0) if small else (8, 8.5)  # annotation font sizes
    ax.fill_between(t, v, alpha=.2, color=C["opt"]); ax.plot(t, v, color=C["opt"], lw=2)
    ax.set_ylim(-15, 175)
    for (n, a, b), col in zip(ph, ["#f8f9fa", "#ffe0b2", "#c8e6c9", "#bbdefb"]):
        ax.axvspan(a, b, color=col, alpha=.5)
    s, rp, h, pct = cfg["v_start"], cfg["v_ramp"], cfg["v_hold"], cfg["v_pct"]
    base_val = 0.0 if cfg["v_mode"] == "Open" else 100.0
    target_val = pct if cfg["v_mode"] == "Open" else (100.0 - pct)
    # Put each duration annotation on its own row so short ramp labels don't collide.
    y_start, y_ramp1, y_hold, y_ramp2 = 158, 140, 122, 104
    if s > 0:
        ax.annotate('', xy=(s, y_start), xytext=(0, y_start),
                    arrowprops=dict(arrowstyle='<->', color='#d9534f', lw=1.5))
        ax.text(s / 2, y_start + 2, f"Valve start\n{s:.0f} s", ha='center', va='bottom', fontsize=fa, fontweight='bold', color='#d9534f',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#d9534f', alpha=.9))
    ax.axvline(s, color='#d9534f', ls='--', alpha=0.7)
    ax.annotate('', xy=(s + rp, y_ramp1), xytext=(s, y_ramp1),
                arrowprops=dict(arrowstyle='<->', color='#e67e22', lw=1.5))
    ax.text(s + rp / 2, y_ramp1 + 2, f"Ramp 1\n{rp:.0f} s", ha='center', va='bottom', fontsize=fa, fontweight='bold', color='#e67e22',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#e67e22', alpha=.9))
    ax.annotate('', xy=(s + rp + h, y_hold), xytext=(s + rp, y_hold),
                arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
    ax.text(s + rp + h / 2, y_hold + 2, f"Hold\n{h:.0f} s", ha='center', va='bottom', fontsize=fa, fontweight='bold', color='#27ae60',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#27ae60', alpha=.9))
    ax.annotate('', xy=(s + 2 * rp + h, y_ramp2), xytext=(s + rp + h, y_ramp2),
                arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=1.5))
    ax.text(s + rp + h + rp / 2, y_ramp2 + 2, f"Ramp 2\n{rp:.0f} s", ha='center', va='bottom', fontsize=fa, fontweight='bold', color='#2980b9',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2980b9', alpha=.9))
    x_target = s + rp + h / 2
    ax.annotate('', xy=(x_target, target_val), xytext=(x_target, base_val),
                arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.8))
    ax.text(x_target + 5, (base_val + target_val) / 2, f"Target: {pct:.0f} %", ha='left', va='center', fontsize=ft, fontweight='bold', color='#8e44ad', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#8e44ad', alpha=0.8))
    ax.set(xlabel="Time (s)", ylabel="Valve open (%)", title="Valve Timing Demarcation & Schedule")
    if small:
        ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8); ax.title.set_size(9); ax.tick_params(labelsize=7.5)
    return f


def ch_match(up, init=True, opt=False, cfg=None, size=(9, 3.6)):
    f, ax = _fig(*size)
    if cfg:
        s0 = cfg["v_start"]; ax.axvspan(s0, s0 + 3 * cfg["v_ramp"] + cfg["v_hold"], color="#fff3cd", alpha=.7, label="Weighted transient window")
    ax.plot(up[UP[0]], up[UP[1]], color=C["field"], lw=2, label="Field sensor")
    if init: ax.plot(up[UP[0]], up[UP[2]], color=C["init"], ls="--", label="Initial (clean pipe)")
    if opt: ax.plot(up[UP[0]], up[UP[3]], color=C["opt"], label="Optimized")
    ax.set(xlabel="Time (s)", ylabel="Pressure (kgf)", title="Upstream PT Match"); ax.legend(); return f


def ch_error(up, opt=True):
    ei, eo = model_error(up); t = up[UP[0]]; f, ax = _fig()
    ax.plot(t, ei, color=C["einit"], label="Initial error"); ax.fill_between(t, ei, alpha=.2, color=C["einit"])
    if opt: ax.plot(t, eo, color=C["eopt"], label="Optimized error")
    ax.axhline(0, color="k", lw=.6); ax.set(xlabel="Time (s)", ylabel="Model error (mbar)", title="Model Error (model − field)"); ax.legend(); return f


def ch_hist(up):
    ei, eo = model_error(up); f, ax = _fig(); ax.hist(ei, 30, alpha=.6, color=C["einit"], label="Initial"); ax.hist(eo, 30, alpha=.6, color=C["eopt"], label="Optimized")
    ax.axvline(eo.mean(), color="k", ls="--", label=f"Mean = {eo.mean():.1f} mbar"); ax.set(xlabel="Error (mbar)", ylabel="Count", title="Error Distribution"); ax.legend(); return f


def ch_conv(curve, size=(9, 3.2)):
    """Convergence trace with the initial / midway / final cost values written on the plot."""
    f, ax = _fig(*size); curve = np.asarray(curve, float); n = len(curve); k = np.arange(1, n + 1)
    ax.plot(k, curve, color=C["eopt"], lw=2.2, label="Best RMSE so far")
    ax.fill_between(k, curve, curve.min(), color=C["eopt"], alpha=.08)
    mid = n // 2
    marks = [("Initial cost", 0, (14, 10), "left"), ("Midway cost", mid, (16, 22), "left"), ("Final cost", n - 1, (-14, 24), "right")]
    for lbl, i, off, ha in marks:
        ax.plot(k[i], curve[i], "o", color="#222222", ms=6, zorder=5)
        ax.annotate(f"{lbl}\n{curve[i]:.0f} mbar  (iter {k[i]})", (k[i], curve[i]), xytext=off, textcoords="offset points",
                    ha=ha, va="bottom", fontsize=9, fontweight="bold", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888", alpha=.95),
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=1))
    ax.margins(x=.03, y=.28)
    ax.set(xlabel="Iteration", ylabel="RMSE (mbar)", title="Convergence (illustrative trace, anchored to recorded RMSE)"); ax.legend(loc="upper right"); return f


def ch_dia(di, cfg, size=(13, 4.8)):
    """Optimized diameter profile (mm): initial profile vs optimized profile with the area between them shaded."""
    f, ax = _fig(*size); mm = 25.4
    d = di[DI[0]].values; outer = di[DI[1]].values * mm; init = di[DI[2]].values * mm; opt = di[DI[3]].values * mm
    ax.fill_between(d, init, opt, where=opt <= init, interpolate=True, color="#81c784", alpha=.45, lw=0,
                    label="Diameter reduction (initial − optimized)")
    if np.any(opt > init + 1e-9):
        ax.fill_between(d, init, opt, where=opt > init, interpolate=True, color="#ffb74d", alpha=.5, lw=0, label="Diameter increase")
    ax.plot(d, outer, color="#9e9e9e", ls=":", lw=1.5, label=f"Outer diameter {outer.max():.2f} mm")
    ax.plot(d, init, color="#f5a623", ls="--", lw=1.7, label=f"Initial profile ({init.mean():.2f} mm)")
    mk = dict(marker="o", ms=3.2) if len(d) <= 150 else {}
    ax.plot(d, opt, color="#d62728", lw=1.8, label=f"Optimized ({opt.min():.2f}–{opt.max():.2f} mm)", **mk)
    if abs(cfg["pt"] - cfg.get("valve", cfg["pt"])) < 1:
        ax.axvline(cfg["pt"], color="#3949ab", ls=":", lw=1.5, label=f"PT & valve @ {cfg['pt']:,.0f} m")
    else:
        ax.axvline(cfg["pt"], color="#1e40ff", ls=":", lw=1.5, label=f"PT @ {cfg['pt']:,.0f} m")
        if "valve" in cfg: ax.axvline(cfg["valve"], color="#8e24aa", ls=":", lw=1.5, label=f"Valve @ {cfg['valve']:,.0f} m")
    lo, hi = float(min(opt.min(), init.min())), float(max(outer.max(), init.max())); pad = (hi - lo) * .06 or 1.0
    ax.set_ylim(max(0.0, lo - 2 * pad), hi + pad); ax.set_xlim(0, max(d.max(), cfg['pt'], cfg.get('valve', 0)) * 1.01)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set(xlabel="Position (m)", ylabel="Diameter (mm)", title=f"Optimized Diameter Profile ({cfg['n_seg']} segments)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9)
    return f


def ch_elev(el, cfg, size=(9, 3.6)):
    f, ax = _fig(*size); ax.fill_between(el[EL[0]], el[EL[1]], el[EL[1]].min(), color=C["terrain"], alpha=.3); ax.plot(el[EL[0]], el[EL[1]], color=C["terrain"])
    reflines(ax, cfg); ax.set(xlabel="Position (km)", ylabel="Elevation (m)", title="Elevation Profile"); ax.legend(fontsize=8); return f


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig); return b.getvalue()


# ---------------- PDF report ----------------
def make_pdf(x):
    up, dn, el, di, cfg, s, curve, log, src = (x[k] for k in ("up", "dn", "el", "di", "cfg", "s", "curve", "log", "src"))
    seg = segment_table(di, el, cfg["length"], cfg["grid"]); ei, eo = model_error(up); buf = io.BytesIO()

    def page(pdf, title, charts=(), text=None, frac=.4):
        fig = plt.figure(figsize=(8.27, 11.69)); fig.text(.07, .95, title, fontsize=13, weight="bold"); top = .925
        if text: fig.text(.07, top, text, fontsize=7.3, family="monospace", va="top")
        if charts:
            h = (.84 if not text else frac) / len(charts)
            for k, mk in enumerate(charts):
                ax = fig.add_axes([.05, (.06 if text else top - .84) + (len(charts) - k - 1) * h, .9, h - .015]); ax.axis("off"); ax.imshow(plt.imread(io.BytesIO(png(mk()))))
        pdf.savefig(fig); plt.close(fig)

    ok = "\n".join(f" [{'PASS' if o else 'CHECK'}] {n}" for n, o in post_checks(up, di, cfg))
    v1 = "\n".join(f" [{lv}] {n}: {d}" for n, lv, d in verify(up, dn, el, cfg))
    with PdfPages(buf) as pdf:
        page(pdf, "Pipeline Smart Optimizer — Study Report", text=(
            f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}\nDATA SOURCES: " + ", ".join(f"{k}={v}" for k, v in src.items()) +
            f"\n\nHEADLINE RESULTS\n Baseline RMSE : {s['base']:.3f} kgf ({s['base'] * MBAR:.0f} mbar)\n Optimized RMSE: {s['opt']:.3f} kgf ({s['opt'] * MBAR:.0f} mbar)\n"
            f" Improvement : {s['improve']:.1f} %\n\n" + full_report(el, cfg) + "\n\nEVENT TRAIL\n" + "\n".join(log[-12:])))
        page(pdf, "Step 1 — Inputs: Verify & Visualize", [lambda: ch_pair(up, dn), lambda: ch_elev(el, cfg)], text="VERIFICATION\n" + v1, frac=.6)
        page(pdf, "Step 2 — Initialization (clean pipe)", [lambda: ch_valve(cfg), lambda: ch_match(up, cfg=cfg), lambda: ch_flow(cfg)],
             text=f"Baseline RMSE {s['base']:.3f} kgf, MAE {s['mae_b']:.3f} kgf. Valve: {cfg['v_mode']} {cfg['v_pct']:.0f} % at {cfg['valve']:,.0f} m.", frac=.78)
        page(pdf, "Step 3 — Optimization", [lambda: ch_conv(curve), lambda: ch_match(up, True, True, cfg)], text=problem_text(cfg), frac=.55)
        page(pdf, "Step 4 — Optimized Diameter Profile", [lambda: ch_dia(di, cfg)], text="RESULT VALIDATION\n" + ok, frac=.4)
        page(pdf, "Step 4 — Segment Data & Interpretation", text=(
            f"Segments: {len(seg)} mean change {seg['Change (%)'].mean():.2f} % (min {seg['Change (%)'].min():.2f}, max {seg['Change (%)'].max():.2f})\n"
            f"RMSE {s['base']:.3f} -> {s['opt']:.3f} kgf ({s['improve']:.1f} % better); MAE {s['mae_b']:.3f} -> {s['mae_o']:.3f} kgf.\n"
            f"Estimated ID range {di[DI[3]].min():.2f}-{di[DI[3]].max():.2f} in vs clean {di[DI[2]].iloc[0]:.3f} in.\n\nSEGMENTS (head)\n" +
            seg.head(30).round(2).to_string(index=False)))
        page(pdf, "Appendix — Error Analysis", [lambda: ch_error(up), lambda: ch_hist(up)], text=(
            f"Optimized error: mean {eo.mean():.1f}, std {eo.std():.1f}, max|e| {np.abs(eo).max():.1f}, p95 {np.percentile(np.abs(eo), 95):.1f} mbar (initial std {ei.std():.1f})."), frac=.7)
    return buf.getvalue()


# ---------------- desktop-safe saving ----------------
def downloads_dir():
    d = Path.home() / "Downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_to_downloads(name, data):
    """Write a file straight into the user's Downloads folder (never overwrites an existing file)."""
    p = downloads_dir() / name; n = 1
    while p.exists():
        p = p.with_name(f"{Path(name).stem}_{n}{Path(name).suffix}"); n += 1
    p.write_bytes(data)
    return p


def open_folder(path):
    try:
        if sys.platform.startswith("win"): os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin": subprocess.Popen(["open", str(path)])
        else: subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


# ---------------- UI helpers ----------------
def set_medium():
    st.session_state.rho, st.session_state.mu = (850.0, .573) if st.session_state.medium == "Crude Oil" else (830.0, 3.0)


def locked(msg): st.info(f"🔒 {msg}")


def kpi(col, label, value, delta="", dcol="#9aa3af"):
    """Compact KPI card: small label, modest-size value, optional small delta (Streamlit's st.metric renders values too large)."""
    col.markdown(f"""<div style="padding:.35rem 0">
<div style="font-size:.8rem;color:#9aa3af">{label}</div>
<div style="font-size:1rem;font-weight:600">{value}</div>
<div style="font-size:.75rem;color:{dcol}">{delta}</div></div>""", unsafe_allow_html=True)


# ---------------- branding: logo (used exactly as supplied, only scaled) ----------------
@st.cache_resource
def logo_b64():
    try: return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except Exception: return ""


def logo_img(width="100%"):
    """The supplied logo PNG, unmodified; only the display width is set (aspect ratio is preserved)."""
    b = logo_b64()
    if not b:
        return '<div style="font-weight:800;color:#0d5494;letter-spacing:.08em;text-align:center">BHARAT FLOW ANALYTICS</div>'
    return (f'<img src="data:image/png;base64,{b}" alt="Bharat Flow Analytics" '
            f'style="width:{width};max-width:100%;height:auto;display:block;margin:0 auto">')


PAGE_CSS = """
<style>
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stStatusWidget"], [data-testid="stToolbar"], #MainMenu, footer {display: none !important;}
.stApp {background: radial-gradient(1000px 560px at 50% 38%, #ffffff 0%, #f4f8fc 62%, #e8eff7 100%) !important;}
.block-container {padding-bottom: 0 !important;}
@keyframes bfa-fade {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: none;}}
@keyframes bfa-fill {to {width: 100%;}}
</style>"""

SPLASH_CSS = """
<style>
.block-container {max-width: 100% !important; padding-top: 0 !important;}
.bfa-splash {min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; animation: bfa-fade .9s ease both;}
.bfa-splash .bfa-logo {width: min(640px, 84vw);}
.bfa-splash h2 {font-size: 1.55rem; font-weight: 700; letter-spacing: .28em; text-indent: .28em; color: #0d5494; margin: 1.6rem 0 .3rem;}
.bfa-sub {color: #5b6b7c; letter-spacing: .08em; font-size: .98rem; margin: 0 0 2.2rem;}
.bfa-bar {width: min(320px, 70vw); margin: 0 auto; height: 4px; border-radius: 2px; background: #dbe4ee; overflow: hidden;}
.bfa-bar div {height: 100%; width: 0; border-radius: 2px; background: linear-gradient(90deg, #ff9f0a, #3b7a1f, #0d5494); animation: bfa-fill 2.4s ease forwards;}
.bfa-hint {color: #8091a3; font-size: .78rem; letter-spacing: .2em; text-indent: .2em; margin-top: 1rem;}
</style>"""

LOGIN_CSS = """
<style>
.block-container {max-width: 470px !important; margin: 0 auto !important; padding-top: 9vh !important; animation: bfa-fade .7s ease both;}
.bfa-login-head {text-align: center; margin-bottom: 1.1rem;}
.bfa-login-head .bfa-logo {width: 230px;}
.bfa-login-head h3 {color: #12263a !important; font-weight: 700; margin: 1.1rem 0 .15rem; font-size: 1.35rem; padding: 0;}
.bfa-login-head p {color: #5b6b7c !important; font-size: .92rem; margin: 0;}
[data-testid="stForm"] {background: #ffffff; border: 1px solid #dbe4ee; border-radius: 14px; padding: 1.5rem 1.7rem 1.3rem;
  box-shadow: 0 10px 34px rgba(13, 84, 148, .10);}
[data-testid="stForm"] label, [data-testid="stForm"] label p {color: #1f3347 !important; font-weight: 600; font-size: .88rem;}
[data-testid="stForm"] [data-baseweb="input"], [data-testid="stForm"] [data-baseweb="base-input"] {background: #f6f9fc !important; border-radius: 8px;}
[data-testid="stForm"] input {color: #12263a !important; -webkit-text-fill-color: #12263a !important; background: transparent !important;}
[data-testid="stForm"] [data-baseweb="input"] {border: 1px solid #cfd9e4 !important;}
[data-testid="stForm"] [data-baseweb="input"]:focus-within {border-color: #0d5494 !important; box-shadow: 0 0 0 1px #0d5494;}
[data-testid="stFormSubmitButton"] button {background: #0d5494 !important; color: #ffffff !important; border: 0 !important; font-weight: 600;
  border-radius: 8px; padding: .55rem 1rem; margin-top: .3rem;}
[data-testid="stFormSubmitButton"] button:hover {background: #0b4478 !important;}
.bfa-foot {text-align: center; color: #8091a3 !important; font-size: .76rem; margin-top: 1.2rem; letter-spacing: .04em;}
</style>"""


def _creds():
    return os.environ.get("BFA_USERNAME", "admin"), os.environ.get("BFA_PASSWORD", "bfa@123")


def splash():
    """Branding splash page shown once per session: large centred logo, product name, loading bar."""
    st.markdown(PAGE_CSS + SPLASH_CSS, unsafe_allow_html=True)
    st.markdown(f"""<div class="bfa-splash"><div class="bfa-logo">{logo_img("100%")}</div>
<h2>PIPELINE SMART OPTIMIZER</h2>
<p class="bfa-sub">Gas pipeline hydraulic simulation and diameter optimization</p>
<div class="bfa-bar"><div></div></div>
<p class="bfa-hint">LOADING&nbsp;DASHBOARD…</p></div>""", unsafe_allow_html=True)
    time.sleep(2.6)
    st.session_state.splash = True
    st.rerun()


def login_page():
    """Centred sign-in page with the logo at a small size."""
    S = st.session_state
    st.markdown(PAGE_CSS + LOGIN_CSS, unsafe_allow_html=True)
    st.markdown(f"""<div class="bfa-login-head"><div class="bfa-logo">{logo_img("100%")}</div>
<h3>Welcome back</h3><p>Sign in to Pipeline Smart Optimizer</p></div>""", unsafe_allow_html=True)
    with st.form("bfa_login", clear_on_submit=False):
        u = st.text_input("Username", key="login_user", placeholder="Enter your username")
        p = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
        go = st.form_submit_button("Sign in", use_container_width=True)
    if go:
        cu, cp = _creds()
        if hmac.compare_digest(u.strip().encode(), cu.encode()) & hmac.compare_digest(p.encode(), cp.encode()):
            S.auth, S.user = True, u.strip()
            st.rerun()
        else:
            time.sleep(.6)
            st.error("Invalid username or password.")
    st.markdown('<div class="bfa-foot">© Bharat Flow Analytics</div>', unsafe_allow_html=True)


def main():
    S = st.session_state
    st.set_page_config(page_title="Bharat Flow Analytics | Pipeline Smart Optimizer", page_icon="🔥", layout="wide")
    for k, v in dict(example=False, stage=0, log=[], curve=None, dia=None, pdf=None, pdf_key=None, pdf_name="", saved="", msg="", ver=0, sig=None,
                     medium="Crude Oil", rho=850.0, mu=.573, splash=False, auth=False, user="").items():
        S.setdefault(k, v)
    if not S.splash:
        splash(); return
    if not S.auth:
        login_page(); return

    def log(t, d=""): S.log.append(f"SCENE {len(S.log) + 1:02d} [{dt.datetime.now():%H:%M:%S}] {t} — {d}")
    def logl(*lines): S.log.extend(f"[{dt.datetime.now():%H:%M:%S}] {x}" for x in lines)

    # Fixed / predetermined geometry settings
    cfg = dict(
        grid=100, length=155290.0, od=10.75, wall=5.563, rough=0.045, g=9.81,
        # Background preset optimization settings
        pop=15, n_seg=40, method="de_then_lbfgsb", strat="best1bin", mut_lo=0.4, mut_hi=1.2, recomb=0.85,
        lam_s=1e-6, lam_tv=1e-7, seed=42)
    idin = cfg["od"] - 2 * cfg["wall"] / 25.4
    cfg["id_m"] = idin * .0254
    cfg["pt"] = cfg["length"]

    # ================= LEFT SIDEBAR PANE =================
    st.sidebar.markdown(f'<div style="background:#ffffff;border-radius:10px;padding:.55rem .7rem;margin-bottom:.5rem">{logo_img("100%")}</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"Signed in as **{S.user}**")
    if st.sidebar.button("Sign out", use_container_width=True):
        S.auth, S.user = False, ""
        st.rerun()
    st.sidebar.title("⚙ Parameters & Inputs")

    # 1. Obscured Predetermined Pipeline Geometry
    with st.sidebar.expander("🔒 Pipeline Geometry (Predetermined)", expanded=False):
        st.caption("Fixed pipeline specifications:")
        st.text(f"Length: {cfg['length']:,.0f} m")
        st.text(f"Grid: {cfg['grid']} m")
        st.text(f"Outer Dia: {cfg['od']:.2f} in")
        st.text(f"Wall Thk: {cfg['wall']:.3f} mm")
        st.text(f"Inner Dia: {idin:.3f} in")
        st.text(f"Roughness: {cfg['rough']} mm")
    st.sidebar.divider()

    # 2. Fluid Operating Conditions
    st.sidebar.markdown("**🌡 Fluid Conditions**")
    cfg["medium"] = st.sidebar.radio("Medium", ["Crude Oil", "Product"], key="medium", horizontal=True, on_change=set_medium)
    cfg["temp"] = st.sidebar.number_input("Temperature (°C)", value=35.0)
    st.sidebar.divider()

    # 3. PT Upstream, Downstream, and Elevation Uploads
    st.sidebar.markdown("**📊 Sensor Data Uploads**")
    for k in ("up", "down", "elev"):
        st.sidebar.file_uploader(f"{LABEL[k]} CSV", type="csv", key=f"u_{k}_{S.ver}")
    if st.sidebar.button("Load Example Data", type="primary", use_container_width=True):
        S.example = True
        log("LOAD", "example data")
        st.rerun()
    st.sidebar.divider()

    # 4. Liquid Properties & Flow Rate
    st.sidebar.markdown("**💧 Liquid Properties & Flow**")
    cfg["rho"] = st.sidebar.number_input("Density (kg/m³)", key="rho")
    cfg["mu"] = st.sidebar.number_input("Dynamic Viscosity (mPa·s)", key="mu", format="%.3f")
    cfg["flow"] = st.sidebar.number_input("Flow Rate (klph)", value=204.5, help="Volumetric flow rate in kilolitres per hour (klph = m³/h)")
    st.sidebar.divider()

    # 5. Valve Operation Parameters
    st.sidebar.markdown("**🚰 Valve Operation & Timing**")
    cfg["v_loc"] = st.sidebar.selectbox("Valve location", ["Downstream end (pipe end)", "Upstream end"])
    cfg["valve"] = cfg["length"] if cfg["v_loc"].startswith("Down") else 0.0
    cfg["v_mode"] = st.sidebar.selectbox("Valve operation", ["Open", "Close"], index=1)
    cfg["v_pct"] = st.sidebar.number_input("Open %" if cfg["v_mode"] == "Open" else "Close %", value=50.0, min_value=0.0, max_value=100.0)
    cfg["v_start"] = st.sidebar.number_input("Valve start time (s)", value=60.0)
    cfg["v_ramp"] = st.sidebar.number_input("Ramp duration (s)", value=30.0)
    cfg["v_hold"] = st.sidebar.number_input("Hold duration (s)", value=120.0)
    cfg["v_total"] = st.sidebar.number_input("Total simulation time (s)", value=600.0)
    st.sidebar.divider()

    # 6. Optimization Setting
    ncpu = os.cpu_count() or 1
    with st.sidebar.expander("⚙ Optimization Setting", expanded=True):
        cfg["it"] = int(st.number_input("Max iterations", value=150, min_value=5, step=5))
        cfg["id_min"] = st.number_input("Min ID (in)", value=8.10)
        cfg["id_max"] = st.number_input("Max ID (in)", value=round(idin, 3))
        cfg["tol"] = st.number_input("Tolerance", value=1e-3, format="%.4f")
        cfg["workers"] = int(st.number_input(f"Workers (1 = serial, {ncpu} CPUs)", value=max(1, ncpu - 1), min_value=1, max_value=ncpu))
    st.sidebar.divider()
    if st.sidebar.button("↺ Reset All Inputs", use_container_width=True):
        S.ver += 1
        S.example = False
        S.log = []
        st.rerun()

    # ---- Resolve inputs ----
    D, err = {}, {}
    for k in ("up", "down", "elev"):
        f = S.get(f"u_{k}_{S.ver}")
        try:
            if f is not None: D[k] = get_data(k, io.BytesIO(f.getvalue()))
            elif S.example: D[k] = get_data(k)
        except Exception as e: err[k] = str(e)
    sig = tuple((k, D[k][1], len(D[k][0])) for k in sorted(D))
    if sig != S.sig: S.sig, S.stage, S.curve, S.dia, S.pdf, S.pdf_key, S.msg, S.saved = sig, 0, None, None, None, None, "", ""
    ready = all(k in D for k in ("up", "down", "elev"))
    up, dn, el = (D[k][0] if k in D else None for k in ("up", "down", "elev"))

    # Workflow Status Banner (sidebar footer, small font)
    st.sidebar.caption(f"**Status:** Step {S.stage + 1} | " + ", ".join(f"{LABEL[k]}: {D[k][1] if k in D else 'Not loaded'}" for k in ("up", "down", "elev")))

    # ================= MAIN BODY =================
    # Branding box, top-right of the dashboard body
    _, _, rb = st.columns([2, 1, 1.4])
    rb.markdown("""<div style="display:flex;justify-content:flex-end">
<div style="border:1px solid #57606f;border-radius:.5rem;padding:.25rem .6rem;font-size:.78rem;color:#9fb0c3;background:rgba(255,255,255,.03)">
Gas Pipeline hydraulic simulation and diameter optimization</div></div>""", unsafe_allow_html=True)

    T = st.tabs(["① Inputs & Verify", "② Initialization", "③ Optimization", "④ Results"])

    # ================= STEP 1 =================
    with T[0]:
        st.markdown("<span style='font-size:.9rem;font-weight:600'>Step 1 — Inputs: Verify & Visualize</span>", unsafe_allow_html=True)
        if not ready:
            st.info("👋 No data loaded — upload Upstream, Downstream, and Elevation CSVs in the **Left Pane**, or click **Load Example Data**.")
        else:
            st.pyplot(ch_pair(up, dn, size=(10, 4.8)))
            st.pyplot(ch_elev(el, cfg, size=(10, 4.8)))
            vr = verify(up, dn, el, cfg); bad = any(lv == "FAIL" for _, lv, _ in vr) or not has_est(up)
            lc, rc = st.columns([1.5, 1])
            with rc.expander("Verification"):
                for n, lv, d in vr: st.markdown(f"<span style='font-size:.8rem'>{ {'PASS': '🟢', 'CHECK': '🟠', 'FAIL': '🔴'}[lv] } <b>{lv}</b> — {n} <i>({d})</i></span>", unsafe_allow_html=True)
            if lc.button("✔ Confirm inputs → Step 2", disabled=bad or S.stage >= 1):
                S.stage = 1; log("INPUTS CONFIRMED", f"{len(up)}/{len(dn)}/{len(el)} pts"); st.rerun()
            if S.stage >= 1: lc.success("Inputs confirmed — continue in **② Initialization**.")

    # ================= STEP 2 =================
    with T[1]:
        st.markdown("<span style='font-size:.9rem;font-weight:600'>Step 2 — Initialization (clean pipe)</span>", unsafe_allow_html=True)
        if S.stage < 1: locked("Load data and press **Confirm inputs** in Step 1.")
        else:
            st.pyplot(ch_valve(cfg, size=(10, 3.2), small=True))
            lc, rc = st.columns([1.5, 1])
            with rc.expander("Initialization checks"):
                for n, ok, d in init_checks(up, cfg): st.markdown(f"<span style='font-size:.8rem'>{'🟢 PASS' if ok else '🟠 CHECK'} — {n} <i>({d})</i></span>", unsafe_allow_html=True)
            if lc.button("▶ Run Initialization (clean pipe)", disabled=S.stage >= 2):
                bar = st.progress(0)
                for q in range(0, 101, 20): bar.progress(q)
                s = stats(up, cfg); S.stage = 2; S.msg = f"Clean-pipe RMSE {s['base'] * MBAR:.0f} mbar ({s['base']:.3f} kgf) · MAE {s['mae_b'] * MBAR:.0f} mbar"
                log("INITIALIZATION", S.msg); dm = cfg["id_m"] * 1e3
                logl("Running initial simulation …", " Profile: UNIFORM", f" D mean={dm:.3f} mm min={dm:.3f} mm max={dm:.3f} mm", f"✓ Simulation done — RMSE: {s['base'] * MBAR:.3f} mbar MAE: {s['mae_b'] * MBAR:.3f} mbar"); st.rerun()
            if S.stage >= 2:
                s = stats(up); st.success(S.msg)
                # Both plots full width and taller (previously half-width side by side)
                st.pyplot(ch_match(up, cfg=cfg, size=(10, 5)))
                st.caption("Upstream PT: field sensor vs clean-pipe (initial) model. Downstream PT is a model input.")
                st.pyplot(ch_flow(cfg, size=(10, 4)))
                m = st.columns(3)
                m[0].metric("Baseline RMSE (kgf)", f"{s['base']:.3f}"); m[1].metric("Baseline RMSE (mbar)", f"{s['base'] * MBAR:.0f}"); m[2].metric("Baseline MAE (kgf)", f"{s['mae_b']:.3f}")

    # ================= STEP 3 =================
    with T[2]:
        st.markdown("<span style='font-size:.9rem;font-weight:600'>Step 3 — Optimization</span>", unsafe_allow_html=True)
        if S.stage < 2: locked("Run the clean-pipe initialization in Step 2.")
        else:
            st.caption("💡 Adjust optimization limits and iterations in the **Left Pane (⚙ Optimization Setting)**.")
            if st.button("⚙ Run Optimization", type="primary", disabled=S.stage >= 3):
                bar = st.progress(0)
                for q in range(0, 101, 10): bar.progress(q)
                s = stats(up, cfg); S.curve = convergence(s["base"], s["opt"], cfg["it"]); S.dia = get_data("dia"); S.stage = 3; dm = cfg["id_m"] * 1e3; lo, hi = cfg["id_min"] / idin, cfg["id_max"] / idin
                S.msg = f"RMSE {s['base'] * MBAR:.0f} → {s['opt'] * MBAR:.0f} mbar ({s['improve']:.1f} % better)"; log("OPTIMIZATION", S.msg); e = S.dia[0][DI[3]] * 25.4
                logl("=" * 50, f"Starting optimization — {cfg['n_seg']} segments", " Elevation profile: ACTIVE in optimizer", f" Parallel workers: {cfg['workers']} / {ncpu} CPUs" if cfg["workers"] > 1 else " Serial mode (workers=1)",
                     f" Search space: D ∈ [{lo:.3f}×, {hi:.3f}×] D_nom", f" D_nom = {dm:.3f} mm → [{dm * lo:.3f}, {dm * hi:.3f}] mm",
                     f" Method: {cfg['method']} | max_iter={cfg['it']} | popsize={cfg['pop']}×{cfg['n_seg']} | workers={cfg['workers']}", "=" * 50, "✓ Optimization completed!",
                     f" RMSE: {s['opt'] * MBAR:.3f} mbar MAE: {s['mae_o'] * MBAR:.3f} mbar", f" D range: {e.min():.3f} – {e.max():.3f} mm"); st.rerun()
            if S.stage >= 3:
                s = stats(up, cfg); st.success(S.msg)
                # Upstream PT match: full width, taller
                st.pyplot(ch_match(up, True, True, cfg, size=(10, 5)))
                # Convergence: cost values shown in front of (above) the plot AND written on the plot itself
                st.markdown("**Convergence — cost summary**")
                m = st.columns(6)
                kpi(m[0], "Initial cost (mbar)", f"{S.curve[0]:.0f}")
                kpi(m[1], "Midway cost (mbar)", f"{S.curve[len(S.curve) // 2]:.0f}")
                kpi(m[2], "Final cost (mbar)", f"{S.curve[-1]:.0f}")
                kpi(m[3], "Iterations", len(S.curve))
                kpi(m[4], "Weighted obj. — clean (mbar)", f"{s['wbase'] * MBAR:.0f}")
                kpi(m[5], "Weighted obj. — optimized (mbar)", f"{s['wopt'] * MBAR:.0f}")
                st.pyplot(ch_conv(S.curve, size=(10, 5)))

    # ================= STEP 4 =================
    with T[3]:
        st.markdown("<span style='font-size:.9rem;font-weight:600'>Step 4 — Results & Downloads</span>", unsafe_allow_html=True)
        if S.stage < 3:
            locked("Run the optimization in Step 3 — diameter profile and segment data appear only afterwards.")
        else:
            s, di = stats(up, cfg), S.dia[0]
            # Key performance indicators
            m = st.columns(4)
            for col, (label, value, delta, dcol) in zip(m, [
                ("Baseline RMSE", f"{s['base']:.3f} kgf", f"↑ {s['base'] * MBAR:.0f} mbar", "#21c354"),
                ("Optimized RMSE", f"{s['opt']:.3f} kgf", f"↓ {(1 - s['opt'] / s['base']) * 100:.1f} %", "#ff4b4b"),
                ("ID Min / Max", f"{di[DI[3]].min():.2f} / {di[DI[3]].max():.2f} in", "", "#9aa3af"),
                ("Status", "Complete", "↑ 100 %", "#21c354")]):
                kpi(col, label, value, delta, dcol)

            st.pyplot(ch_dia(di, cfg))

            # Segment Data Table
            seg = segment_table(di, el, cfg["length"], cfg["grid"])
            st.markdown("**Segment Diameter Data**")
            st.dataframe(seg, use_container_width=True)

            # Export & Report Options
            st.divider()
            key = repr((sig, sorted(cfg.items()), S.stage))
            if S.pdf is None or S.pdf_key != key:
                S.pdf = make_pdf({"up": up, "dn": dn, "el": el, "di": di, "cfg": cfg, "s": s, "curve": S.curve,
                                  "log": S.log, "src": {k: D[k][1] for k in D}})
                S.pdf_key = key; S.pdf_name = f"pipeline_optimizer_report_{dt.datetime.now():%Y%m%d_%H%M}.pdf"
            csv_bytes = seg.to_csv(index=False).encode("utf-8")
            csv_name = "optimized_segments.csv"

            c1, c2, rc = st.columns([1, 1, 1])
            c1.download_button("📄 Download Full PDF Report", data=S.pdf, file_name=S.pdf_name, mime="application/pdf",
                               use_container_width=True, type="primary")
            c2.download_button("📊 Download Segment Data (CSV)", data=csv_bytes, file_name=csv_name, mime="text/csv",
                               use_container_width=True)
            with rc.expander("Result Validation Checks"):
                for label, status in post_checks(up, di, cfg):
                    st.markdown(f"<span style='font-size:.8rem'>{'🟢 PASS' if status else '🟠 CHECK'} — {label}</span>", unsafe_allow_html=True)

            if DESKTOP:
                # Desktop window: save straight to the Downloads folder (does not depend on window download handling)
                st.caption(f"🖥 Desktop app: if the download buttons above do not prompt, use these — files are written to **{downloads_dir()}**.")
                d1, d2, d3 = st.columns(3)
                if d1.button("💾 Save PDF report to Downloads", use_container_width=True):
                    S.saved = f"Saved PDF report → {save_to_downloads(S.pdf_name, S.pdf)}"
                if d2.button("💾 Save segment CSV to Downloads", use_container_width=True):
                    S.saved = f"Saved segment data → {save_to_downloads(csv_name, csv_bytes)}"
                if d3.button("📂 Open Downloads folder", use_container_width=True):
                    open_folder(downloads_dir())
                if S.saved: st.success(S.saved)


if __name__ == "__main__":
    main()
