"""A3 section 4: label-free behaviour scoring of Slackline Level 1 TAGS (no labels are read here)."""
import ast, json, numpy as np, pandas as pd
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import BALLADEER_ROOT; ROOT = BALLADEER_ROOT.rstrip("/") + "/"
INV = pd.read_csv("balladeer/A3_file_inventory_121.csv")
FL = json.load(open(ROOT + "slackline_flags_info.json"))["slackline_levels_flags_info"]
L1 = [f for f in FL if f["level"] == "Level1"][0]["flags"]
N_TGT = sum(f["flag_type"] == "circle" for f in L1); N_NON = len(L1) - N_TGT
dur = max(f["flag_spawn_time"] for f in L1) / 60.0          # minutes of flag presentation (approx.)
rows = []
for _, r in INV.iterrows():
    T = pd.read_csv(ROOT + r.user + "/" + r.SlacklineLvl1_tagspath)
    ev = [ast.literal_eval(v)["reactionOrOmission"][0] for v in T["value"]]
    E = pd.DataFrame(ev)
    for c in ["reacted", "correct", "duplicated"]:
        E[c] = E[c].astype(str) == "True"
    E["ft"] = E.flagType.apply(lambda x: x[0])
    tgt = E.ft == 0; non = E.ft.isin([1, 2, 3])
    hits = E[tgt & E.reacted & ~E["duplicated"]]
    comm = E[non & E.reacted & ~E["duplicated"]]
    n_hit = len(hits)
    rt = hits.reactionTime.values
    rows.append(dict(subject=r.user, n_rows=len(E), n_hit=n_hit, n_comm=len(comm),
                     n_tgt_unreacted_rows=int((tgt & ~E.reacted).sum()),
                     n_dup=int(E["duplicated"].sum()), n_noflag=int((E.ft == -1).sum()),
                     flag_types_seen=",".join(map(str, sorted(E.ft.unique()))),
                     s2_ok=bool(n_hit <= N_TGT and len(comm) <= N_NON),
                     B1_omission_rate=(N_TGT - min(n_hit, N_TGT)) / N_TGT,
                     B2_commission_rate=min(len(comm), N_NON) / N_NON,
                     B3_mean_RT_hits=rt.mean() if n_hit >= 3 else np.nan,
                     B4_RT_cv_hits=rt.std(ddof=1) / rt.mean() if n_hit >= 3 else np.nan,
                     D_RT_sd_hits=rt.std(ddof=1) if n_hit >= 3 else np.nan,
                     D_dup_per_min=E["duplicated"].sum() / dur, D_noflag_per_min=(E.ft == -1).sum() / dur,
                     max_generalTime=E.generalTime.max()))
B = pd.DataFrame(rows); B.to_csv("derived/behaviour_L1_A3.csv", index=False)
print("targets", N_TGT, "non-targets", N_NON)
print("files parsed:", len(B), " s2_ok:", int(B.s2_ok.sum()), " hits<3:", int((B.n_hit < 3).sum()))
print(B.drop(columns=["subject", "flag_types_seen"]).describe().T[["mean", "min", "50%", "max"]].round(3).to_string())
print(B.flag_types_seen.value_counts().to_string())
