#!/usr/bin/env python
"""Cross-architecture generalization figure (fig:xarch).
Reads ../../experiments/results/cross_arch/cross_arch_data.json (NOT hard-coded)
and plots per-kernel library efficiency on A100 (sm_80) vs RTX 4000 Ada (sm_89).
Regenerate:  python gen_cross_arch.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"../../experiments/results/cross_arch/cross_arch_data.json")
d=json.load(open(DATA))["figure"]
fig,ax=plt.subplots(figsize=(5.0,4.2))
styles={"triton":dict(marker="o",color="#1f77b4",label="Triton"),
        "tilelang":dict(marker="^",color="#d62728",label="TileLang")}
annot={"layer_norm","conv2d","matmul","rms_norm"}
for impl,sty in styles.items():
    ada=d[impl]["ada"]; a100=d[impl]["a100"]
    xs,ys,ks=[],[],[]
    for k in sorted(set(ada)&set(a100)):
        x,y=ada[k],a100[k]
        if x>0 and y>0: xs.append(x); ys.append(y); ks.append(k)
    ax.scatter(xs,ys,s=42,edgecolor="k",linewidth=0.4,alpha=0.85,zorder=3,**sty)
    for x,y,k in zip(xs,ys,ks):
        if k in annot:
            ax.annotate(k,(x,y),fontsize=6.8,xytext=(4,3),textcoords="offset points",zorder=4)
lim=[0.05,2000]
ax.plot(lim,lim,"--",color="gray",lw=1,zorder=1,label="parity (Ada = A100)")
ax.axhline(100,color="k",lw=0.5,alpha=0.3); ax.axvline(100,color="k",lw=0.5,alpha=0.3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"RTX 4000 Ada (sm_89)  $E_\mathrm{lib}$ (%)")
ax.set_ylabel(r"A100-PCIE (sm_80)  $E_\mathrm{lib}$ (%)")
ax.set_title("Per-kernel DSL efficiency tracks across architectures",fontsize=9)
ax.legend(fontsize=7,loc="lower right",framealpha=0.9)
ax.grid(True,which="both",ls=":",alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE,"cross_arch_efficiency.pdf"))
fig.savefig(os.path.join(HERE,"cross_arch_efficiency.png"),dpi=150)
print("wrote cross_arch_efficiency.pdf / .png")
