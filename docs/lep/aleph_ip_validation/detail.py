import glob, numpy as np, awkward as ak
def rsig(x):
    x=x[np.isfinite(x)]; q75,q25=np.percentile(x,[75,25]); return (q75-q25)/1.349
R={k:[] for k in "dF d0e z0v vfar dbs".split()}
for f in sorted(glob.glob("/local/laurits/ALEPH_ntuples/events/data_ALEPH1994_000[0-4]0_1.parquet")):
    t=ak.from_parquet(f,columns=["event_has_vertex","event_vertex_x","event_vertex_y","event_vertex_z","part_px","part_py","part_has_track","part_d0","part_d0_err","part_phi0","part_z0","part_z0_err"])
    hv=ak.to_numpy(t.event_has_vertex).astype(bool)
    vxa=ak.to_numpy(t.event_vertex_x); vya=ak.to_numpy(t.event_vertex_y)
    mx,my=np.median(vxa[hv]),np.median(vya[hv])
    far=hv&((np.abs(vxa-mx)>1)|(np.abs(vya-my)>1))
    vx=ak.Array(vxa); vy=ak.Array(vya); vz=ak.Array(ak.to_numpy(t.event_vertex_z))
    pt=np.sqrt(t.part_px**2+t.part_py**2); s=(t.part_has_track==1)&(pt>2)
    phi=t.part_phi0
    dF=t.part_d0-(vx*np.sin(phi)-vy*np.cos(phi))
    dbs=t.part_d0-(mx*np.sin(phi)-my*np.cos(phi))
    hvb=ak.broadcast_arrays(ak.Array(hv),t.part_d0)[0]; farb=ak.broadcast_arrays(ak.Array(far),t.part_d0)[0]
    sv=s&hvb&~farb
    fl=lambda a,m: ak.to_numpy(ak.flatten(a[m]))
    R["dF"].append(fl(dF,sv)); R["d0e"].append(fl(t.part_d0_err,sv)); R["z0v"].append(fl(t.part_z0-vz,sv))
    R["vfar"].append(np.array([far.sum(), hv.sum(), len(hv)])); R["dbs"].append(fl(dbs,s&~hvb))
r={k:np.concatenate(v) if k!="vfar" else np.sum(v,axis=0) for k,v in R.items()}
print("events: vertex far (>1 mm from the median) / with vertex / all:", r["vfar"], f"-> {r['vfar'][0]/r['vfar'][1]:.3f} of vertices")
d,e=r["dF"],r["d0e"]
print(f"good-vertex events, pT>2: d0 wrt V (flipped sign) core {1e3*rsig(d):.0f} um, pull core {rsig(d/e):.2f}")
for lo,hi in [(0,0.03),(0.03,0.06),(0.06,0.15),(0.15,10)]:
    m=(e>=lo)&(e<hi); print(f"  d0_err {lo*1e3:.0f}-{hi*1e3:.0f} um: frac {m.mean():.2f}, d0 core {1e3*rsig(d[m]):.0f} um, pull core {rsig(d[m]/e[m]):.2f}")
print(f"z0 - vz core {1e3*rsig(r['z0v']):.0f} um (sign of z0 is not affected by the d0 convention)")
print(f"events without vertex, d0 wrt the file-median vertex (flipped): core {1e3*rsig(r['dbs']):.0f} um")
