import glob, sys, numpy as np, awkward as ak
def rsig(x):
    q75,q25=np.percentile(x[np.isfinite(x)],[75,25]); return (q75-q25)/1.349
for name, pat, B in [("ALEPH","/local/laurits/ALEPH_ntuples/events/data_ALEPH1994_000[0-4]0_1.parquet",1.5),("CLD","/local/laurits/CLD_ntuples/events/data_Zqq_00010_1.parquet",2.0)]:
    res={k:[] for k in ["lcio","flip","raw","dxy","lifeL","lifeF"]}
    for f in sorted(glob.glob(pat)):
        t=ak.from_parquet(f,columns=["event_has_vertex","event_vertex_x","event_vertex_y","part_px","part_py","part_pz","part_energy","part_has_track","part_d0","part_phi0","part_dxy"])
        hv=ak.to_numpy(t.event_has_vertex).astype(bool); t=t[hv]
        vx=ak.Array(ak.to_numpy(t.event_vertex_x)); vy=ak.Array(ak.to_numpy(t.event_vertex_y))
        pt=np.sqrt(t.part_px**2+t.part_py**2); s=(t.part_has_track==1)&(pt>2)
        phi=t.part_phi0; d0=t.part_d0
        # LCIO: PCA = ref + (-d0 sin phi, d0 cos phi) -> d0(wrt V) = d0 - (-vx sin + vy cos)
        dL=d0-(-vx*np.sin(phi)+vy*np.cos(phi)); dF=d0-(vx*np.sin(phi)-vy*np.cos(phi))
        res["lcio"].append(ak.to_numpy(ak.flatten(dL[s]))); res["flip"].append(ak.to_numpy(ak.flatten(dF[s])))
        res["raw"].append(ak.to_numpy(ak.flatten(d0[s]))); res["dxy"].append(ak.to_numpy(ak.flatten(t.part_dxy[s])))
        # lifetime sign w.r.t. the hemisphere axis in the transverse plane, PCA direction for each convention
        lead=ak.argmax(t.part_energy,axis=1,keepdims=True)
        nx=ak.firsts(t.part_px[lead]); ny=ak.firsts(t.part_py[lead]); nz=ak.firsts(t.part_pz[lead])
        side=(t.part_px*nx+t.part_py*ny+t.part_pz*nz)>0
        hx=ak.where(side,ak.sum(t.part_px*side,axis=1),ak.sum(t.part_px*~side,axis=1))
        hy=ak.where(side,ak.sum(t.part_py*side,axis=1),ak.sum(t.part_py*~side,axis=1))
        # PCA - V direction: LCIO (-sin, cos)*dL ; flipped (sin, -cos)*dF
        sL=np.sign(dL*(-np.sin(phi)*hx+np.cos(phi)*hy)); sF=np.sign(dF*(np.sin(phi)*hx-np.cos(phi)*hy))
        res["lifeL"].append(ak.to_numpy(ak.flatten((np.abs(dL)*sL)[s]))); res["lifeF"].append(ak.to_numpy(ak.flatten((np.abs(dF)*sF)[s])))
    r={k:np.concatenate(v) for k,v in res.items()}
    print(f"== {name}: tracks pT>2 GeV in events with vertex: {len(r['raw'])}")
    print(f"  robust sigma [um]: d0 raw (wrt ref point) {1e3*rsig(r['raw']):.0f} | d0 wrt V, LCIO sign {1e3*rsig(r['lcio']):.0f} | d0 wrt V, flipped sign {1e3*rsig(r['flip']):.0f} | stored dxy {1e3*rsig(r['dxy']):.0f}")
    for k in ["lifeL","lifeF"]:
        x=r[k]; p=np.mean(x>0.2); m=np.mean(x<-0.2)
        print(f"  {k}: frac >+200um {p:.4f}, <-200um {m:.4f}, asym {(p-m)/(p+m):+.3f}; median signed {1e3*np.median(x):+.1f} um")
