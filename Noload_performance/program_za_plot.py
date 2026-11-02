# -*- coding: utf-8 -*-
import os
os.environ["MPLBACKEND"] = "agg"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Ulazni podatci (rezultati FEMM analize)
COG_FILE  = "afpm_femm_cogging.txt"
FLUX_FILE = "afpm_femm_flux.txt"
R_av_mm   = 113.75
n_rpm     = 2500.0 #Ovo je proizvoljan podatak (bitno da je isto definiran i za analltički model)

opseg = 2*np.pi*R_av_mm
elec  = opseg/5.0  #Jedan elektricni period u mm (p=5 broj parova polova magneta)
v_lin = (2*np.pi*n_rpm/60.0)*(R_av_mm*1e-3)

def rms(a): return float(np.sqrt(np.mean(np.asarray(a)**2)))
def smooth(y,k=1): #Dio uz Claude (smoothing 'raw' podataka iz femm)
    yy=np.r_[y[-k:],y,y[:k]]; s=np.convolve(yy,np.ones(2*k+1)/(2*k+1),'same'); return s[k:-k]
def best_shift(xr,yr,xs,ys,P,n=400):
    g=np.linspace(0,P,n,endpoint=False); yrr=np.interp(g%P,xr%P,yr,period=P); b=(1e18,0.0)
    for s in np.linspace(0,P,n,endpoint=False):
        e=np.mean((yrr-np.interp((g+s)%P,xs%P,ys,period=P))**2)
        if e<b[0]: b=(e,s)
    return b[1]
def metrics(da,ya,d,yf,sh):
    g=np.linspace(0,elec,400,endpoint=False)
    a=np.interp(g%elec,da%elec,ya,period=elec)
    f=np.interp((g+sh)%elec,d%elec,yf,period=elec)
    return np.corrcoef(a,f)[0,1], 100*rms(a-f)/rms(a)

#Iščitavanje podataka iz .txt FEMM fileova
cog=np.loadtxt(COG_FILE,skiprows=1); flx=np.loadtxt(FLUX_FILE,skiprows=1)
d=cog[:,0]; Tf=cog[:,1]
Psi={'A':flx[:,1],'B':flx[:,2],'C':flx[:,3]}
EMFf={ph:-np.gradient(Psi[ph],d*1e-3)*v_lin for ph in 'ABC'}

#Unošenje analitičkog rješenja
import afpm_finalni as P
da=P.theta*P.radijus_rotora; Ta=P.T_cog; EMFa=P.EMF

#Zajednicki fazni pomak (isti d=0 offset za sve); racunamo iz faze A
sh=best_shift(da,EMFa['A'],d,EMFf['A'],elec)

#Plot
def fig_phase(ph):
    r,e=metrics(da,EMFa[ph],d,EMFf[ph],sh)
    plt.figure(figsize=(8,4.6))
    plt.plot(da,EMFa[ph],lw=2.2,label='Analitičko rješenje')
    plt.plot((d+sh)%elec,EMFf[ph],'o',ms=4,alpha=.85,label='FEMM')
    plt.xlabel('pomak [mm]'); plt.ylabel('EMF [V]')
    plt.title(f'Inducirani napon: faza {ph}\n'
              f'RMS: FEMM {rms(EMFf[ph]):.1f} V / Analit {rms(EMFa[ph]):.1f} V;  '
              )
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    fn=f'emf_faza_{ph}.png'; plt.savefig(fn,dpi=130); plt.close()

for ph in 'ABC': fig_phase(ph)

#Sve tri faze zajedno (analitika linije + FEMM tocke)
plt.figure(figsize=(9,4.6))
for ph,c in zip('ABC',['C0','C1','C2']):
    plt.plot(da,EMFa[ph],c,lw=2.0,label=f'Analit {ph}')
    plt.plot((d+sh)%elec,EMFf[ph],c,marker='o',ls='',ms=3,alpha=.7,label=f'FEMM {ph}')
plt.xlabel('pomak [mm]'); plt.ylabel('EMF [V]')
plt.title('Trofazni inducirani napon: analitičko rješenje i FEMM numeričko')
plt.legend(ncol=3,fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('emf_tri_faze.png',dpi=130); plt.close()

#Cogging (pulsirajući moment)
plt.figure(figsize=(9,4.6))
plt.plot(da,Ta,lw=2.2,label=f'Analitičko rješenje (amp {(Ta.max()-Ta.min())/2:.2f} N·m)')
plt.plot(d,Tf,'.',ms=4,color='C1',alpha=.35,label='FEMM (izvorni rezultati)')
plt.plot(d,smooth(Tf,1),'-',color='C1',lw=1.3,label='FEMM (zaglađeno)')
plt.xlabel('pomak [mm]'); plt.ylabel(r'$T_{cog}$ [N·m]')
plt.title('Pulsirajući moment sile; analitičko rješenje i FEMM numeričko')
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('cogging.png',dpi=130); plt.close()

print(np.average(Tf), np.average(Ta))
