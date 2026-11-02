# -*- coding: utf-8 -*-
"""=====================================================================================
 afpm_femm.py  --  FEMM (pyfemm) model aksijalnog stroja (razmotani 12/10)
 -----------------------------------------------------------------------------------
 Racuna:  (1) COGGING moment  T(d) = Fx(d) * R_av
          (2) ULANCENI TOK po fazi  Psi_A/B/C(d)  ->  EMF = -dPsi/dt u Pythonu
 Izlaz:   afpm_femm_cogging.txt  i  afpm_femm_flux.txt  (za afpm_study.py)
=====================================================================================
"""
import numpy as np
import femm

# NACIN RADA
MODE = "build"     #"build" = nacrtaj jedan polozaj (d=0) i STANI za vizualnu provjeru
                   #"sweep" = pokreni cijeli sweep i izvezi .txt

# Fina mreza u unutarnjem zraku (raspor+utori) -> gladji, tocniji cogging.
# Manja vrijednost = cisci cogging ali SPORIJE (vise elemenata po polozaju).
#   1.0  -> dobar kompromis
#   0.6  -> jos glatkije, osjetno sporije
#   None -> automatska mreza (brzo, ali suminiji cogging kao prije)
GAP_MESH_MM = 1.0

# ===========================ULAZNI PARAMETRI  ===========================
r_core_in, r_core_out = 0.663085884, 0.729990258
r_shoe = 0.9
r_mag  = 0.6944444444
D_in, D_out = 180.0, 275.0
Nm, Ns = 10, 12
mag_th, gap = 5.0, 1.5
rotor_th, shoe_th, core_th = 10.0, 8.0, 40.0
cu_w = 4.0                              # sirina bakrene trake
Br, mu_r, Nc = 1.309, 1.03466875126278, 8

R_av = (D_in/2 + D_out/2)/2            # 113.75 mm
L    = 2*np.pi*R_av                    # opseg (X domena), 714.712 mm
L_r  = (D_out - D_in)/2                # dubina modela (radijalna duljina), 47.5 mm
slot_pitch = L/Ns
pole_pitch = L/Nm
mag_w  = (L/Nm)*r_mag
shoe_w = (L/Ns)*r_shoe
core_w = (L/Ns)*((r_core_in+r_core_out)/2)

mu0 = 4*np.pi*1e-7
Hc  = Br/(mu0*mu_r)  #koercitivnost magneta [A/m]

#PORAVNANJA za provjeru uz Ansys
mag_offset   = 0.0
tooth_offset = slot_pitch/2.0

# --- aksijalni (Y) slojevi, dvostrani ---
Y = {}
Y['rot_b']=(0.0,rotor_th); Y['mag_b']=(rotor_th,rotor_th+mag_th)
Y['gap_b']=(Y['mag_b'][1],Y['mag_b'][1]+gap)
Y['shoe_b']=(Y['gap_b'][1],Y['gap_b'][1]+shoe_th)
Y['core']=(Y['shoe_b'][1],Y['shoe_b'][1]+core_th)
Y['shoe_t']=(Y['core'][1],Y['core'][1]+shoe_th)
Y['gap_t']=(Y['shoe_t'][1],Y['shoe_t'][1]+gap)
Y['mag_t']=(Y['gap_t'][1],Y['gap_t'][1]+mag_th)
Y['rot_t']=(Y['mag_t'][1],Y['mag_t'][1]+rotor_th)
H = Y['rot_t'][1]                       # 89.0 mm
margin = 5.0                            # tanki sloj zraka gore/dolje (A=0 rub)

ROTOR_GRP, STATOR_GRP = 1, 2

# --- namot 12/10 (tooth 1-based -> (faza, predznak)) ---
_wind = {"A":{1:+1,6:+1,7:-1,12:-1}, "B":{5:+1,10:+1,4:-1,11:-1}, "C":{2:+1,9:+1,3:-1,8:-1}}
TOOTH2PH = {}
for ph,d in _wind.items():
    for t,s in d.items():
        TOOTH2PH[t-1] = (ph, s)        # 0-based


def _rect(x0, y01, w, mat, group, magdir=0, circuit='', turns=0):
    """Nacrtaj pravokutnik [x0,x0+w] x [y01] i dodijeli materijal/svojstva."""
    x1 = x0 + w
    y0, y1 = y01
    femm.mi_drawrectangle(x0, y0, x1, y1)
    xc, yc = (x0+x1)/2.0, (y0+y1)/2.0
    femm.mi_addblocklabel(xc, yc)
    femm.mi_selectlabel(xc, yc)
    femm.mi_setblockprop(mat, 1, 0, circuit, magdir, group, turns)
    femm.mi_clearselected()


def _magnet_segments(d):
    """Vrati listu (x0, x1, magdir) magneta za polozaj rotora d, omotano u [0,L]."""
    out = []
    for k in range(Nm):
        cx = (k*pole_pitch + mag_offset + d) % L
        x0, x1 = cx - mag_w/2.0, cx + mag_w/2.0
        magdir = 90 if k % 2 == 0 else 270          # +Y / -Y (isti gore i dolje)
        if x0 < 0:
            segs = [(x0+L, L), (0.0, x1)]
        elif x1 > L:
            segs = [(x0, L), (0.0, x1-L)]
        else:
            segs = [(x0, x1)]
        for a, b in segs:
            if b-a > 1e-4:
                out.append((a, b, magdir))
    return out


def build(d):
    """Izgradi kompletan FEMM model za polozaj rotora d [mm]."""
    femm.newdocument(0)                                     # 0 = magnetics
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, L_r, 30)

    # --- materijali ---
    femm.mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    femm.mi_addmaterial('Steel', 10000, 10000, 0, 0, 0, 0, 0, 1, 0, 0, 0)  # mu_r=1e4 (kao Ansys)
    femm.mi_addmaterial('Copper', 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)
    femm.mi_addmaterial('NdFeB', mu_r, mu_r, Hc, 0, 0, 0, 0, 1, 0, 0, 0)

    # --- strujni krugovi faza (0 A, serijski) ---
    for ph in ('A', 'B', 'C'):
        femm.mi_addcircprop(ph, 0, 1)

    # --- rotorske ploce (pune trake) ---
    _rect(0, Y['rot_b'], L, 'Steel', ROTOR_GRP)
    _rect(0, Y['rot_t'], L, 'Steel', ROTOR_GRP)

    # --- magneti (omotani), isti smjer gore/dolje ---
    for (a, b, md) in _magnet_segments(d):
        _rect(a, Y['mag_b'], b-a, 'NdFeB', ROTOR_GRP, magdir=md)
        _rect(a, Y['mag_t'], b-a, 'NdFeB', ROTOR_GRP, magdir=md)

    # --- stator: 12 zubaca (papuca + jezgra) + coils u utorima ---
    for k in range(Ns):
        cx = k*slot_pitch + tooth_offset
        _rect(cx-shoe_w/2, Y['shoe_b'], shoe_w, 'Steel', STATOR_GRP)
        _rect(cx-shoe_w/2, Y['shoe_t'], shoe_w, 'Steel', STATOR_GRP)
        _rect(cx-core_w/2, Y['core'],   core_w, 'Steel', STATOR_GRP)
        ph, s = TOOTH2PH[k]
        # dvije strane svitka, PRISLONJENE uz jezgru (bez procjepa -> nema izoliranog zraka)
        _rect(cx-core_w/2-cu_w, Y['core'], cu_w, 'Copper', STATOR_GRP, circuit=ph, turns=+s*Nc)
        _rect(cx+core_w/2,      Y['core'], cu_w, 'Copper', STATOR_GRP, circuit=ph, turns=-s*Nc)

    # --- zrak: TOCNO 3 povezana podrucja -> 3 labele (visak labela = greska u pyfemm) ---
    # sav unutarnji zrak (oba raspora + utori + medju-magnetni prostori) je JEDNO
    # povezano podrucje jer je donji raspor neprekinuta traka koja sve spaja.
    femm.mi_drawrectangle(0, -margin, L, H+margin)          # vanjski rub domene
    ygb = (Y['gap_b'][0]+Y['gap_b'][1])/2.0                 # sredina donjeg raspora (spojnica)
    # gornji i donji zrak: gruba (automatska) mreza je sasvim dovoljna
    for (xa, ya) in [(L*0.5, H+margin/2.0), (L*0.5, -margin/2.0)]:
        femm.mi_addblocklabel(xa, ya)
        femm.mi_selectlabel(xa, ya)
        femm.mi_setblockprop('Air', 1, 0, '', 0, 0, 0)
        femm.mi_clearselected()
    # UNUTARNJI zrak (raspor + utori): FINA, jednolika mreza -> gladji cogging
    xi, yi = (tooth_offset + slot_pitch/2.0) % L, ygb
    femm.mi_addblocklabel(xi, yi)
    femm.mi_selectlabel(xi, yi)
    if GAP_MESH_MM and GAP_MESH_MM > 0:
        femm.mi_setblockprop('Air', 0, GAP_MESH_MM, '', 0, 0, 0)   # fiksna velicina elemenata [mm]
    else:
        femm.mi_setblockprop('Air', 1, 0, '', 0, 0, 0)             # automatska
    femm.mi_clearselected()

    # --- rubni uvjeti ---
    # periodicni lijevo/desno (Bdep = +Bind, puni krug)
    femm.mi_addboundprop('Periodic', 0, 0, 0, 0, 0, 0, 0, 0, 4)    # 4 = periodic
    # A=0 gore/dolje (vanjski rub)
    femm.mi_addboundprop('A0', 0, 0, 0, 0, 0, 0, 0, 0, 0)          # 0 = prescribed A

    # dodijeli periodic na lijevi (x=0) i desni (x=L) vertikalni rub
    for xside in (0.0, L):
        femm.mi_selectsegment(xside, (H)/2.0)
        femm.mi_setsegmentprop('Periodic', 0, 1, 0, 0)
        femm.mi_clearselected()
    # dodijeli A0 na gornji (y=H+margin) i donji (y=-margin) rub
    for yside in (-margin, H+margin):
        femm.mi_selectsegment(L/2.0, yside)
        femm.mi_setsegmentprop('A0', 0, 1, 0, 0)
        femm.mi_clearselected()

    femm.mi_zoomnatural()


def solve_and_extract(d):
    """Rijesi model na polozaju d i vrati (T_cog, Psi_A, Psi_B, Psi_C)."""
    femm.mi_saveas('afpm_tmp.fem')
    femm.mi_analyze(1)
    femm.mi_loadsolution()
    # cogging: x-sila na rotor grupu (weighted stress tensor, kod 18)
    femm.mo_groupselectblock(ROTOR_GRP)
    Fx = femm.mo_blockintegral(18)                 # [N] (ukljucuje dubinu L_r)
    femm.mo_clearblock()
    T_cog = Fx * (R_av*1e-3)                        # [N*m]
    # ulanceni tok po fazi iz strujnih krugova
    psi = {}
    for ph in ('A', 'B', 'C'):
        _, _, flux = femm.mo_getcircuitproperties(ph)
        psi[ph] = flux
    return T_cog, psi['A'], psi['B'], psi['C']


# ================================ GLAVNI PROGRAM ================================
if __name__ == "__main__":
    femm.openfemm()       

    if MODE == "build":
        build(0.0)
        femm.mi_saveas('afpm_build_check.fem')
        print("Model izgradjen na d=0 i spremljen kao afpm_build_check.fem.")
        print(f"  opseg L={L:.2f} mm, visina H={H:.1f} mm, dubina L_r={L_r:.1f} mm")
        # ostavi FEMM otvoren da vidis geometriju:
        # femm.closefemm()   # <- zakomentirano namjerno

    elif MODE == "sweep":
        # jedan ELEKTRICNI period = L/p (142.9 mm) = 12 cogging perioda.
        # -> cijeli val toka (EMF) I 12 cogging perioda (za glatko usrednjavanje).
        p = Nm//2
        elec_period = L/p
        n_steps = 72                                     # 6 tocaka po cogging periodu
        d_vals = np.linspace(0.0, elec_period, n_steps, endpoint=False)
        rows_cog, rows_flux = [], []
        for i, d in enumerate(d_vals):
            build(d)
            T, pa, pb, pc = solve_and_extract(d)
            rows_cog.append((d, T))
            rows_flux.append((d, pa, pb, pc))
            print(f"  [{i+1:2d}/{n_steps}] d={d:6.3f} mm  T_cog={T:+.4f} N*m  "
                  f"Psi=({pa:+.3e},{pb:+.3e},{pc:+.3e}) Wb")

        np.savetxt('afpm_femm_cogging.txt', np.array(rows_cog),
                   header='pomak_mm   T_cog_Nm', comments='')
        np.savetxt('afpm_femm_flux.txt', np.array(rows_flux),
                   header='pomak_mm   Psi_A   Psi_B   Psi_C', comments='')
        print("\nSpremljeno: afpm_femm_cogging.txt, afpm_femm_flux.txt")
        femm.closefemm()
