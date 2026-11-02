# No-load performance

Proširenje analitičkog modela iz [Air_gap_field_model](../Air_gap_field_model) relativnim gibanjem rotora. Iz polja u rasporu računaju se inducirani napon praznog hoda (back EMF) i pulsirajući moment sile (cogging torque) YASA motora 12/10, a rezultati se provjeravaju 2D FEM simulacijom u programu FEMM.

Istraživački rad: *Od magnetskog polja do performansi aksijalnog elektromotora u praznom hodu: inducirani napon i pulsirajući moment sile*, PMF Split, 2026. ([Istrazivacki_rad_Karla_Belak.pdf](Istrazivacki_rad_Karla_Belak.pdf))

| Datoteka | Opis |
|---|---|
| `afpm_finalni.py` | Glavni kod (Dodatak A.1): analitički model, sweep po kutu rotora, cogging moment (Maxwellov tenzor) i fazni EMF. |
| `program_za_plot.py` | Usporedba s FEMM-om (Dodatak A.2), slike 7–11 iz rada. Importa `afpm_finalni.py`. |
| `afpm_femm.py` | Izrada i rješavanje FEMM modela preko pyfemm-a (Dodatak A.3). Skripta je izrađena uz pomoć Claudea (Anthropic) i provjerena od autorice. |
| `afpm_femm_cogging.txt` | FEMM rezultati: cogging moment za 72 položaja rotora. |
| `afpm_femm_flux.txt` | FEMM rezultati: ulančani tok faza A, B, C za 72 položaja rotora. |

## Pokretanje

```bash
pip install numpy scipy matplotlib
python program_za_plot.py      # analitički model + usporedba s priloženim FEMM rezultatima
```

FEMM simulacija nije potrebna za reprodukciju grafova jer su njezini rezultati priloženi u `.txt` datotekama. Za ponovno pokretanje potrebni su Windows, [FEMM 4.2](https://www.femm.info) i `pip install pyfemm`. U `afpm_femm.py` postavi `MODE = "build"` za provjeru geometrije ili `MODE = "sweep"` za puni izračun (72 položaja).

Analitički rezultati (n = 2500 o/min): EMF RMS 64.74 V, vršno 90.44 V; cogging moment peak-to-peak 3.03 N·m.
