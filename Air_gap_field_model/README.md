# Air gap field model

Analitički model magnetskog polja u zračnom rasporu aksijalnog (YASA) elektromotora s permanentnim magnetima: polje magneta iz Fourierova reda, a utjecaj statorskih utora preko kompleksne relativne provodljivosti λ dobivene konformnim preslikavanjem. Rezultati se uspoređuju s numeričkim (Ansys Maxwell) rješenjem.

Završni rad: *Elektromagnetsko polje u aksijalnom elektromotoru: teorijski model i simulacija*, PMF Split, 2025. ([Zavrsni_rad_Karla_Belak.pdf](Zavrsni_rad_Karla_Belak.pdf))

| Datoteka | Opis |
|---|---|
| `pmsm_model.py` | Glavni kod (Dodatak A): λ(z), polje magneta, polje s utorima i usporedba s Ansysom. Generira slike 15–20 iz rada. |
| `num_rez.py` | Prikaz samih numeričkih (Ansys) rješenja, slike 11–14 iz rada. |
| `By.txt`, `Bx.txt` | Ansys Maxwell: B_y i B_x po sredini raspora, model bez utora. |
| `By_utori.txt`, `Bx_utori.txt` | Ansys Maxwell: B_y i B_x po sredini raspora, model s utorima. |

## Pokretanje

```bash
pip install -r requirements.txt
python pmsm_model.py
python num_rez.py
```

Skripte se pokreću iz ove mape jer `.txt` datoteke čitaju relativnom putanjom.
