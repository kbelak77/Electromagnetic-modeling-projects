import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#rotor ima 10 polova magneta
#stator ima 10 jezgri

#ulazni parametri
#središnji dio statorske jezgre ima omjer kutnog raspona (u odnosu na korak utora) na unutarnjem radijusu
omjer_kuta_jezgre_unutarnji = 0.663085884
#-II- u odnosu na korak utora na vanjskom radijusu
omjer_kuta_jezgre_vanjski = 0.729990258
#omjer širine papuče zuba statora i koraka utora na vanjskom radijusu.
omjer_kuta_papuce_unutarnji = 0.9
#-II- na unutarnjem radijusu
omjer_kuta_papuce_vanjski = 0.9
#omjer luka pola i koraka pola na vanjskom radijusu magneta
omjer_magneta_vanjski = 0.6944444444
#-II- na unutarnjem radijusu magneta
omjer_magneta_unutarnji = 0.6944444444

promjer_magneta_unutarnji = 180.0
promjer_magneta_vanjski = 275.0

Nm = 10  #Broj magneta
Ns = 12  #Broj jezgri statora

debljina_magneta = 5.0 
air_gap = 1.5  #Dimenzija raspora

g_crtica = debljina_magneta + air_gap #udaljenost ploče rotora i statora
#prosječan raadijus rotora na kojem se vrši 2D simulacija- sljedeće varijable se ref na njega
radijus_rotora = (promjer_magneta_unutarnji / 2 + promjer_magneta_vanjski / 2) / 2
opseg_rotora = 2 * np.pi * radijus_rotora
sirina_magneta = (opseg_rotora / Nm) * (omjer_magneta_unutarnji + omjer_magneta_vanjski) / 2
sirina_papuce = (opseg_rotora / Ns) * (omjer_kuta_papuce_vanjski + omjer_kuta_papuce_unutarnji) / 2
b_0_crtica = opseg_rotora / Ns - sirina_papuce #otvor utora na prosječnom radijusu

print(opseg_rotora)
#PRVI DIO-----#
#simulacija učinka postojanja utora statora konformnim transformacijama
#Iteracija po kompleksnoj Z-ravnini u svrhu pronalaska svih točaka w koje kad se preslikaju u 
#W-ravninu konformnom transformacijom u funkciji calculate_z(w) spadaju unutar područja interesa.
#imaginarni dio z je blizu 0,75 mm (tj. središte zračnog raspora)
#realni dio z leži unutar unaprijed definiranog raspona (što odgovara fizičkim položajima duž zračnog raspora)
#svaka w koja zadovoljava ove kriterije pohranjuje se, zajedno s odgovarajućim 𝑧 = z(w) i 𝜆(𝑤) 
#(magnetskim skalarnim potencijalom).

#definirani koeficijenti iz rada [249]
b = (b_0_crtica / (2 * g_crtica) + np.sqrt((b_0_crtica / (2 * g_crtica)) ** 2 + 1)) ** 2
a = 1 / b

#odabir početka izračuna od središta raspora- računamo u centru tok
target_imag_z = air_gap / 2
#npr vrijednost 0, za taj raspon real_z ne pronalazi točke u otvoru raspora- postavljen početak u centru otvora utora
#poslije mirroran preko osi -b_0_crtica / 2 i dupliciran 12 puta (12 jezgri)
target_real_z_min = -b_0_crtica / 2
#varijabilan- potreban radi definiranja adekvatnog raspona ovisno o rezultatima;
#->ustanovljeno da se magnetski tok stabilizira na udaljenosti znatno kraćoj od širine polne papučice.
target_real_z_max = sirina_papuce / 2

def calculate_z(w: np.ndarray) -> np.ndarray:
    #z(w)
    term = np.sqrt((w - b) / (w - a))
    return g_crtica / np.pi * (
        np.log((1 + term) / (1 - term))
        - np.log((b + term) / (b - term))
        - 2 * (b - 1) / np.sqrt(b) * np.arctan(term / np.sqrt(b))
    )

def calculate_lambda(w: np.ndarray) -> np.ndarray:
    return (w - 1) / (np.sqrt(w - a) * np.sqrt(w - b))

#definiramo raspon traženja w- uvjetovan dobivenim grafovima ovisnosti realnih i imaginarnih 
#dijelova w za koje je zadovoljen uvjet target_imag_z = 0.75 ->moguće iz grafova odrediti
#smjer protezanja točaka w koje zadovoljavaju
#realni dio w [0,inf> imag [-2, inf> međutim protezanje do konačne granice (veće) daje dobre rez

#mreža pretraživanja re,im dijelova w
real_min, real_max = 0.0, 1200.0
imag_min, imag_max = -2.0, 600.0

#def više koraka radi optimizacije simulacije
small_step1 = 0.005
small_step2 = 0.05
medium_step = 1.0
base_large_step = 6.0

step = medium_step / 5.0

found_w = []
found_z = []
lambda_values = []


#manji korak u osjetljivijem području (u okolini otvora utora)
real_part = real_min
while real_part <= real_max:
    if real_part < 3:
        step_size = small_step1
    elif real_part < 15:
        step_size = small_step2
    elif real_part <= 50:
        step_size = medium_step
    else:
        step_size = base_large_step + 6.0 * np.floor((real_part - 50.0) / 50.0)
    #stvaranje mreže za re, im dijelove
    real_grid, imag_grid = np.meshgrid(
        np.arange(real_part, real_part + step_size + 1e-12, step_size),
        np.arange(imag_min, imag_max + 1e-12, step_size),
        indexing="xy",
    )
    w_grid = real_grid + 1j * imag_grid
    z_grid = calculate_z(w_grid)

    #mask u rasponu za realni odnosno imaginarni dio
    #provjera uvjeta i pohrana zadovoljavajućih w,z, lambda
    z_imag_close = np.abs(np.imag(z_grid) - target_imag_z) < 1e-2
    z_real_in = (np.real(z_grid) >= target_real_z_min) & (np.real(z_grid) <= target_real_z_max)
    z_mask = z_imag_close & z_real_in
    #pohrana u polja
    if np.any(z_mask):
        vw = w_grid[z_mask]
        vz = z_grid[z_mask]
        vl = calculate_lambda(vw)
        found_w.append(vw.ravel())
        found_z.append(vz.ravel())
        lambda_values.append(vl.ravel())

    real_part += step

if len(found_w) == 0:
    raise RuntimeError("Nisu pronađene zadovoljavajuće točke w. Potrebno je promijeniti mrežu/korake.")

found_w = np.concatenate(found_w).astype(np.complex128)
found_z = np.concatenate(found_z).astype(np.complex128)
lambda_values = np.concatenate(lambda_values).astype(np.complex128)

#filtriranje vrijednosti lambda
real_lambda = np.real(lambda_values)
imag_lambda = np.imag(lambda_values)
valid_indices = real_lambda > 0.45

filtered_real_lambda = real_lambda[valid_indices]
filtered_imag_lambda = imag_lambda[valid_indices]
filtered_real_z = np.real(found_z[valid_indices])

#rezultati samo za jednu stranu otvora utora- koristi se periodična simetrija motora
#sveukupno polje se računa zrcaljenjem rezultata duž osi simetrije

#podatci se zrcale počevši od središta otvora utora- podudarnost s 2D modelom
mirror_axis_real_z = -b_0_crtica / 2
mirrored_real_z = 2 * mirror_axis_real_z - filtered_real_z

#pohrana ponavljajućih vrijednosti duž oboda
all_real_z = []
all_lambda = []
all_mirrored_real_z = []
all_mirrored_lambda = []
all_added_real_z = []
all_added_lambda = []
all_imag_lambda = []
all_mirrored_imag_lambda = []

#kopiranje i translatacija zrcaljenih i originalnih funkcija duž cijelog oboda rotora
num_repeats = Ns #jednak broju jezgri statora
spacing = opseg_rotora / num_repeats #razmak ponavljanja
offset_shift = b_0_crtica / 2 #početak

#pomak grafa ulijevo da bi se pokrio cijeli opseg rotora-računa se kao da ima 13 jezgri
#ali se uzima raspon 12 uzimajući interval od 0 do opsega rotora
for repeat_idx in range(0, num_repeats + 2):
    #pomak za svaku iteraciju
    offset = repeat_idx * spacing + offset_shift

    #original
    idx_orig = (filtered_real_z + offset >= 0) & (filtered_real_z + offset <= opseg_rotora)
    if np.any(idx_orig):
        all_real_z.append(filtered_real_z[idx_orig] + offset)
        all_lambda.append(filtered_real_lambda[idx_orig])
        all_imag_lambda.append(filtered_imag_lambda[idx_orig])

    #zrcaljene
    idx_m = (mirrored_real_z + offset >= 0) & (mirrored_real_z + offset <= opseg_rotora)
    if np.any(idx_m):
        all_mirrored_real_z.append(mirrored_real_z[idx_m] + offset)
        all_mirrored_lambda.append(filtered_real_lambda[idx_m])
        all_mirrored_imag_lambda.append(-filtered_imag_lambda[idx_m])

    #dodane točke u području konst permeabilnosti zbog boljeg prikaza (lambda real=1, imag=0)
    x1_vals = np.arange(offset_shift + sirina_papuce / 20 * 7, (sirina_papuce / 20 * 13 + offset_shift) + 1e-12, sirina_papuce / 20)
    x1_vals = x1_vals + repeat_idx * spacing
    mask = (x1_vals >= 0) & (x1_vals <= opseg_rotora)
    if np.any(mask):
        all_added_real_z.append(x1_vals[mask])
        all_added_lambda.append(np.ones(np.count_nonzero(mask)))

#spajanje lista
def _concat(lst):
    return np.concatenate(lst) if len(lst) > 0 else np.array([], dtype=float)

all_real_z = _concat(all_real_z)
all_lambda = _concat(all_lambda)
all_mirrored_real_z = _concat(all_mirrored_real_z)
all_mirrored_lambda = _concat(all_mirrored_lambda)
all_added_real_z = _concat(all_added_real_z)
all_added_lambda = _concat(all_added_lambda)
all_imag_lambda = _concat(all_imag_lambda)
all_mirrored_imag_lambda = _concat(all_mirrored_imag_lambda)

#za interpolaciju kroz točke (real(lambda) vs real(z))
combined_real_z = np.concatenate([all_real_z, all_mirrored_real_z, all_added_real_z])
combined_lambda = np.concatenate([all_lambda, all_mirrored_lambda, all_added_lambda])

sort_idx = np.argsort(combined_real_z, kind="mergesort")
sorted_real_z = combined_real_z[sort_idx]
sorted_lambda = combined_lambda[sort_idx]

#uklanjanje duplikata
unique_vals, unique_idx = np.unique(sorted_real_z, return_index=True)
unique_real_z = unique_vals
unique_lambda = sorted_lambda[unique_idx]

#parametri interpolacije
num_points = 10000
interpolated_real_z = np.linspace(np.min(unique_real_z), np.max(unique_real_z), num_points)

#linearna interpolacija
interp_fun = interp1d(unique_real_z, unique_lambda, kind="linear", bounds_error=False, fill_value=np.nan)
interpolated_lambda = interp_fun(interpolated_real_z)

if np.isnan(interpolated_lambda).any():
    not_nan = ~np.isnan(interpolated_lambda)
    if np.any(not_nan):
        first_valid = np.argmax(not_nan)
        last_valid = len(interpolated_lambda) - 1 - np.argmax(not_nan[::-1])
        interpolated_lambda[:first_valid] = interpolated_lambda[first_valid]
        interpolated_lambda[last_valid+1:] = interpolated_lambda[last_valid]
        nans = np.isnan(interpolated_lambda)
        interpolated_lambda[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), interpolated_lambda[~nans])

#za imag(lambda)
combined_real_z_imag = np.concatenate([all_real_z, all_mirrored_real_z])
combined_imag_lambda = np.concatenate([all_imag_lambda, all_mirrored_imag_lambda])

sort_idx_im = np.argsort(combined_real_z_imag, kind="mergesort")
sorted_real_z_imag = combined_real_z_imag[sort_idx_im]
sorted_imag_lambda = combined_imag_lambda[sort_idx_im]

unique_vals_im, unique_idx_im = np.unique(sorted_real_z_imag, return_index=True)
unique_real_z_imag = unique_vals_im
unique_imag_lambda = sorted_imag_lambda[unique_idx_im]

interpolated_imag_real_z = np.linspace(np.min(unique_real_z_imag), np.max(unique_real_z_imag), num_points)
interp_fun_im = interp1d(unique_real_z_imag, unique_imag_lambda, kind="linear", bounds_error=False, fill_value=np.nan)
interpolated_imag_lambda = interp_fun_im(interpolated_imag_real_z)

if np.isnan(interpolated_imag_lambda).any():
    not_nan = ~np.isnan(interpolated_imag_lambda)
    if np.any(not_nan):
        first_valid = np.argmax(not_nan)
        last_valid = len(interpolated_imag_lambda) - 1 - np.argmax(not_nan[::-1])
        interpolated_imag_lambda[:first_valid] = interpolated_imag_lambda[first_valid]
        interpolated_imag_lambda[last_valid+1:] = interpolated_imag_lambda[last_valid]
        nans = np.isnan(interpolated_imag_lambda)
        interpolated_imag_lambda[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), interpolated_imag_lambda[~nans])

#print(x_3[-1])
#print(x_4[-1])
#print(len(new_x))
#print(new_x[len(x_3)])
#2764
#1843
#1957
i=0
for element in interpolated_real_z:
    i+=1
    if element>=140:
        print(element)
        print(i)
        break



#grafovi prvog dijela
plt.figure()
plt.plot(all_real_z[:2700], all_lambda[:2700], 'o', label='Izvorne točke')
plt.plot(all_mirrored_real_z[:1800], all_mirrored_lambda[:1800], 'o', label='Zrcaljene točke')
plt.plot(interpolated_real_z[:1957], interpolated_lambda[:1957], '-', linewidth=2, label='Interpolirana krivulja')
plt.xlabel(r'$\Re\{z\}$')
plt.ylabel(r'$\Re\{\lambda\}$')
plt.legend()
plt.grid(True)
plt.savefig("permanenc_re.pdf", bbox_inches="tight") 

plt.figure()
plt.plot(all_real_z[:2700], all_imag_lambda[:2700], 'o', label='Izvorne točke')
plt.plot(all_mirrored_real_z[:1800], all_mirrored_imag_lambda[:1800], 'o', label='Zrcaljene točke')
plt.plot(interpolated_imag_real_z[:1957], interpolated_imag_lambda[:1957], '-', linewidth=2, label='Interpolirana krivulja')
plt.xlabel(r'$\Re\{z\}$')
plt.ylabel(r'$\Im\{\lambda\}$')
plt.legend()
plt.grid(True)
plt.savefig("permanenc_im.pdf", bbox_inches="tight") 

#DRUGI DIO-----#
#modeliranje doprinosa toku od strane pernamentnih magneta pomoću Maxwellovih jednadžbi

Br = 1.309  #remanencija
ni_0 = 4 * np.pi * 1e-7  #permeabilnost vakuuma
alfa_pm = sirina_magneta / (opseg_rotora / Nm)  #omjer širine magneta i koraka pola
ni_r = 1.03466875126278  #relativna permeabilnost magneta

Y_m = debljina_magneta
g = air_gap
Y_2 = Y_m + g
tau_m = opseg_rotora / Nm #korak pola magneta(udaljenost od jednog pola do susjednog suprotnog)

n_max = 1000 #broj iteracija
n = np.arange(1, n_max + 1, 2)  #definirana magnetizacija- neparan n
U_n = 2 * np.pi * n / (2 * tau_m) #prostorna frekvencija za svaku harmonijsku komponentu
#računa Fourierove koeficijente za magnetizaciju
M_n = -4 * Br / (n * np.pi * ni_0) * np.sin(n * np.pi * alfa_pm / 2)

delta = ni_r * np.cosh(U_n * Y_m) * np.sinh(U_n * (Y_2 - Y_m)) + np.cosh(U_n * (Y_2 - Y_m)) * np.sinh(U_n * Y_m)

#stvara prostorni vektor koji pokriva 12 koraka među polovima s rezolucijom od 0,1
x = np.arange(0, 12 * tau_m + 0.1, 0.1)

#REGION I
umnozak_y = (M_n[:, None] * np.sinh(U_n[:, None] * Y_m) / delta[:, None]
             * np.cosh(U_n[:, None] * (Y_2 - (Y_m + Y_2) / 2)) * np.cos(U_n[:, None] * x[None, :])) #da se odnosi na središte raspora
suma_y = np.sum(umnozak_y, axis=0)

B_y1 = ni_0 * suma_y

umnozak_x = (M_n[:, None] * np.sinh(U_n[:, None] * Y_m) / delta[:, None]
             * np.sinh(U_n[:, None] * (Y_2 - (Y_m + Y_2) / 2)) * np.sin(U_n[:, None] * x[None, :]))
suma_x = np.sum(umnozak_x, axis=0)
B_x1 = ni_0 * suma_x


#izvlačenje rezultata unutar raspona [0, opseg_rotora]
valid_indices = (x >= 0) & (x <= opseg_rotora)
x_valid = x[valid_indices]
B_y1_valid = B_y1[valid_indices]
B_x1_valid = B_x1[valid_indices]

#isti broj točaka interpolacije: 10000
new_x = np.linspace(np.min(x_valid), np.max(x_valid), num_points) #novi xevi za interpolaciju
#
#interpolacija koristeći cubic/spline metodu
interp_By = interp1d(x_valid, B_y1_valid, kind="cubic")
B_y1_interp = interp_By(new_x)

interp_Bx = interp1d(x_valid, B_x1_valid, kind="cubic")
B_x1_interp = -interp_Bx(new_x)  #suprotan preznak- osiguralo da komponenta magnetskog polja odgovara 
#referentnom smjeru korištenom u Maxwellovom simulacijskom modelu.

###############ANSYS DIO---BEZ UTORA
data_1 = np.loadtxt("By.txt", skiprows=7)
data_2 = np.loadtxt("Bx.txt", skiprows=7)

x_1 = data_1[:, 0] 
By1 = data_1[:, 1]

x_2 = data_2[:, 0] 
Bx2 = data_2[:, 1]
Bx2 = np.array(Bx2)*2

plt.figure()
plt.plot(new_x, B_y1_interp, '-', linewidth=2, label=r'$B_y$ analitičko rješenje')
plt.plot(x_1, By1, label=r"$B_y$ numeričko rješenje")
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{y}$')
plt.legend(loc = 'upper right')
plt.grid(True)
plt.savefig("By_bezutori.pdf", bbox_inches="tight") 

plt.figure()
plt.plot(new_x, B_x1_interp, '-', linewidth=2, label=r'$B_x$ analitičko rješenje')
plt.plot(x_2, Bx2, label=r"$B_x$ numeričko rješenje")
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{x}$')
plt.legend()
plt.grid(True)
plt.savefig("Bx_bezutori.pdf", bbox_inches="tight") 

#TREĆI DIO-----#
#uračunavanje efekta utora statora s modelom za doprinos PM
B_krajnji_y = -B_y1_interp * interpolated_lambda + B_x1_interp * interpolated_imag_lambda
B_krajnji_x = B_x1_interp * interpolated_lambda - B_y1_interp * interpolated_imag_lambda


###############ANSYS DIO---
data_3 = np.loadtxt("By_utori.txt", skiprows=7)
data_4 = np.loadtxt("Bx_utori.txt", skiprows=7)

x_3 = data_3[:, 0] 
By3 = data_3[:, 1]

x_4 = data_4[:, 0] 
Bx4 = data_4[:, 1]
#print(x_3[-1])
#print(x_4[-1])
#print(len(new_x))
#print(new_x[len(x_3)])
i=0
for element in new_x:
    i+=1
    if element>=356:
        #print(element)
        #print(i)
        break
plt.figure()
plt.plot(new_x[:4982], B_krajnji_y[:4982], '-', label = r'$B_y$ analitičko rješenje')
plt.plot(x_3, By3, label=r"$B_y$ numeričko rješenje")
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{\mathrm{y}}$')
plt.legend()
plt.grid(True)
plt.savefig("By_utori.pdf", bbox_inches="tight") 

plt.figure()
plt.plot(new_x[:4982], B_krajnji_x[:4982], '-', label = r'$B_x$ analitičko rješenje')
plt.plot(x_4, Bx4, label=r"$B_x$ numeričko rješenje")
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{\mathrm{x}}$')
plt.legend(loc = 'upper right')
plt.grid(True)
plt.savefig("Bx_utori.pdf", bbox_inches="tight") 


if __name__ == "__main__":
    plt.show()
