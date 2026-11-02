# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Ulazni parametri modela (nalaze se u tablicama u radu)
#Omjeri kutnih raspona (unutarnji /vanjski radijus); ovdje se koriste usrednjeni
omjer_kuta_jezgre_unutarnji = 0.663085884
omjer_kuta_jezgre_vanjski   = 0.729990258
omjer_kuta_papuce_unutarnji = 0.9
omjer_kuta_papuce_vanjski   = 0.9
omjer_magneta_vanjski       = 0.6944444444
omjer_magneta_unutarnji     = 0.6944444444

promjer_magneta_unutarnji = 180.0   #mm
promjer_magneta_vanjski   = 275.0   #mm

Nm = 10   #Broj magneta (polova), (p = Nm/2 parova)
Ns = 12   #Broj jezgri statora (zubaca/utora)

debljina_magneta = 5.0   #mm  (Y_m)
air_gap          = 1.5   #mm  (g)

#Izvedeni geometrijski parametri (tablica 2)
g_crtica       = debljina_magneta + air_gap   #Y2 = Ym + g
radijus_rotora = (promjer_magneta_unutarnji/2 + promjer_magneta_vanjski/2) / 2   #R_av, mm
opseg_rotora   = 2*np.pi*radijus_rotora  # C, mm
sirina_magneta = (opseg_rotora/Nm)*(omjer_magneta_unutarnji + omjer_magneta_vanjski)/2
sirina_papuce  = (opseg_rotora/Ns)*(omjer_kuta_papuce_vanjski + omjer_kuta_papuce_unutarnji)/2
b_0_crtica     = opseg_rotora/Ns - sirina_papuce  #Otvor utora, mm

#Radijalna (trenutna) duljina motora
R_i_mm = promjer_magneta_unutarnji/2.0
R_o_mm = promjer_magneta_vanjski/2.0
L_r_mm = R_o_mm - R_i_mm  #Trenutna radijalna duljina (kao t_cp za n_s = 1)
 
#Električni parametri (tablice)
p = Nm // 2          #Broj pari polova
N_c = 8                #Broj zavoja po zavojnici 
n_rpm    = 2500.0           #Slobodno odabrana brzina vrtnje (nije toliko bitno ne utječe na poklapanje>
#Back-EMF skalira LINEARNO s n_rpm; cogging ne ovisi o brzini

n_airgaps = 2   #Prati YASA topologiju: dva zracna (poravnata) raspora
omega_mech = 2*np.pi*n_rpm/60.0  #Mehanicka kutna brzina, rad/s
f_elec     = p*n_rpm/60.0 #Elektricna frekvencija, Hz

mu0 = 4*np.pi*1e-7 

#NumPy 1.x/2.x kompatibilnost (trapz -> trapezoid), Bira specif način integracije za moment sile
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

#Prvi dio; KOMPLEKSNA RELATIVNA PROVODLJIVOST; lambda(x); stator-fiksan (Isto kao završni)
b = (b_0_crtica/(2*g_crtica) + np.sqrt((b_0_crtica/(2*g_crtica))**2 + 1))**2
a = 1.0/b

target_imag_z    = air_gap/2.0  #Računa se po sredini zračnog raspora
target_real_z_min = -b_0_crtica/2.0
target_real_z_max = sirina_papuce/2.0

def calculate_z(w):
    term = np.sqrt((w - b)/(w - a))
    return g_crtica/np.pi*(
        np.log((1 + term)/(1 - term))
        - np.log((b + term)/(b - term))
        - 2*(b - 1)/np.sqrt(b)*np.arctan(term/np.sqrt(b))
    )

def calculate_lambda(w):
    return (w - 1)/(np.sqrt(w - a)*np.sqrt(w - b))

real_min, real_max = 0.0, 1200.0
imag_min, imag_max = -2.0, 600.0
small_step1, small_step2, medium_step, base_large_step = 0.005, 0.05, 1.0, 6.0
step = medium_step/5.0

found_w, found_z, lambda_values = [], [], []
real_part = real_min
while real_part <= real_max:
    if real_part < 3:
        step_size = small_step1
    elif real_part < 15:
        step_size = small_step2
    elif real_part <= 50:
        step_size = medium_step
    else:
        step_size = base_large_step + 6.0*np.floor((real_part - 50.0)/50.0)
    real_grid, imag_grid = np.meshgrid(
        np.arange(real_part, real_part + step_size + 1e-12, step_size),
        np.arange(imag_min, imag_max + 1e-12, step_size),
        indexing="xy",
    )
    w_grid = real_grid + 1j*imag_grid
    z_grid = calculate_z(w_grid)
    z_imag_close = np.abs(np.imag(z_grid) - target_imag_z) < 1e-2
    z_real_in    = (np.real(z_grid) >= target_real_z_min) & (np.real(z_grid) <= target_real_z_max)
    z_mask = z_imag_close & z_real_in
    if np.any(z_mask):
        vw = w_grid[z_mask]; vz = z_grid[z_mask]; vl = calculate_lambda(vw)
        found_w.append(vw.ravel()); found_z.append(vz.ravel()); lambda_values.append(vl.ravel())
    real_part += step

if len(found_w) == 0:
    raise RuntimeError("Nisu pronadene zadovoljavajuce tocke w.")

found_w = np.concatenate(found_w).astype(np.complex128)
found_z = np.concatenate(found_z).astype(np.complex128)
lambda_values = np.concatenate(lambda_values).astype(np.complex128)

real_lambda = np.real(lambda_values)
imag_lambda = np.imag(lambda_values)
valid = real_lambda > 0.45
filtered_real_lambda = real_lambda[valid]
filtered_imag_lambda = imag_lambda[valid]
filtered_real_z      = np.real(found_z[valid])

mirror_axis_real_z = -b_0_crtica/2.0
mirrored_real_z    = 2*mirror_axis_real_z - filtered_real_z

all_real_z=[]; all_lambda=[]; all_mirrored_real_z=[]; all_mirrored_lambda=[]
all_added_real_z=[]; all_added_lambda=[]; all_imag_lambda=[]; all_mirrored_imag_lambda=[]
num_repeats = Ns
spacing     = opseg_rotora/num_repeats
offset_shift = b_0_crtica/2.0
for repeat_idx in range(0, num_repeats + 2):
    offset = repeat_idx*spacing + offset_shift
    idx_orig = (filtered_real_z + offset >= 0) & (filtered_real_z + offset <= opseg_rotora)
    if np.any(idx_orig):
        all_real_z.append(filtered_real_z[idx_orig] + offset)
        all_lambda.append(filtered_real_lambda[idx_orig])
        all_imag_lambda.append(filtered_imag_lambda[idx_orig])
    idx_m = (mirrored_real_z + offset >= 0) & (mirrored_real_z + offset <= opseg_rotora)
    if np.any(idx_m):
        all_mirrored_real_z.append(mirrored_real_z[idx_m] + offset)
        all_mirrored_lambda.append(filtered_real_lambda[idx_m])
        all_mirrored_imag_lambda.append(-filtered_imag_lambda[idx_m])
    x1_vals = np.arange(offset_shift + sirina_papuce/20*7,
                        (sirina_papuce/20*13 + offset_shift) + 1e-12, sirina_papuce/20)
    x1_vals = x1_vals + repeat_idx*spacing
    mask = (x1_vals >= 0) & (x1_vals <= opseg_rotora)
    if np.any(mask):
        all_added_real_z.append(x1_vals[mask])
        all_added_lambda.append(np.ones(np.count_nonzero(mask)))

def _concat(lst):
    return np.concatenate(lst) if len(lst) > 0 else np.array([], dtype=float)

all_real_z=_concat(all_real_z); all_lambda=_concat(all_lambda)
all_mirrored_real_z=_concat(all_mirrored_real_z); all_mirrored_lambda=_concat(all_mirrored_lambda)
all_added_real_z=_concat(all_added_real_z); all_added_lambda=_concat(all_added_lambda)
all_imag_lambda=_concat(all_imag_lambda); all_mirrored_imag_lambda=_concat(all_mirrored_imag_lambda)

combined_real_z = np.concatenate([all_real_z, all_mirrored_real_z, all_added_real_z])
combined_lambda = np.concatenate([all_lambda, all_mirrored_lambda, all_added_lambda])
sidx = np.argsort(combined_real_z, kind="mergesort")
uz, uidx = np.unique(combined_real_z[sidx], return_index=True)
unique_real_z = uz; unique_lambda = combined_lambda[sidx][uidx]

num_points = 10000
interpolated_real_z = np.linspace(np.min(unique_real_z), np.max(unique_real_z), num_points)
interp_fun = interp1d(unique_real_z, unique_lambda, kind="linear", bounds_error=False, fill_value=np.nan)
interpolated_lambda = interp_fun(interpolated_real_z)
if np.isnan(interpolated_lambda).any():
    nn = ~np.isnan(interpolated_lambda)
    interpolated_lambda[np.isnan(interpolated_lambda)] = np.interp(
        np.flatnonzero(np.isnan(interpolated_lambda)), np.flatnonzero(nn), interpolated_lambda[nn])

combined_real_z_imag = np.concatenate([all_real_z, all_mirrored_real_z])
combined_imag_lambda = np.concatenate([all_imag_lambda, all_mirrored_imag_lambda])
sidx_im = np.argsort(combined_real_z_imag, kind="mergesort")
uz_im, uidx_im = np.unique(combined_real_z_imag[sidx_im], return_index=True)
unique_real_z_imag = uz_im; unique_imag_lambda = combined_imag_lambda[sidx_im][uidx_im]
interpolated_imag_real_z = np.linspace(np.min(unique_real_z_imag), np.max(unique_real_z_imag), num_points)
interp_fun_im = interp1d(unique_real_z_imag, unique_imag_lambda, kind="linear", bounds_error=False, fill_value=np.nan)
interpolated_imag_lambda = interp_fun_im(interpolated_imag_real_z)
if np.isnan(interpolated_imag_lambda).any():
    nn = ~np.isnan(interpolated_imag_lambda)
    interpolated_imag_lambda[np.isnan(interpolated_imag_lambda)] = np.interp(
        np.flatnonzero(np.isnan(interpolated_imag_lambda)), np.flatnonzero(nn), interpolated_imag_lambda[nn])


#Polje od permanentnih magneta B_PM(x), (rotor-fiksan, sredina raspora)
Br    = 1.309
alfa_pm = sirina_magneta/(opseg_rotora/Nm)
ni_r  = 1.03466875126278
Y_m   = debljina_magneta
g     = air_gap
Y_2   = Y_m + g
tau_m = opseg_rotora/Nm
n_max = 1000
n = np.arange(1, n_max + 1, 2)
U_n = 2*np.pi*n/(2*tau_m)
M_n = -4*Br/(n*np.pi*mu0)*np.sin(n*np.pi*alfa_pm/2)
delta = ni_r*np.cosh(U_n*Y_m)*np.sinh(U_n*(Y_2 - Y_m)) + np.cosh(U_n*(Y_2 - Y_m))*np.sinh(U_n*Y_m)

x = np.arange(0, 12*tau_m + 0.1, 0.1)
umn_y = (M_n[:, None]*np.sinh(U_n[:, None]*Y_m)/delta[:, None]
         * np.cosh(U_n[:, None]*(Y_2 - (Y_m + Y_2)/2))*np.cos(U_n[:, None]*x[None, :]))
#Zbrajanje harmonika
B_y1 = mu0*np.sum(umn_y, axis=0)
umn_x = (M_n[:, None]*np.sinh(U_n[:, None]*Y_m)/delta[:, None]
         * np.sinh(U_n[:, None]*(Y_2 - (Y_m + Y_2)/2))*np.sin(U_n[:, None]*x[None, :]))
B_x1 = mu0*np.sum(umn_x, axis=0)

valid_idx = (x >= 0) & (x <= opseg_rotora)
x_valid = x[valid_idx]; B_y1_valid = B_y1[valid_idx]; B_x1_valid = B_x1[valid_idx]
new_x = np.linspace(np.min(x_valid), np.max(x_valid), num_points)
B_y1_interp =  interp1d(x_valid, B_y1_valid, kind="cubic")(new_x)
B_x1_interp = -interp1d(x_valid, B_x1_valid, kind="cubic")(new_x) #Predznak: konvencija iz zavrsnog rada

#Uvođenje relativnog gibanja i izračun performansnih veličina
#Zajednicka periodicka mreza [0, C) (C je ovdje opseg):  na nju preslikavamo lambda i B_PM

Nx = 7200
x_grid = np.linspace(0.0, opseg_rotora, Nx, endpoint=False)  #mm , periodicka domena
dx_mm  = x_grid[1] - x_grid[0]

lam_a = np.interp(x_grid, interpolated_real_z,      interpolated_lambda)  #Re(lambda), stator
lam_b = np.interp(x_grid, interpolated_imag_real_z, interpolated_imag_lambda)  #Im(lambda), stator
By_PM = np.interp(x_grid, new_x, B_y1_interp)  #Rotor-fiksni bazni aksijalni PM tok
Bx_PM = np.interp(x_grid, new_x, B_x1_interp)  #Rotor-fiksni bazni tangencijalni PM tok

#Uzorkovanje periodicke funkcije f(x_grid) na proizvoljnim tockama xq (mm), period = C (ovdje opseg motora)
def sample_periodic(f, xq):
    xq_mod = np.mod(xq, opseg_rotora)
    xe = np.concatenate([x_grid, [x_grid[0] + opseg_rotora]])
    fe = np.concatenate([f, [f[0]]])
    return np.interp(xq_mod, xe, fe)

#Rezultantno polje u rasporu za mehanicki kut rotora theta, u rad
#Polje magneta se pomakne za x0 = R_av*theta (rotor klizi), lambda ostaje fiksna
#Vraca (By, Bx) na x_grid, u konvenciji zavrsnog rada

#By = -By_PM*lam_a + Bx_PM*lam_b
#Bx =  Bx_PM*lam_a - By_PM*lam_b

def slotted_field_at_angle(theta):
    x0 = radijus_rotora*theta #[mm]
    By_s = sample_periodic(By_PM, x_grid - x0)
    Bx_s = sample_periodic(Bx_PM, x_grid - x0)
    By = -By_s*lam_a + Bx_s*lam_b
    Bx =  Bx_s*lam_a - By_s*lam_b
    return By, Bx


#Lokacije zubaca (pronalazi se iz funkcije provodljivosti); utor = minimum Re(lambda)
slot_pitch = opseg_rotora/Ns
first_window = x_grid < slot_pitch
i_slot0 = np.argmin(lam_a[first_window])
x_slot0 = x_grid[first_window][i_slot0] #Centar prvog utora
tooth_centers = (x_slot0 + slot_pitch/2.0 + np.arange(Ns)*slot_pitch) % opseg_rotora
tooth_centers = np.sort(tooth_centers)

#Raspored namota
#12-zub / 10-pol dvoslojni koncentrirani obrazac
#Potpisani spoj svitka za svaku fazu; indeks = redni broj zupca (po rastucem x)
# +A:T1,T6  -A:T7,T12 | +B:T5,T10  -B:T4,T11 | +C:T2,T9  -C:T3,T8   (1-indeksirano)
winding = {
    "A": {1:+1, 6:+1, 7:-1, 12:-1},
    "B": {5:+1, 10:+1, 4:-1, 11:-1},
    "C": {2:+1, 9:+1, 3:-1, 8:-1},
}
phase_sign = {ph: np.zeros(Ns) for ph in winding}
for ph, d in winding.items():
    for tooth_1idx, s in d.items():
        phase_sign[ph][tooth_1idx - 1] = s

#Aksijalni tok kroz svaki zub (preko sirine papuce) za dano polje By(x_grid), Wb
def tooth_flux(By):
    half = sirina_papuce/2.0
    flux = np.zeros(Ns)
    for k, xc in enumerate(tooth_centers):
        # prozor sirine papuce oko centra zupca; koristi periodicko uzorkovanje
        xs = np.linspace(xc - half, xc + half, 400)
        Byk = sample_periodic(By, xs)
        integ_mm = _trapz(Byk, xs)                 # [T*mm]
        flux[k] = (L_r_mm*1e-3)*(integ_mm*1e-3)      # L_r[m] * integral[T*m] = Wb
    return flux


#Petlja po razl kutovima rotora  (jedna elektricna perioda = 2 koraka pola)
theta_max = 2*np.pi/p  #jedna elektr. perioda, mehanički radijani
n_theta   = 721
theta = np.linspace(0.0, theta_max, n_theta)

T_cog   = np.zeros(n_theta) #Polje za cogging moment, N*m
Psi     = {ph: np.zeros(n_theta) for ph in winding}  #Ulanceni tok po fazi, Wb-t

x_grid_m = x_grid*1e-3
for it, th in enumerate(theta):
    By, Bx = slotted_field_at_angle(th)
    #1) Cogging: Maxwellov tenzor u sredini raspora, integracija po cijelom opsegu
    T_cog[it] = n_airgaps*(radijus_rotora*1e-3)*(L_r_mm*1e-3)/mu0*_trapz(Bx*By, x_grid_m)
    #2) Ulanceni tok svake faze
    flux_k = tooth_flux(By)
    for ph in winding:
        Psi[ph][it] = N_c*np.sum(phase_sign[ph]*flux_k)

#Inducirani napon; e = -dPsi/dt = -omega * dPsi/dtheta
dtheta = theta[1] - theta[0]
EMF = {ph: -omega_mech*np.gradient(Psi[ph], dtheta) for ph in winding}

theta_elec_deg = np.degrees(theta*p) #elektricni stupnjevi
theta_mech_deg = np.degrees(theta) #mehanicki stupnjevi
t_ms = theta/omega_mech*1e3 #vrijeme, ms

def rms(a):
    return np.sqrt(np.mean(a**2))

#Grafovi
#(Nije uključen) Provjera: polje pri theta = 0 mora reproducirati zavrsni rad
By0, Bx0 = slotted_field_at_angle(0.0)
plt.figure(figsize=(9,4))
plt.plot(x_grid, By0, label=r"$B_y$ ($\theta=0$)")
plt.plot(x_grid, Bx0, label=r"$B_x$ ($\theta=0$)", alpha=0.8)
plt.xlabel("opseg [mm]"); plt.ylabel(r"$B$ [T]")
plt.title("Provjera: rezultantno polje u rasporu pri $\\theta=0$")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("Polje_theta0.png", dpi=130)