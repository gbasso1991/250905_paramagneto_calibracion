#Analis pendiente obtenida de curve_fit(lineal,campo_m,magnetizacion_ua_m_filtrada)
#%%
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from uncertainties import ufloat,unumpy
from glob import glob
import seaborn as sns
#%% Pendientes 
pend_300=glob('300_15_to_5/**/pendientes.txt',recursive=True)
pend_300.sort(reverse=True)

pend_270=glob('270_15_to_5/**/pendientes.txt',recursive=True)
pend_270.sort(reverse=True)

pend_239=glob('239_15_to_5/**/pendientes.txt',recursive=True)
pend_239.sort(reverse=True)

pend_212=glob('212_15_to_5/**/pendientes.txt',recursive=True)
pend_212.sort(reverse=True)

pend_135=glob('135_15_to_5/**/pendientes.txt',recursive=True)
pend_135.sort(reverse=True)

pend_081=glob('081_15_to_5/**/pendientes.txt',recursive=True)
pend_081.sort(reverse=True)

#%%
def leer_file_pendientes(archivo):
    data=np.loadtxt(archivo)
    mean=np.mean(data[:])*1e14
    std=np.std(data[:])*1e14
    return ufloat(mean,std)
#%%
m_300 = [leer_file_pendientes(fpath) for fpath in pend_300]
m_270 = [leer_file_pendientes(fpath) for fpath in pend_270]
m_239 = [leer_file_pendientes(fpath) for fpath in pend_239]
m_212 = [leer_file_pendientes(fpath) for fpath in pend_212]
m_135 = [leer_file_pendientes(fpath) for fpath in pend_135]
m_081 = [leer_file_pendientes(fpath) for fpath in pend_081]

# Extraer solo los valores nominales (mean) para el heatmap
m_300_nominal = [val.n for val in m_300]
m_270_nominal = [val.n for val in m_270]
m_239_nominal = [val.n for val in m_239]
m_212_nominal = [val.n for val in m_212]
m_135_nominal = [val.n for val in m_135]
m_081_nominal = [val.n for val in m_081]

#%%
m = [m_081, m_135, m_212, m_239, m_270, m_300]
# Crear matriz para el heatmap (usando valores nominales)
m_nominal = np.array([m_081_nominal, m_135_nominal, m_212_nominal, 
                     m_239_nominal, m_270_nominal, m_300_nominal])

frecuencias = [81, 135, 212, 239, 270, 300]  # kHz
H0 = [20, 24, 27, 31, 35, 38, 42, 46, 50, 53, 57]  # amplitud de campo
#%%
# Crear figura y ejes
plt.figure(figsize=(12, 6),constrained_layout=True)

# Crear heatmap con valores nominales
heatmap = sns.heatmap(
    m_nominal,
    xticklabels=H0,
    yticklabels=frecuencias,
    annot=m,  # Muestra los valores en las celdas
    fmt='.1uS',   # Formato de 3 decimales
    cmap='viridis',
    cbar_kws={'label': 'Pendiente m (x10^14) [Vs/A/m]'},
    linewidths=0.5,
    linecolor='gray'
)

# Configurar etiquetas y título
plt.xlabel('H$_0$ [kA/m]', fontsize=12, fontweight='bold')
plt.ylabel('Frecuencia [kHz]', fontsize=12, fontweight='bold')
plt.title('Heatmap de Pendiente m vs Frecuencia y Amplitud de campo H$_0$', fontsize=14, fontweight='bold')

# Rotar las etiquetas para mejor legibilidad
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('heatmap_pendiente_m_vs_frecuencia_amplitud_H0.png', dpi=300)
plt.show()
#%% Veo de corregir pendientes por inclinación del fondo
def corregir_por_inclinacion(pendientes, devolver_pendientes_corregidas=True, 
                             threshold_linear=1e-6, use_longdouble=True):
    """
    pendientes : iterable de floats (último elemento = referencia)
    devolver_pendientes_corregidas : si True devuelve pendientes corregidas (tan(delta_theta)),
                                     si False devuelve solo media y std de delta_theta (radianes)
    threshold_linear : si |m| < threshold_linear se usa la aproximación lineal arctan(m) ~ m
    use_longdouble : usar mayor precisión (np.longdouble) si está disponible
    """
    dtype = np.longdouble if use_longdouble else float
    arr = np.array(pendientes, dtype=dtype)
    m_ref = arr[-1]
    ms = arr[:-1]

    # criterio: si ambos (max absoluto) son muy pequeños, usar aproximación lineal
    if np.max(np.abs(arr)) < threshold_linear:
        # small-angle approx: delta_theta ≈ m - m_ref
        delta_theta = ms - m_ref
    else:
        # fórmula exacta para diferencia de arctan:
        # delta_theta = arctan((m_i - m_ref)/(1 + m_i*m_ref))
        numer = ms - m_ref
        denom = 1 + ms * m_ref
        ratio = numer / denom
        delta_theta = np.arctan(ratio)

    # estadísticas sobre delta_theta
    # por defecto devolvemos media/std de las PENDIENTES CORREGIDAS;
    # pero puedes querer media/std de ángulos (radianes). Aquí calculamos ambas opciones.
    media_delta = np.mean(delta_theta)
    std_delta = np.std(delta_theta, ddof=1)

    if devolver_pendientes_corregidas:
        # convertir de nuevo a pendiente corregida: m_corr = tan(delta_theta)
        pendientes_corr = np.tan(delta_theta)
        media_m = np.mean(pendientes_corr)
        std_m = np.std(pendientes_corr, ddof=1)
        return {
            "pendientes_corregidas": pendientes_corr,
            "media_pendientes_corregidas": media_m,
            "std_pendientes_corregidas": std_m,
            "media_delta_theta_rad": media_delta,
            "std_delta_theta_rad": std_delta
        }
    else:
        return {
            "media_delta_theta_rad": media_delta,
            "std_delta_theta_rad": std_delta,
            "delta_theta_array": delta_theta
        }

pendientes = [
    1.164212e-13, 1.160360e-13, 1.177865e-13, 1.158811e-13,
    1.170346e-13, 1.165802e-13, 1.162353e-13, 1.161804e-13,
    1.169756e-13, 1.169290e-13, 1.173470e-13, 1.160335e-13,
    1.163908e-13, 1.161268e-13, 1.165103e-13, 1.165971e-13,
    1.161385e-13, 1.152694e-13, 1.168027e-13, 1.169992e-13,
    1.157680e-13, 1.162035e-13, 2.218629e-14
]

res = corregir_por_inclinacion(pendientes)
for k,v in res.items():
    print(k, ":", v)

#%%
import numpy as np
import matplotlib.pyplot as plt

# Tus pendientes
pendientes = np.array([
    1.164212e-13, 1.160360e-13, 1.177865e-13, 1.158811e-13,
    1.170346e-13, 1.165802e-13, 1.162353e-13, 1.161804e-13,
    1.169756e-13, 1.169290e-13, 1.173470e-13, 1.160335e-13,
    1.163908e-13, 1.161268e-13, 1.165103e-13, 1.165971e-13,
    1.161385e-13, 1.152694e-13, 1.168027e-13, 1.169992e-13,
    1.157680e-13, 1.162035e-13, 2.218629e-14
])

m_ref = pendientes[-1]
ms = pendientes[:-1]

# Rango de campo
x = np.linspace(-24e3, 24e3, 200)  # -24 a 24 kA/m

# Loop de gráficas
for i, m in enumerate(ms, start=1):
    # calcular pendiente corregida restando ángulos
    delta_theta = np.arctan(m) - np.arctan(m_ref)
    m_corr = np.tan(delta_theta)

    # y-values
    y_orig = m * x
    y_ref = m_ref * x
    y_corr = m_corr * x

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(x, y_orig, label=f'Original m{i}', color='blue')
    plt.plot(x, y_ref, label='Referencia', color='red', linestyle='--')
    plt.plot(x, y_corr, label='Corregida', color='green')
    plt.xlabel('Campo (A/m)')
    plt.ylabel('Respuesta (u.a.)')
    plt.title(f'Comparación pendiente {i}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#%%





















#%% Amplitudes de señal
amp_300=glob('300_15_to_5/**/amplitudes.txt',recursive=True)
amp_300.sort(reverse=True)
amp_270=glob('270_15_to_5/**/amplitudes.txt',recursive=True)
amp_270.sort(reverse=True)
amp_239=glob('239_15_to_5/**/amplitudes.txt',recursive=True)
amp_239.sort(reverse=True)
amp_212=glob('212_15_to_5/**/amplitudes.txt',recursive=True)
amp_212.sort(reverse=True)
amp_135=glob('135_15_to_5/**/amplitudes.txt',recursive=True)
amp_135.sort(reverse=True)
amp_081=glob('081_15_to_5/**/amplitudes.txt',recursive=True)
amp_081.sort(reverse=True)

def leer_file_amplitudes(archivo):
    data=np.loadtxt(archivo,skiprows=2)
    mean=np.mean(data[:-1])
    std=np.std(data[:-1])
    return ufloat(mean,std)
#%%
a_300 = [leer_file_amplitudes(fpath) for fpath in amp_300]
a_270 = [leer_file_amplitudes(fpath) for fpath in amp_270]
a_239 = [leer_file_amplitudes(fpath) for fpath in amp_239]
a_212 = [leer_file_amplitudes(fpath) for fpath in amp_212]
a_135 = [leer_file_amplitudes(fpath) for fpath in amp_135]
a_081 = [leer_file_amplitudes(fpath) for fpath in amp_081]  

# Extraer solo los valores nominales (mean) para el heatmap
a_300_nominal = [val.n for val in a_300]
a_270_nominal = [val.n for val in a_270]
a_239_nominal = [val.n for val in a_239]
a_212_nominal = [val.n for val in a_212]
a_135_nominal = [val.n for val in a_135]
a_081_nominal = [val.n for val in a_081]        

a = [a_081, a_135, a_212, a_239, a_270, a_300]
# Crear matriz para el heatmap (usando valores nominales)
a_nominal = np.array([a_081_nominal, a_135_nominal, a_212_nominal, 
                     a_239_nominal, a_270_nominal, a_300_nominal])

# Crear figura y ejes
plt.figure(figsize=(12, 6),constrained_layout=True)

# Crear heatmap con valores nominales
heatmap = sns.heatmap(
    a_nominal,
    xticklabels=H0,
    yticklabels=frecuencias,
    annot=a,  # Muestra los valores en las celdas
    fmt='.1uS',   # Formato de 3 decimales
    cmap='viridis',
    cbar_kws={'label': 'Amplitud de señal [mV]'},
    linewidths=0.5,
    linecolor='gray'
)

# Configurar etiquetas y título
plt.xlabel('H$_0$ [kA/m]', fontsize=12, fontweight='bold')
plt.ylabel('Frecuencia [kHz]', fontsize=12, fontweight='bold')
plt.title('Heatmap de Amplitud señal vs Frecuencia y Amplitud de campo H$_0$', fontsize=14, fontweight='bold')

# Rotar las etiquetas para mejor legibilidad
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('heatmap_amplitud_señal_vs_frecuencia_amplitud_H0.png', dpi=300)
plt.show()






# %%
