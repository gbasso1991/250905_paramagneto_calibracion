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


def leer_file_pendientes(archivo):
    data=np.loadtxt(archivo)
    mean=np.mean(data[:])*1e14
    std=np.std(data[:])*1e14
    return ufloat(mean,std)

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


m = [m_081, m_135, m_212, m_239, m_270, m_300]
# Crear matriz para el heatmap (usando valores nominales)
m_nominal = np.array([m_081_nominal, m_135_nominal, m_212_nominal, 
                     m_239_nominal, m_270_nominal, m_300_nominal])

frecuencias = [81, 135, 212, 239, 270, 300]  # kHz
H0 = [20, 24, 27, 31, 35, 38, 42, 46, 50, 53, 57]  # amplitud de campo

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
