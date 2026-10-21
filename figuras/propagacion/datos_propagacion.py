"""
Genera los archivos .dat de las figuras del capítulo de propagación guiada.

    cd figuras/propagacion && python3 datos_propagacion.py

Dos modelos:
  * linea.py (unidades normalizadas L = C = 1, v = 1, pulso de ancho T = 1, línea de largo
    ELL = 3): solución exacta con generador de Thévenin y reflexiones múltiples.
    Figuras: pulso reflejado (fig:pulso reflejado) y ecos (fig:ecos).
  * Cable con efecto pelicular, ecuación (eq:H coaxil) sin el retardo t_d:
        H(f) = exp(-(1+j) a sqrt(f T)),   a = k ell / sqrt(T)   (f > 0, hermítica)
    Figuras: pulso NRZ por el cable, diagramas de ojo y amplificación de ruido del ZFE.
"""
import numpy as np
from linea import propagar_thevenin, ELL


def guardar(nombre, cabecera, columnas, fmt="%.5f"):
    np.savetxt(nombre, np.column_stack(columnas), fmt=fmt, header=cabecera, comments="")


# ---------------------------------------------------------------- línea con generador
DT = 1 / 80
N = int(512 / DT)
T_ALL = np.arange(N) * DT
PULSO = ((T_ALL >= 0) & (T_ALL < 1)).astype(float)


def pulso_reflejado():
    """Instantáneas v(z,t) para Z_G = Z_0, Z_L = 4 Z_0 (Gamma_L = 0.6)."""
    z = np.linspace(0, ELL, 301)
    tiempos = (1.5, 3.5, 4.5)
    n_keep = int(max(tiempos) / DT) + 2
    INC, REF = propagar_thevenin(PULSO, DT, z, n_keep, zs=1.0, z_l=4.0)
    cols, cab = [z], ["z"]
    for k, tt in enumerate(tiempos):
        n = int(round(tt / DT))
        cols += [INC[:, n], REF[:, n], INC[:, n] + REF[:, n]]
        cab += [f"inc{k}", f"ref{k}", f"tot{k}"]
    guardar("pulso_reflejado.dat", " ".join(cab), cols)


def ecos():
    """Tensión en la carga con ambos extremos desadaptados: Z_G = Z_0/4, Z_L = 4 Z_0."""
    z = np.array([ELL])
    t_end = 7 * ELL
    n_keep = int(t_end / DT) + 1
    INC, REF = propagar_thevenin(PULSO, DT, z, n_keep, zs=0.25, z_l=4.0)
    t = np.arange(n_keep) * DT
    v = (INC + REF)[0]
    k = slice(None, None, 2)
    guardar("ecos.dat", "t v", [t[k] / ELL, v[k]])        # t en unidades de t_d = ELL


# ---------------------------------------------------------------- cable (eq:H coaxil)
def h_cable(f, a, fase=True):
    if fase:
        return np.exp(-a * np.sqrt(2j * f))                # (1+j) sqrt(f) para f > 0
    return np.exp(-a * np.sqrt(np.abs(f)))


def respuesta_pulso(a, fase=True, dt=1 / 200, n=2 ** 16):
    t = (np.arange(n) - n // 4) * dt
    f = np.fft.fftfreq(n, dt)
    p = ((t >= 0) & (t < 1)).astype(float)
    y = np.fft.ifft(np.fft.fft(p) * h_cable(f, a, fase)).real
    return t, p, y


def pulso_cable(a=1.0):
    t, p, yc = respuesta_pulso(a)
    _, _, ya = respuesta_pulso(a, fase=False)
    m = (t >= -1.5) & (t <= 6)
    k = np.flatnonzero(m)[::4]
    guardar("pulso_cable.dat", "t tx aten completo", [t[k], p[k], ya[k], yc[k]])


def ojo(a, nombre, n_sim=300, n_trazas=64, seed=3):
    dt = 1 / 40
    rng = np.random.default_rng(seed)
    b = rng.choice([-1, 1], n_sim)
    ns = int(round(1 / dt))
    x = np.repeat(b, ns).astype(float)
    n = 2 ** int(np.ceil(np.log2(2 * len(x))))
    xs = np.zeros(n)
    xs[:len(x)] = x
    f = np.fft.fftfreq(n, dt)
    y = np.fft.ifft(np.fft.fft(xs) * h_cable(f, a)).real
    # trazas de dos símbolos, centradas en el instante de muestreo (fin de cada símbolo)
    filas = []
    tt = np.arange(-ns, ns + 1) * dt
    for k in range(30, 30 + n_trazas):
        c = (k + 1) * ns - 1                              # muestra al final del símbolo k
        seg = y[c - ns:c + ns + 1]
        filas += [np.column_stack([tt, seg]), np.array([[np.nan, np.nan]])]
    np.savetxt(nombre, np.vstack(filas), fmt="%.4f", header="t v", comments="")


def amplificacion_ruido(valores_a=(2.0, 4.0)):
    ff = np.linspace(0, 0.5, 101)
    cols, cab = [ff], ["f"]
    for i, a in enumerate(valores_a):
        t, _, y = respuesta_pulso(a)
        dt = t[1] - t[0]
        i0 = np.argmax(y)
        m = np.arange(-3, 200)
        pm = y[i0 + np.round(m / dt).astype(int)]
        Pd = np.abs(np.exp(-2j * np.pi * np.outer(ff, m)) @ pm)
        cols += [20 * np.log10(Pd), -20 * np.log10(Pd)]
        cab += [f"P{i}", f"C{i}"]
    guardar("amplificacion_ruido.dat", " ".join(cab), cols, fmt="%.4f")


if __name__ == "__main__":
    pulso_reflejado()
    ecos()
    pulso_cable()
    ojo(0.5, "ojo_corto.dat")
    ojo(2.0, "ojo_largo.dat")
    amplificacion_ruido()
