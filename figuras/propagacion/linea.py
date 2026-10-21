"""
Modelo común de la línea con pérdidas y propagación de señales arbitrarias por FFT.

Unidades normalizadas: L = C = 1 (Z0 nominal = 1, v = 1/sqrt(LC) = 1), f0 = 1,
z en lambdas de la línea sin pérdidas a f0, t en períodos T = 1/f0.

  Z(f) = j 2pi f L + R0 sqrt(2 j f / f0)    efecto pelicular: Re{Z} = R0 sqrt(f/f0)
  Y(f) = j 2pi f C + 2pi |f| C tan(delta)   pérdidas dieléctricas
  gamma = sqrt(Z Y),  Z0 = Z / gamma,  Gamma_L = (Z_L - Z0)/(Z_L + Z0)

Fuente adaptada (Gamma_s = 0): la reflejada se absorbe al volver a z = 0.
"""
import numpy as np

ELL = 3.0
R0 = 0.20
TAN_D = 0.01
Z_L = 4.0


def parametros(f, r0=R0, tan_d=TAN_D, z_l=Z_L):
    Zs = 2j * np.pi * f + r0 * np.sqrt(2j * f)
    Ys = 2j * np.pi * f + 2 * np.pi * np.abs(f) * tan_d
    gam = np.sqrt(Zs * Ys)
    gam = np.where(gam.real < 0, -gam, gam)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z0 = np.where(f == 0, 1.0, Zs / np.where(gam == 0, 1, gam))
    GL = (z_l - Z0) / (z_l + Z0)
    return gam, Z0, GL


def propagar(s, dt, z, n_keep, ell=ELL, **kw):
    """
    s: excitación (incidente en z=0) muestreada con paso dt, real o compleja,
       con suficiente cola en cero para evitar el solapamiento circular.
    Devuelve INC, REF de forma (len(z), n_keep), con causalidad impuesta
    (nada por delante de los frentes que viajan a v = 1).
    """
    N = len(s)
    S = np.fft.fft(s)
    f = np.fft.fftfreq(N, dt)
    gam, _, GL = parametros(f, **kw)
    t = np.arange(n_keep) * dt
    t_on = dt * np.argmax(np.abs(s) > 0)   # instante en que arranca la excitación
    real = np.isrealobj(s)
    INC = np.empty((len(z), n_keep), complex)
    REF = np.empty((len(z), n_keep), complex)
    for i, zi in enumerate(z):
        INC[i] = np.fft.ifft(S * np.exp(-gam * zi))[:n_keep]
        REF[i] = np.fft.ifft(S * GL * np.exp(-gam * (2 * ell - zi)))[:n_keep]
        INC[i, t < t_on + zi - 1e-9] = 0
        REF[i, t < t_on + 2 * ell - zi - 1e-9] = 0
    if real:
        INC, REF = INC.real, REF.real
    return INC, REF


def propagar_thevenin(vs, dt, z, n_keep, zs=1.0, ell=ELL, **kw):
    """
    Solución exacta con generador de Thévenin (fem vs(t), impedancia interna zs) en z = 0:
      A(f)     = Vs(f) Z0 / (Z0 + Zs) / (1 - Gs GL e^{-2 gamma l})
      V(z, f)  = A e^{-gamma z}  +  A GL e^{-gamma (2l - z)}
    con Gs = (Zs - Z0)/(Zs + Z0). El denominador incluye todas las reflexiones múltiples
    (con zs = Z0 nominal y línea con pérdidas, Gs es pequeño pero no nulo).
    Devuelve INC, REF (onda hacia +z y hacia -z), reales, con causalidad impuesta.
    """
    N = len(vs)
    VS = np.fft.fft(vs)
    f = np.fft.fftfreq(N, dt)
    gam, Z0, GL = parametros(f, **kw)
    GS = (zs - Z0) / (zs + Z0)
    A = VS * Z0 / (Z0 + zs) / (1 - GS * GL * np.exp(-2 * gam * ell))
    t = np.arange(n_keep) * dt
    t_on = dt * np.argmax(np.abs(vs) > 0)
    INC = np.empty((len(z), n_keep))
    REF = np.empty((len(z), n_keep))
    for i, zi in enumerate(z):
        INC[i] = np.fft.ifft(A * np.exp(-gam * zi))[:n_keep].real
        REF[i] = np.fft.ifft(A * GL * np.exp(-gam * (2 * ell - zi)))[:n_keep].real
        INC[i, t < t_on + zi - 1e-9] = 0
        REF[i, t < t_on + 2 * ell - zi - 1e-9] = 0
    return INC, REF
