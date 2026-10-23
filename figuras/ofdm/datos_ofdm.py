"""
Datos del ejemplo de carga de bits del capítulo de OFDM.

    cd figuras/ofdm && python3 datos_ofdm.py

Par trenzado AWG24 del Ejemplo ejemplo:H canal (modelo de efecto pelicular):
atenuación 12.1 dB/km a 1 MHz, proporcional a sqrt(f). Tramo de 3 km.
256 subportadoras separadas 4.3125 kHz. Relación señal a ruido sin atenuación: 50 dB.
Probabilidad de error de símbolo objetivo en cada subportadora: 1e-6, con M-QAM cuadrada:
    P_e = 4 (1 - 1/sqrt(M)) Q( sqrt(3 SNR_k / (M - 1)) ).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

N, DF = 256, 4.3125e3
ELL_KM, A1MHZ, SNR0_DB, PE = 3.0, 12.1, 50.0, 1e-6


def snr_requerida(M):
    f = lambda s: 4 * (1 - 1 / np.sqrt(M)) * norm.sf(np.sqrt(3 * s / (M - 1))) - PE
    return 10 * np.log10(brentq(f, 1e-3, 1e8))


if __name__ == "__main__":
    umbrales = {b: snr_requerida(2 ** b) for b in (2, 4, 6, 8)}
    for b, s in umbrales.items():
        print(f"{2**b:4d}-QAM ({b} bits): SNR >= {s:.1f} dB")
    k = np.arange(1, N)
    f = k * DF
    aten = A1MHZ * ELL_KM * np.sqrt(f / 1e6)
    snr = SNR0_DB - aten
    bits = np.zeros_like(snr)
    for b, s in umbrales.items():
        bits[snr >= s] = b
    np.savetxt("carga_bits.dat", np.column_stack([f / 1e6, snr, bits]), fmt="%.5f",
               header="f snr bits", comments="")
    total = bits.sum()
    print(f"bits por símbolo OFDM: {total:.0f}; a 4000 símbolos/s: {total * 4000 / 1e6:.2f} Mbit/s")
    print(f"SNR en f={f[0]/1e3:.1f} kHz: {snr[0]:.1f} dB; en f={f[-1]/1e6:.3f} MHz: {snr[-1]:.1f} dB")
    for b in (0, 2, 4, 6, 8):
        print(b, int((bits == b).sum()))
