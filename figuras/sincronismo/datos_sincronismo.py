"""
Genera los .dat de las figuras del capítulo de sincronismo (T_s = 1, sigma^2 = 1).

    cd figuras/sincronismo && python3 datos_sincronismo.py

  gardner_curva.dat : E[e_k] exacto en función de tau, pulso coseno alzado alpha = 0.35,
                      E[e_k] = sum_n p(n - 1/2 + tau) [p(n + tau) - p(n - 1 + tau)].
  gardner_onda.dat  : onda PAM y(t) alrededor de una transición +1 -> -1.
"""
import numpy as np

ALPHA = 0.35


def p(t):
    t = np.asarray(t, float)
    den = 1 - (2 * ALPHA * t) ** 2
    with np.errstate(all="ignore"):
        v = np.sinc(t) * np.cos(np.pi * ALPHA * t) / den
    return np.where(np.abs(den) < 1e-9, np.pi / 4 * np.sinc(1 / (2 * ALPHA)), v)


def dp(t, h=1e-6):
    return (p(t + h) - p(t - h)) / (2 * h)


def esperanza_gardner(tau, K=200):
    n = np.arange(-K, K + 1)
    return np.sum(p(n - 0.5 + tau) * (p(n + tau) - p(n - 1 + tau)))


if __name__ == "__main__":
    taus = np.linspace(-0.5, 0.5, 81)
    E = np.array([esperanza_gardner(t) for t in taus])
    np.savetxt("gardner_curva.dat", np.column_stack([taus, E]), fmt="%.5f", header="tau E", comments="")
    n = np.arange(1, 400)
    S = -2 * (dp(0.5) - np.sum(dp(n) * (p(n - 0.5) - p(n + 0.5))))
    print(f"pendiente exacta S = {S:.4f},  término dominante -2p'(1/2) = {-2 * dp(0.5):.4f}")

    # onda con símbolos antisimétricos alrededor de la transición entre n = 0 (+1) y n = 1 (-1)
    sim = {-4: -1, -3: 1, -2: -1, -1: 1, 0: 1, 1: -1, 2: -1, 3: 1, 4: -1, 5: 1}
    t = np.linspace(-1.5, 2.5, 321)
    y = sum(a * p(t - k) for k, a in sim.items())
    np.savetxt("gardner_onda.dat", np.column_stack([t, y]), fmt="%.5f", header="t y", comments="")
    for tau in (0.0, 0.25):
        ys = [sum(a * p(tt - k) for k, a in sim.items()) for tt in (tau, 0.5 + tau, 1 + tau)]
        print(f"tau={tau}: y_(k-1)={ys[0]:.3f} y_(k-1/2)={ys[1]:.3f} y_k={ys[2]:.3f}  e={ys[1]*(ys[2]-ys[0]):.3f}")


def p_alpha(t, a):
    t = np.asarray(t, float)
    if a == 0:
        return np.sinc(t)
    den = 1 - (2 * a * t) ** 2
    with np.errstate(all="ignore"):
        v = np.sinc(t) * np.cos(np.pi * a * t) / den
    return np.where(np.abs(den) < 1e-9, np.pi / 4 * np.sinc(1 / (2 * a)), v)


def media_cuadrado():
    """E[y^2(t)]/sigma^2 = sum_k p^2(t - k T_s), periódica de período T_s."""
    t = np.linspace(-1, 1, 201)
    k = np.arange(-400, 401)
    cols = [t]
    for a in (0.0, 0.35, 1.0):
        cols.append(np.array([np.sum(p_alpha(tt - k, a) ** 2) for tt in t]))
    np.savetxt("media_cuadrado.dat", np.column_stack(cols), fmt="%.5f", header="t a0 a35 a100", comments="")
    for a, c in zip((0, 0.35, 1), cols[1:]):
        print(f"alpha={a}: min {c.min():.4f} max {c.max():.4f}")


if __name__ == "__main__":
    media_cuadrado()


def plano_fase(zeta=0.7071, iniciales=(1.0, 2.0, 3.0, 3.5)):
    """Lazo de segundo orden con detector sinusoidal, en tiempo normalizado w_n t:
    theta'' + 2 zeta cos(theta) theta' + sin(theta) = 0, theta(0) = 0, theta'(0) = 2 pi Delta f / w_n.
    Guarda theta/pi y theta' para cada condición inicial."""
    from scipy.integrate import solve_ivp
    t = np.arange(0, 30, 0.05)
    cols, cab = [], []
    for i, w0 in enumerate(iniciales):
        s = solve_ivp(lambda tt, y: [y[1], -2 * zeta * np.cos(y[0]) * y[1] - np.sin(y[0])],
                      [0, 30], [0, w0], t_eval=t, max_step=0.01, rtol=1e-9, atol=1e-9)
        cols += [s.y[0] / np.pi, s.y[1]]
        cab += [f"th{i}", f"d{i}"]
        print(f"theta'(0)={w0}: engancha en theta = {s.y[0, -1] / np.pi:.2f} pi")
    np.savetxt("plano_fase.dat", np.column_stack(cols), fmt="%.5f", header=" ".join(cab), comments="")


if __name__ == "__main__":
    plano_fase()
