# ══════════════════════════════════════════════════════════════════════════════
#  visualizar.py — dibuja un horario multi-trabajador
#
#  Uso en el notebook (después de evaluar el modelo):
#      from visualizar import mostrar_horario
#      mostrar_horario(env_eval.horarios, DEMANDA_REAL, K=5)
#
#  Panel 1: horario de cada trabajador (verde = trabaja, A = almuerzo, P = pausa)
#  Panel 2: oferta (barras) contra demanda (línea roja)
#  Panel 3: cuántos almuerzan a la vez (con el techo K si se indica)
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

NO_TRABAJA, PAUSA, ALMUERZO, TRABAJA = 0, 1, 2, 3
# valor del horario -> índice de color (-1 = sin asignar)
_COLORES = {-1: "#ffffff", NO_TRABAJA: "#ececec", PAUSA: "#9ad8f0",
            ALMUERZO: "#f5c542", TRABAJA: "#a5e3a0"}


def _etiquetas(n, inicio_min=7 * 60 + 30, paso=15):
    return [f"{(inicio_min + paso * i) // 60:02d}:{(inicio_min + paso * i) % 60:02d}" for i in range(n)]


def mostrar_horario(horarios, demanda, K=None, guardar=None):
    H = np.asarray(horarios)
    M, N = H.shape
    demanda = np.asarray(demanda)[:N]
    oferta = (H == TRABAJA).sum(axis=0)
    almorzando = (H == ALMUERZO).sum(axis=0)
    et = _etiquetas(N)

    # índice de color por celda
    orden = [-1, NO_TRABAJA, PAUSA, ALMUERZO, TRABAJA]
    idx = np.vectorize(orden.index)(H).T                      # (N slots, M trabajadores)
    cmap = ListedColormap([_COLORES[v] for v in orden])

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15, 0.28 * N + 2.2), sharey=True,
        gridspec_kw={"width_ratios": [M * 0.55, 3.2, 1.8], "wspace": 0.08})

    # ── Panel 1: horario ─────────────────────────────────────────────────────
    ax1.imshow(idx, cmap=cmap, vmin=0, vmax=len(orden) - 1, aspect="auto", interpolation="nearest")
    for s in range(N):
        for w in range(M):
            v = H[w, s]
            if v == ALMUERZO: ax1.text(w, s, "A", ha="center", va="center", fontsize=7, color="#6b4e00")
            elif v == PAUSA:  ax1.text(w, s, "P", ha="center", va="center", fontsize=7, color="#0b5a73")
    ax1.set_xticks(range(M)); ax1.set_xticklabels([f"W{w + 1}" for w in range(M)])
    ax1.xaxis.tick_top()
    ax1.set_yticks(range(0, N, 2)); ax1.set_yticklabels(et[::2], fontsize=8)
    ax1.set_xticks(np.arange(-.5, M, 1), minor=True); ax1.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=0.8); ax1.tick_params(which="minor", length=0)
    for sp in ax1.spines.values(): sp.set_visible(False)
    ax1.set_title("Horario por trabajador", loc="left", fontsize=11, pad=22)

    # ── Panel 2: oferta vs demanda ───────────────────────────────────────────
    y = np.arange(N)
    deficit = oferta < demanda
    ax2.barh(y, oferta, height=0.8, color=np.where(deficit, "#e8a09a", "#9fc5e8"))
    # la demanda se dibuja como escalones alineados con cada fila
    ax2.stairs(demanda, np.arange(N + 1) - 0.5, orientation="horizontal",
               baseline=None, color="#c0392b", linewidth=2)
    ax2.set_xlim(0, max(demanda.max(), oferta.max()) + 1)
    ax2.set_title("Oferta vs demanda", loc="left", fontsize=11, pad=22)
    ax2.xaxis.tick_top(); ax2.tick_params(axis="y", length=0)
    ax2.legend(handles=[Patch(color="#9fc5e8", label="oferta (cubre)"),
                        Patch(color="#e8a09a", label="oferta (déficit)"),
                        plt.Line2D([], [], color="#c0392b", lw=2, label="demanda")],
               fontsize=7, loc="lower right", frameon=False)

    # ── Panel 3: almuerzos simultáneos ───────────────────────────────────────
    ax3.barh(y, almorzando, height=0.8, color="#f5c542")
    if K is not None:
        ax3.axvline(K, color="#6b4e00", linestyle="--", linewidth=1.5)
        ax3.text(K + 0.08, N - 1, f"K = {K}", ha="left", va="bottom", fontsize=8, color="#6b4e00")
    ax3.set_xlim(0, max(almorzando.max(), K or 0) + 1)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax3.set_title(f"Almorzando a la vez (máx. {almorzando.max()})", loc="left", fontsize=11, pad=22)
    ax3.xaxis.tick_top(); ax3.tick_params(axis="y", length=0)

    ax1.set_ylim(N - 0.5, -0.5)                                 # 07:30 arriba

    for ax in (ax2, ax3):
        for sp in ("right", "bottom", "left"): ax.spines[sp].set_visible(False)

    d = demanda.astype(int) - oferta.astype(int)
    alto = 0.28 * N + 2.2
    fig.suptitle(f"{M} trabajadores · déficit {int(d[d > 0].sum())} · "
                 f"exceso {int(-d[d < 0].sum())} · slots sin nadie: {int((oferta == 0).sum())}",
                 x=0.01, y=1 - 0.1 / alto, va="top", ha="left", fontsize=10, color="#444")
    fig.subplots_adjust(top=1 - 1.0 / alto)
    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
