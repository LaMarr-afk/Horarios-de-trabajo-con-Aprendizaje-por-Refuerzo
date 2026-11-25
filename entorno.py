# ══════════════════════════════════════════════════════════════════════════════
#  entorno.py — Entorno de RL para construir el horario diario de M trabajadores
#
#  TODO EN UN SOLO ARCHIVO. Cómo leerlo, en orden:
#    1. CONSTANTES Y PARÁMETROS     números del problema y dos parámetros ajustables
#    2. REGLAS DE UN TRABAJADOR     qué acciones son válidas (R1-R5) y backtracking
#                                   para saber si el horario aún se puede completar
#    3. ALMUERZOS SIMULTÁNEOS       la regla "máximo K almorzando a la vez"
#    4. EL ENTORNO                  HorarioMultiEnv (observación, máscara, recompensa)
#    5. VALIDADOR Y UTILIDADES      validar_multi, mask_fn, generar_horario, ...
#
#  Uso típico:
#      from entorno import *
#      env = HorarioMultiEnv(num_trabajadores=8, demanda=DEMANDA_REAL, max_almorzando=5)
#
#  Si cambias MIN_TRABAJO_TRAS_DESCANSO o PESO_ALMUERZO, REINICIA el kernel: las
#  funciones con caché (lru_cache) guardan resultados calculados con el valor viejo.
# ══════════════════════════════════════════════════════════════════════════════
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ══════════════════════════════════════════════════════════════════════════════
#  1. CONSTANTES Y PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════
N_SLOTS        = 46    # 7:30 → 18:45, cada slot = 15 min
N_ACTIONS      = 4

SLOTS_JORNADA  = 32    # 8h: Trabaja + Pausa Activa
SLOTS_ALMUERZO = 6     # 1h30, NO cuenta en la jornada

# Cuántos slots ocupa cada acción
ACCION_SLOTS = {
    0: 1,  # No trabaja
    1: 1,  # Pausa Activa
    2: 6,  # Almuerzo
    3: 1,  # Trabaja
}

NO_TRABAJA  = 0
PAUSA       = 1
ALMUERZO    = 2
TRABAJA     = 3

MIN_TRABAJO_CONTINUO = 4   # slots (1h) mínimo antes de pausa/almuerzo      (R1)
MAX_TRABAJO_CONTINUO = 8   # slots (2h) máximo sin pausa                    (R2)

SLOT_ALM_MIN = 16  # 11:30 — primer slot en que se puede empezar el almuerzo
SLOT_ALM_MAX = 24  # 13:30 — último slot en que se puede empezar el almuerzo

# ── Parámetros ajustables ─────────────────────────────────────────────────────
# Después de una pausa o un almuerzo deben quedar al menos estos slots de jornada
# por trabajar (4 = 1 hora). Evita "descansar y se va". 0 desactiva la regla.
MIN_TRABAJO_TRAS_DESCANSO = 4

# Peso del castigo por la cobertura que se pierde durante el almuerzo.
PESO_ALMUERZO = 0.3

N_VENT = SLOT_ALM_MAX - SLOT_ALM_MIN + 1        # 9 posibles inicios de almuerzo
_N_OCC = N_VENT + SLOTS_ALMUERZO - 1            # 14 slots donde alguien puede almorzar


def generar_etiquetas():
    et = []
    for i in range(N_SLOTS):
        t = 7*60 + 30 + i*15
        h, m = divmod(t, 60)
        et.append(f"{h:02d}:{m:02d}")
    return et

ETIQUETAS = generar_etiquetas()


def costo_slot(oferta, demanda):
    """Costo de cobertura de UN slot: el déficit pesa 1.5 y el exceso 0.5."""
    return 1.5 * (demanda - oferta) if oferta < demanda else 0.5 * (oferta - demanda)


# ══════════════════════════════════════════════════════════════════════════════
#  2. REGLAS DE UN TRABAJADOR Y BACKTRACKING
#
#  Estado de un trabajador mientras se arma su horario:
#      est = (slot, jornada_rest, almuerzo_tomado, trabajo_continuo, iniciada, ultimo)
#  Las funciones de esta sección solo dependen de ese estado (son "puras").
# ══════════════════════════════════════════════════════════════════════════════

def _locales_base(s, jr, alm, cont, ini):
    """Acciones permitidas por las reglas R1-R5 en el slot s (sin mirar el futuro)."""
    if s >= N_SLOTS:
        return []
    v = []
    # R4: "No trabaja" solo antes de empezar, o cuando ya terminó todo
    if (not ini) or (jr == 0 and alm):
        v.append(NO_TRABAJA)
    # R1: pausa solo tras >= 4 slots de trabajo seguido, y si queda jornada
    if jr >= 1 and cont >= MIN_TRABAJO_CONTINUO:
        v.append(PAUSA)
    # Almuerzo: dentro de la ventana, tras >= 4 slots de trabajo, y que quepan los 6 slots
    if ((not alm) and SLOT_ALM_MIN <= s <= SLOT_ALM_MAX
            and cont >= MIN_TRABAJO_CONTINUO and N_SLOTS - s >= SLOTS_ALMUERZO):
        v.append(ALMUERZO)
    # R2: no más de 8 slots de trabajo seguidos
    if jr >= 1 and cont < MAX_TRABAJO_CONTINUO:
        v.append(TRABAJA)
    return v


def _descanso_ok(a, jr):
    """Regla de realismo: tras una pausa o un almuerzo deben quedar slots por trabajar.
    jr = slots de jornada que faltan ANTES de la acción a."""
    if a == PAUSA:
        return jr - 1 >= MIN_TRABAJO_TRAS_DESCANSO     # la pausa también consume jornada
    if a == ALMUERZO:
        return jr >= MIN_TRABAJO_TRAS_DESCANSO         # el almuerzo no consume jornada
    return True


def _locales(s, jr, alm, cont, ini):
    """R1-R5 + regla de trabajo mínimo tras descanso."""
    return [a for a in _locales_base(s, jr, alm, cont, ini) if _descanso_ok(a, jr)]


def _siguiente(est, a):
    """Estado que resulta de aplicar la acción a."""
    s, jr, alm, cont, ini, ult = est
    if a == NO_TRABAJA: return (s + 1, jr, alm, cont, ini, ult)
    if a == PAUSA:      return (s + 1, jr - 1, alm, 0, True, PAUSA)
    if a == ALMUERZO:   return (s + SLOTS_ALMUERZO, jr, True, 0, True, ALMUERZO)
    return (s + 1, jr - 1, alm, cont + 1, True, TRABAJA)


_TODOS_LOS_INICIOS = (1 << N_VENT) - 1      # máscara de bits con los 9 inicios de almuerzo


@lru_cache(maxsize=None)
def _factible(est, permitidos):
    """BACKTRACKING: ¿existe alguna forma de completar el horario desde este estado
    cumpliendo todas las reglas (R3 y R5 incluidas)?

    'permitidos' es una máscara de bits con los inicios de almuerzo que se pueden
    usar (bit i = slot SLOT_ALM_MIN + i). Con todos los bits en 1 no hay restricción.
    El resultado se guarda en caché, así que cada estado se calcula una sola vez."""
    s, jr, alm, cont, ini, ult = est
    if s >= N_SLOTS:
        return s == N_SLOTS and jr == 0 and alm and ult == TRABAJA
    for a in _locales(s, jr, alm, cont, ini):
        if a == ALMUERZO and not (permitidos >> (s - SLOT_ALM_MIN)) & 1:
            continue
        if _factible(_siguiente(est, a), permitidos):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  3. ALMUERZOS SIMULTÁNEOS: máximo K trabajadores almorzando a la vez
#
#  Los horarios se construyen de uno en uno, así que para el trabajador actual ya
#  se sabe cuántos almuerzos hay en cada slot. Un inicio de almuerzo es "permitido"
#  solo si (a) no supera K en ninguno de sus 6 slots y (b) los trabajadores que
#  faltan todavía pueden colocar el suyo sin superar K. Sin (b), los últimos
#  podrían quedarse sin hueco.
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def _colocables(occ, r, k, s_min):
    """¿Pueden r trabajadores más colocar su almuerzo (inicio entre SLOT_ALM_MIN y
    SLOT_ALM_MAX) sin que ningún slot pase de k almorzando? occ = ocupación actual."""
    if r == 0:
        return True
    for s in range(s_min, SLOT_ALM_MAX + 1):
        i = s - SLOT_ALM_MIN
        if all(occ[j] < k for j in range(i, i + SLOTS_ALMUERZO)):
            nuevo = list(occ)
            for j in range(i, i + SLOTS_ALMUERZO):
                nuevo[j] += 1
            if _colocables(tuple(nuevo), r - 1, k, s):
                return True
    return False


def k_minimo(M):
    """Menor K con el que M trabajadores pueden almorzar sin pasarse."""
    k = 1
    while not _colocables((0,) * _N_OCC, M, k, SLOT_ALM_MIN):
        k += 1
    return k


# ══════════════════════════════════════════════════════════════════════════════
#  4. EL ENTORNO
# ══════════════════════════════════════════════════════════════════════════════

class HorarioMultiEnv(gym.Env):
    """
    Construye los horarios de M trabajadores, uno tras otro (W1 completo, luego W2...).
    El agente decide slot a slot entre 4 acciones; un episodio dura unas M x 35 decisiones.

    Observación (todo escalado a rangos pequeños):
        trabajador_actual, slot_actual   dónde va
        horario_actual                   lo que lleva armado el trabajador actual
        horarios_previos                 los trabajadores ya terminados (M x 46)
        slots_jornada_rest, ...          cuánto le falta de jornada / almuerzo / trabajo seguido
        demanda                          demanda por slot
        deficit                          demanda - oferta de los trabajadores ya terminados
        def_ventana_alm                  déficit total de cada posible ventana de almuerzo

    Máscara de acciones (_acciones_validas): solo deja acciones que respetan las
    reglas y desde las cuales el horario todavía se puede completar (backtracking).
    Si max_almorzando=K, además nunca habrá más de K trabajadores almorzando a la vez.

    Recompensa:
        - Cada TRABAJA recibe el cambio real de costo de cobertura de su slot, según la
          oferta que ya dejaron los trabajadores anteriores. La suma de esos premios es
          igual a la penalización de cobertura del día completo.
        - El almuerzo se castiga según la cobertura que se pierde en sus 6 slots.
        - Pequeños premios/castigos por pausas, tiempos muertos y reglas rotas, y un
          bono (+30) por cada trabajador que cumple todas las reglas.

    Parámetros:
        num_trabajadores   M
        demanda            vector de 46 valores; None = demanda aleatoria en cada reset
        max_almorzando     K, máximo de almuerzos simultáneos; None = sin tope
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 num_trabajadores: int = 2,
                 demanda: Optional[np.ndarray] = None,
                 render_mode: Optional[str] = None,
                 max_almorzando: Optional[int] = None):
        super().__init__()

        self.M              = num_trabajadores
        self.demanda_fija   = demanda
        self.render_mode    = render_mode
        self.max_almorzando = max_almorzando
        if max_almorzando is not None and not _colocables((0,) * _N_OCC, self.M,
                                                          max_almorzando, SLOT_ALM_MIN):
            raise ValueError(f"max_almorzando={max_almorzando} es muy bajo para M={self.M}; "
                             f"el mínimo posible es {k_minimo(self.M)}")

        M = self.M
        self.observation_space = spaces.Dict({
            "trabajador_actual"  : spaces.Discrete(M),
            "slot_actual"        : spaces.Discrete(N_SLOTS + 1),
            "horario_actual"     : spaces.Box(-1.0, 1.0, shape=(N_SLOTS,),   dtype=np.float32),
            "horarios_previos"   : spaces.Box(-1.0, 1.0, shape=(M, N_SLOTS), dtype=np.float32),
            "slots_jornada_rest" : spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "slots_almuerzo_rest": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "almuerzo_tomado"    : spaces.Discrete(2),
            "jornada_iniciada"   : spaces.Discrete(2),
            "trabajo_continuo"   : spaces.Box(0.0, 2.0, shape=(1,), dtype=np.float32),
            "demanda"            : spaces.Box(0.0, 4.0, shape=(N_SLOTS,), dtype=np.float32),
            "deficit"            : spaces.Box(-4.0, 4.0, shape=(N_SLOTS,), dtype=np.float32),
            "def_ventana_alm"    : spaces.Box(0.0, 4.0, shape=(N_VENT,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

    # ── Estado del trabajador actual ─────────────────────────────────────────

    def _jornada_completa(self) -> bool:
        return self.slots_jornada_rest == 0

    def _todo_completo(self) -> bool:
        return self._jornada_completa() and self.almuerzo_tomado

    def _iniciar_trabajador(self):
        """Resetea el estado interno para empezar a asignar un nuevo trabajador."""
        self.slot_actual         = 0
        self.horario_actual      = np.full(N_SLOTS, -1, dtype=np.int32)
        self.slots_jornada_rest  = SLOTS_JORNADA
        self.slots_almuerzo_rest = SLOTS_ALMUERZO
        self.almuerzo_tomado     = False
        self.jornada_iniciada    = False
        self.trabajo_continuo    = 0

    def _estado(self):
        """El estado del trabajador actual, en el formato que usan las funciones de la sección 2."""
        ultimo = next((int(v) for v in reversed(self.horario_actual)
                       if v in (PAUSA, ALMUERZO, TRABAJA)), 0)
        return (self.slot_actual, self.slots_jornada_rest, self.almuerzo_tomado,
                self.trabajo_continuo, self.jornada_iniciada, ultimo)

    def _oferta(self):
        """Cuántos trabajadores ya terminados están en TRABAJA, por slot."""
        return (self.horarios == TRABAJA).sum(axis=0).astype(np.int32)

    def costo_cobertura(self) -> float:
        """Costo de cobertura del día completo (déficit x 1.5 + exceso x 0.5)."""
        of = self._oferta()
        return float(sum(costo_slot(int(of[s]), int(self.demanda[s])) for s in range(N_SLOTS)))

    # ── Máscara de acciones ──────────────────────────────────────────────────

    def _validas_locales(self) -> list:
        """Reglas del slot actual (sección 2), sin mirar si el horario se podrá completar."""
        est = self._estado()
        s, jr, alm, cont, ini, _ = est
        v = _locales_base(s, jr, alm, cont, ini) or [NO_TRABAJA]
        f = [a for a in v if _descanso_ok(a, jr)]
        return f or v

    def _almuerzos_permitidos(self) -> int:
        """Máscara de bits con los inicios de almuerzo permitidos para el trabajador actual
        (solo se usa si hay tope K; ver sección 3)."""
        t, k = self.trabajador_actual, self.max_almorzando
        occ = tuple(int((self.horarios[:t, SLOT_ALM_MIN + j] == ALMUERZO).sum())
                    for j in range(_N_OCC))
        faltan = self.M - t - 1                 # trabajadores después del actual
        mask = 0
        for s in range(SLOT_ALM_MIN, SLOT_ALM_MAX + 1):
            i = s - SLOT_ALM_MIN
            if all(occ[j] < k for j in range(i, i + SLOTS_ALMUERZO)):
                nuevo = list(occ)
                for j in range(i, i + SLOTS_ALMUERZO):
                    nuevo[j] += 1
                if _colocables(tuple(nuevo), faltan, k, SLOT_ALM_MIN):
                    mask |= 1 << i
        return mask

    def _acciones_validas(self) -> list:
        """Máscara final: reglas locales + 'desde ahí todavía se puede terminar' (backtracking)
        + tope de almuerzos simultáneos si hay K."""
        locales = self._validas_locales()
        if self.slot_actual >= N_SLOTS:
            return locales
        est        = self._estado()
        permitidos = (_TODOS_LOS_INICIOS if self.max_almorzando is None
                      else self._almuerzos_permitidos())
        seguras = []
        for a in locales:
            if a == ALMUERZO and not (permitidos >> (self.slot_actual - SLOT_ALM_MIN)) & 1:
                continue
            if _factible(_siguiente(est, a), permitidos):
                seguras.append(a)
        return seguras or locales

    # ── Observación e info ───────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        f = np.float32
        deficit = (self.demanda - self._oferta()).astype(f)
        pos = np.maximum(0, deficit)
        return {
            "trabajador_actual"  : self.trabajador_actual,
            "slot_actual"        : self.slot_actual,
            "horario_actual"     : (self.horario_actual / 3.0).astype(f),
            "horarios_previos"   : (self.horarios / 3.0).astype(f),
            "slots_jornada_rest" : np.array([self.slots_jornada_rest  / SLOTS_JORNADA],  dtype=f),
            "slots_almuerzo_rest": np.array([self.slots_almuerzo_rest / SLOTS_ALMUERZO], dtype=f),
            "almuerzo_tomado"    : int(self.almuerzo_tomado),
            "jornada_iniciada"   : int(self.jornada_iniciada),
            "trabajo_continuo"   : np.array([min(self.trabajo_continuo, MAX_TRABAJO_CONTINUO + 1)
                                             / MAX_TRABAJO_CONTINUO], dtype=f),
            "demanda"            : (self.demanda / 16.0).astype(f),
            "deficit"            : (deficit / 16.0).astype(f),
            "def_ventana_alm"    : np.array([pos[s:s + SLOTS_ALMUERZO].sum() / (6 * 16.0)
                                             for s in range(SLOT_ALM_MIN, SLOT_ALM_MAX + 1)], dtype=f),
        }

    def _get_info(self) -> dict:
        return {
            "trabajador_actual"  : self.trabajador_actual,
            "slot_actual"        : self.slot_actual,
            "slots_jornada_rest" : self.slots_jornada_rest,
            "slots_almuerzo_rest": self.slots_almuerzo_rest,
            "almuerzo_tomado"    : self.almuerzo_tomado,
            "trabajo_continuo"   : self.trabajo_continuo,
            "acciones_validas"   : self._acciones_validas(),
            "horarios"           : self.horarios.copy(),
        }

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Matriz de todos los horarios: shape (M, N_SLOTS), -1 = sin asignar
        self.horarios          = np.full((self.M, N_SLOTS), -1, dtype=np.int32)
        self.trabajador_actual = 0

        if self.demanda_fija is not None:
            self.demanda = np.asarray(self.demanda_fija).copy()
        else:
            self.demanda = self.np_random.integers(1, 17, size=N_SLOTS, dtype=np.int32)

        self._iniciar_trabajador()

        return self._get_obs(), self._get_info()

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        assert self.trabajador_actual < self.M, "Episodio terminado. Llama reset()."

        action    = int(action)          # model.predict devuelve un arreglo de NumPy
        reward    = 0.0
        truncated = False
        n_slots   = ACCION_SLOTS[action]
        s         = self.slot_actual

        accion_valida = action in self._acciones_validas()
        if not accion_valida:
            reward -= 8.0

        # ── Efectos de la acción ─────────────────────────────────────────────

        if action == NO_TRABAJA:
            if self.jornada_iniciada and not self._todo_completo():
                reward -= 6.0
            else:
                reward -= 0.02

        elif action == PAUSA:
            if accion_valida:
                reward += 0.5
                self.slots_jornada_rest -= 1
            self.trabajo_continuo = 0
            self.jornada_iniciada = True

        elif action == ALMUERZO:
            if accion_valida:
                # castigo por la cobertura que se pierde durante los 6 slots del almuerzo
                falta = np.maximum(0, self.demanda[s:s + SLOTS_ALMUERZO]
                                      - self._oferta()[s:s + SLOTS_ALMUERZO]).sum()
                reward -= PESO_ALMUERZO * 1.5 * float(falta)
                self.slots_almuerzo_rest = 0
                self.almuerzo_tomado     = True
            self.trabajo_continuo = 0
            self.jornada_iniciada = True

        elif action == TRABAJA:
            if accion_valida:
                # recompensa DENSA de cobertura: cuánto baja el costo de este slot
                # al añadir un trabajador más a la oferta que ya hay
                dem_s = int(self.demanda[s])
                of_s  = int(self._oferta()[s])
                reward -= costo_slot(of_s + 1, dem_s) - costo_slot(of_s, dem_s)
                self.slots_jornada_rest -= 1
            self.trabajo_continuo += 1
            self.jornada_iniciada  = True

        # ── Llenar horario del trabajador actual ─────────────────────────────
        slots_a_llenar = min(n_slots, N_SLOTS - self.slot_actual)
        for i in range(slots_a_llenar):
            self.horario_actual[self.slot_actual + i] = action
        self.slot_actual += slots_a_llenar

        # ── ¿Terminó el trabajador actual? ───────────────────────────────────
        trabajador_terminado = (self.slot_actual >= N_SLOTS)

        if trabajador_terminado:
            jornada_ok  = self._jornada_completa()
            almuerzo_ok = self.almuerzo_tomado

            # R5: último estado activo debe ser TRABAJA
            ultimo = None
            for v in reversed(self.horario_actual):
                if v not in (NO_TRABAJA, -1):
                    ultimo = v
                    break
            regla5_ok = (ultimo == TRABAJA)

            if jornada_ok and almuerzo_ok and regla5_ok:
                reward += 30.0   # bonus por trabajador bien asignado
            else:
                reward -= self.slots_jornada_rest  * 2.0
                reward -= self.slots_almuerzo_rest * 1.5
                if not almuerzo_ok:
                    reward -= 5.0
                if not regla5_ok:
                    reward -= 8.0

            # Guardar horario en la matriz global
            self.horarios[self.trabajador_actual] = self.horario_actual.copy()
            self.trabajador_actual += 1

            # ¿Quedan trabajadores por asignar?
            if self.trabajador_actual < self.M:
                self._iniciar_trabajador()

        # ── ¿Terminó el episodio? (todos los trabajadores asignados) ─────────
        # (la penalización final de cobertura ya se fue repartiendo en cada TRABAJA)
        terminated = (self.trabajador_actual >= self.M and trabajador_terminado)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode != "human":
            return

        nombres = {-1: "?", 0: "NO", 1: "PA", 2: "ALM", 3: "T"}
        iconos  = {-1: "⚪", 0: "🔴", 1: "🟠", 2: "🍽️ ", 3: "🟢"}

        print("\n" + "═" * (12 + self.M * 6))
        print(f"  HORARIO MULTI ({self.M} trabajadores)")
        print("═" * (12 + self.M * 6))

        header = f"  {'Hora':<6}  " + "  ".join([f"  W{w+1} " for w in range(self.M)]) + "  Dem  Oferta"
        print(header)
        print("─" * len(header))

        for s in range(N_SLOTS):
            row = f"  {ETIQUETAS[s]}  "
            oferta = 0
            for w in range(self.M):
                v = self.horarios[w, s]
                row += f" {iconos.get(v,'?')}{nombres.get(v,'?'):<3}"
                if v == TRABAJA:
                    oferta += 1
            dem  = self.demanda[s]
            diff = oferta - dem
            signo = "+" if diff >= 0 else ""
            row += f"  {dem:>3}  {oferta:>3} ({signo}{diff})"
            print(row)

        print("═" * len(header))
        # Resumen de cobertura
        oferta_total  = np.sum(self.horarios == TRABAJA, axis=0)
        diffs         = oferta_total.astype(int) - self.demanda.astype(int)
        deficit       = int(np.sum(diffs[diffs < 0]))
        exceso        = int(np.sum(diffs[diffs > 0]))
        print(f"  Déficit total (slots×trabajadores): {abs(deficit)}")
        print(f"  Exceso  total (slots×trabajadores): {exceso}")
        print(f"  MAE cobertura: {np.mean(np.abs(diffs)):.2f} trabajadores/slot")
        print("═" * (12 + self.M * 6))


# ══════════════════════════════════════════════════════════════════════════════
#  5. VALIDADOR Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
def validar_multi(horarios: np.ndarray, demanda: np.ndarray, verbose: bool = True) -> dict:
    """
    Valida la matriz completa (M, N_SLOTS) de horarios.
    Revisa R1-R5 por trabajador y la cobertura global.
    """
    M   = horarios.shape[0]
    res = {"por_trabajador": []}

    for w in range(M):
        h  = horarios[w]
        wr = {"trabajador": w + 1}

        # R3
        wr["R3_jornada"] = int(np.sum((h == PAUSA) | (h == TRABAJA))) == SLOTS_JORNADA
        wr["R3_almuerzo"] = int(np.sum(h == ALMUERZO)) == SLOTS_ALMUERZO

        # R4
        en_j, jt, sv, av, r4 = False, False, 0, False, True
        for v in h:
            if v in (TRABAJA, PAUSA):
                sv += 1; en_j = True
            elif v == ALMUERZO:
                en_j = True; av = True
            if sv >= SLOTS_JORNADA and av:
                jt = True
            if v == NO_TRABAJA and en_j and not jt:
                r4 = False; break
        wr["R4_continuidad"] = r4

        # R5
        ultimo = next((v for v in reversed(h) if v not in (NO_TRABAJA, -1)), None)
        wr["R5_termina_T"] = (ultimo == TRABAJA)

        # R1 y R2
        continuo, vr1, vr2 = 0, [], []
        for i, v in enumerate(h):
            if v == TRABAJA:
                continuo += 1
                if continuo > MAX_TRABAJO_CONTINUO:
                    vr2.append(i)
            elif v in (PAUSA, ALMUERZO):
                if 0 < continuo < MIN_TRABAJO_CONTINUO:
                    vr1.append(i)
                continuo = 0
        wr["R1_ok"] = len(vr1) == 0
        wr["R2_ok"] = len(vr2) == 0

        wr["CUMPLE_TODO"] = all([wr["R3_jornada"], wr["R3_almuerzo"],
                                  wr["R4_continuidad"], wr["R5_termina_T"],
                                  wr["R1_ok"], wr["R2_ok"]])
        res["por_trabajador"].append(wr)

    # Cobertura global
    oferta = np.sum(horarios == TRABAJA, axis=0).astype(int)
    diffs  = oferta - demanda.astype(int)
    res["MAE_cobertura"]   = float(np.mean(np.abs(diffs)))
    res["deficit_slots"]   = int(np.sum(diffs[diffs < 0]))
    res["exceso_slots"]    = int(np.sum(diffs[diffs > 0]))
    res["slots_perfectos"] = int(np.sum(diffs == 0))

    # R6: siempre hay al menos 1 trabajando
    res["R6_cobertura_minima"] = bool(np.all(oferta >= 1))

    if verbose:
        print("\n── Validación multi-trabajador ────────────────")
        for wr in res["por_trabajador"]:
            icono = "✅" if wr["CUMPLE_TODO"] else "❌"
            print(f"  {icono} W{wr['trabajador']}: jornada={wr['R3_jornada']} "
                  f"alm={wr['R3_almuerzo']} cont={wr['R4_continuidad']} "
                  f"R5={wr['R5_termina_T']} R1={wr['R1_ok']} R2={wr['R2_ok']}")
        print(f"\n  📊 Cobertura (oferta vs demanda):")
        print(f"     MAE         : {res['MAE_cobertura']:.2f} trabajadores/slot")
        print(f"     Déficit     : {abs(res['deficit_slots'])} slot·trabajador")
        print(f"     Exceso      : {res['exceso_slots']} slot·trabajador")
        print(f"     Slots exactos: {res['slots_perfectos']}/{N_SLOTS}")
        icono6 = "✅" if res["R6_cobertura_minima"] else "❌"
        print(f"  {icono6} R6 cobertura mínima (≥1 trabajando siempre): {res['R6_cobertura_minima']}")
        print("────────────────────────────────────────────────")

    return res


def mask_fn(env):
    """Máscara booleana de acciones válidas, en el formato que pide ActionMasker."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for a in env.unwrapped._acciones_validas():
        mask[a] = True
    return mask


def cargar_demanda(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=';', header=0, names=['fecha_hora', 'demanda'])
    df = df.dropna()
    df = df[df['demanda'].astype(str).str.strip() != '']
    return df['demanda'].astype(int).values[:N_SLOTS]


def generar_horario(model, num_trabajadores, demanda, max_almorzando=None, semilla=0,
                    render_mode=None):
    """Deja que el modelo arme un horario completo y devuelve el entorno terminado.
    Uso:  env = generar_horario(model, 8, DEMANDA_REAL, max_almorzando=5)
          env.horarios, env.costo_cobertura(), validar_multi(env.horarios, DEMANDA_REAL)"""
    env = HorarioMultiEnv(num_trabajadores=num_trabajadores, demanda=demanda,
                          max_almorzando=max_almorzando, render_mode=render_mode)
    obs, _ = env.reset(seed=semilla)
    terminado = False
    while not terminado:
        accion, _ = model.predict(obs, action_masks=mask_fn(env), deterministic=True)
        obs, _, terminado, _, _ = env.step(int(accion))
    return env


# Nombre que usaban los notebooks anteriores
HorarioMultiEnv2 = HorarioMultiEnv
