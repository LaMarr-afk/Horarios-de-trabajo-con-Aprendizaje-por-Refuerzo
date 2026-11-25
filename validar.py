import pandas as pd
import numpy as np
from acciones import configurar_jornada


config = configurar_jornada()

N_SLOTS = config["N_SLOTS"]
ETIQUETAS = config["ETIQUETAS"]
ACCION_SLOTS = config["ACCION_SLOTS"]
N_ACTIONS =config["N_ACTIONS"]
SLOTS_JORNADA = config["SLOTS_JORNADA"]
SLOTS_ALMUERZO= config["SLOTS_ALMUERZO"]
NO_TRABAJA= config["NO_TRABAJA"]
PAUSA=config["PAUSA"]
ALMUERZO=config["ALMUERZO"]
TRABAJA=config["TRABAJA"]
MIN_TRABAJO_CONTINUO= config["MIN_TRABAJO_CONTINUO"]
MAX_TRABAJO_CONTINUO=config["MAX_TRABAJO_CONTINUO"]
SLOT_ALM_MIN=config["SLOT_ALM_MIN"]
SLOT_ALM_MAX=config["SLOT_ALM_MAX"]

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

def cargar_demanda(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=';', header=0, names=['fecha_hora', 'demanda'])
    df = df.dropna()
    df = df[df['demanda'].astype(str).str.strip() != '']
    return df['demanda'].astype(int).values[:N_SLOTS]