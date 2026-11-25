
def configurar_jornada():
    """
    Configura los parámetros de la jornada laboral,
    genera las etiquetas horarias y muestra el ambiente.

    Returns:
        dict: Diccionario con los parámetros de la jornada.
    """

    N_SLOTS = 46    # 7:30 → 18:45, cada slot = 15 min
    N_ACTIONS = 4

    SLOTS_JORNADA = 32   # 8h: Trabaja + Pausa Activa
    SLOTS_ALMUERZO = 6   # 1h30, NO cuenta en la jornada

    # Cuántos slots ocupa cada acción
    ACCION_SLOTS = {
        0: 1,  # No trabaja
        1: 1,  # Pausa Activa
        2: 6,  # Almuerzo
        3: 1,  # Trabaja
    }

    NO_TRABAJA = 0
    PAUSA = 1
    ALMUERZO = 2
    TRABAJA = 3

    MIN_TRABAJO_CONTINUO = 4   # slots (1h) mínimo antes de pausa/almuerzo
    MAX_TRABAJO_CONTINUO = 8   # slots (2h) máximo sin pausa

    SLOT_ALM_MIN = 16  # 11:30 — hora mínima de salida a almuerzo
    SLOT_ALM_MAX = 24  # 13:30 — hora máxima de salida a almuerzo

    def generar_etiquetas():
        etiquetas = []

        for i in range(N_SLOTS):
            t = 7 * 60 + 30 + i * 15
            h, m = divmod(t, 60)
            etiquetas.append(f"{h:02d}:{m:02d}")

        return etiquetas

    ETIQUETAS = generar_etiquetas()

    # Mostrar ambiente
    # print(
    #     'Slots:',
    #     N_SLOTS,
    #     '|',
    #     ETIQUETAS[0],
    #     '→',
    #     ETIQUETAS[-1]
    # )

    # print(
    #     f'Ventana almuerzo: '
    #     f'{ETIQUETAS[SLOT_ALM_MIN]} → {ETIQUETAS[SLOT_ALM_MAX]}'
    # )

    return {
        "N_SLOTS": N_SLOTS,
        "N_ACTIONS": N_ACTIONS,
        "SLOTS_JORNADA": SLOTS_JORNADA,
        "SLOTS_ALMUERZO": SLOTS_ALMUERZO,
        "ACCION_SLOTS": ACCION_SLOTS,
        "NO_TRABAJA": NO_TRABAJA,
        "PAUSA": PAUSA,
        "ALMUERZO": ALMUERZO,
        "TRABAJA": TRABAJA,
        "MIN_TRABAJO_CONTINUO": MIN_TRABAJO_CONTINUO,
        "MAX_TRABAJO_CONTINUO": MAX_TRABAJO_CONTINUO,
        "SLOT_ALM_MIN": SLOT_ALM_MIN,
        "SLOT_ALM_MAX": SLOT_ALM_MAX,
        "ETIQUETAS": ETIQUETAS,
    }
