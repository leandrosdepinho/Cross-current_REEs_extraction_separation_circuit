import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Simulador de Extração de Terras Raras",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Simulador de Extração por Solvente — Terras Raras")

st.markdown(
    """
    Este simulador calcula sequencialmente **extração → lavagem → reextração**
    considerando equilíbrio de extração, balanço de massa e balanço de H⁺.
    """
)

# ============================================================
# BANCO DE DADOS
# ============================================================

# Ordem adotada:
# La Ce Pr Nd Sm Eu Gd Tb Dy Ho Y Er Tm Yb Lu
#
# O Y foi colocado entre Ho e Er conforme definido.
#
# IMPORTANTE:
# O corte só pode ocorrer entre elementos vizinhos desta sequência.

METAIS = [
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Y", "Er", "Tm", "Yb", "Lu"
]

# Massa molar aproximada (g/mol)
MASSAS_MOLARES = {
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.249,
    "Tb": 158.92535,
    "Dy": 162.500,
    "Ho": 164.93033,
    "Y": 88.90584,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.045,
    "Lu": 174.9668
}

# Kex do modelo original
KEX = {
    "La": 1.26e-3,
    "Ce": 3.03e-3,
    "Pr": 6.67e-3,
    "Nd": 1.00e-2,
    "Sm": 3.00e-2,
    "Eu": 6.15e-2,
    "Gd": 1.23e-1,
    "Tb": 2.40e-1,
    "Dy": 4.56e-1,
    "Ho": 6.80e-1,
    "Y": 8.43e-1,
    "Er": 1.52e0,
    "Tm": 2.66e0,
    "Yb": 4.51e0,
    "Lu": 7.45e0
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def converter_para_mol_l(valor, unidade, metal):
    """
    Converte concentração informada pelo usuário para mol/L.
    """
    if unidade == "mol/L":
        return valor

    if unidade == "mmol/L":
        return valor / 1000.0

    if unidade == "mg/L":
        return (valor / 1000.0) / MASSAS_MOLARES[metal]

    if unidade == "g/L":
        return valor / MASSAS_MOLARES[metal]

    return 0.0


def converter_de_mol_l(valor, unidade, metal):
    """
    Converte mol/L para unidade escolhida pelo usuário.
    """
    if unidade == "mol/L":
        return valor

    if unidade == "mmol/L":
        return valor * 1000.0

    if unidade == "mg/L":
        return valor * MASSAS_MOLARES[metal] * 1000.0

    if unidade == "g/L":
        return valor * MASSAS_MOLARES[metal]

    return valor


def obter_grupos(corte):
    """
    Define leves e pesados.

    Exemplo:
    corte = Nd/Sm

    leves = La Ce Pr Nd
    pesados = Sm Eu Gd ...

    O Y está entre Ho e Er.
    """

    idx = METAIS.index(corte)

    leves = METAIS[:idx + 1]
    pesados = METAIS[idx + 1:]

    return leves, pesados


def soma_mols(vetor, metais):
    """
    Soma mol/L de uma seleção de metais.
    """
    indices = [METAIS.index(m) for m in metais]
    return np.sum(vetor[indices])


def percentual_seguro(a, b):
    if abs(b) < 1e-30:
        return 0.0
    return 100.0 * a / b


# ============================================================
# MODELO DE EXTRAÇÃO
# ============================================================

def calcular_estagio_extracao(
    caq_in,
    h_in,
    ext_monomer_total,
    saponificacao,
    AO
):
    """
    Calcula um estágio de extração.

    O usuário informa o extratante em concentração de MONÔMERO.

    Internamente:
        [extratante dimérico total] = concentração de monômero / 2

    O modelo de Kex utiliza o extratante livre na forma dimérica.

    A reação simplificada é:

        M(aq) + 3HL(org) ⇌ ML3(org) + 3H+

    e:

        D = Kex * [L]³ / [H+]³
    """

    # O extratante total usado pelo modelo é dimérico.
    ext_dim_total = ext_monomer_total / 2.0

    # Capacidade de neutralização associada à saponificação.
    # Saponificação é informada sobre o monômero.
    #
    # Cada mol de espécie dimérica possui 2 unidades monoméricas.
    # A quantidade de OH equivalente é, portanto, igual à fração
    # saponificada × concentração de monômero.
    capacidade_neutralizacao = saponificacao * ext_monomer_total

    def residual(log_vars):

        # Trabalhamos em log para garantir valores positivos.
        h_eq = np.exp(log_vars[0])
        l_eq = np.exp(log_vars[1])

        D = np.array([
            KEX[m] * (l_eq ** 3) / (h_eq ** 3)
            for m in METAIS
        ])

        caq_eq = caq_in / (1.0 + D * AO)

        corg_eq = caq_in * D * AO / (1.0 + D * AO)

        # Extratante livre dimérico.
        #
        # Cada complexo consome 3 moléculas do extratante
        # dimérico no modelo simplificado.
        ext_dim_calc = ext_dim_total - 3.0 * np.sum(corg_eq)

        if ext_dim_calc <= 0:
            ext_dim_calc = 1e-20

        # H+ gerado pela extração.
        h_gerado = 3.0 * np.sum(corg_eq)

        # Neutralização pela saponificação.
        h_esperado = h_in + max(
            0.0,
            h_gerado - capacidade_neutralizacao
        )

        r_h = np.log(h_eq / max(h_esperado, 1e-20))

        r_l = np.log(l_eq / ext_dim_calc)

        return [r_h, r_l]

    # Chute inicial
    h0 = max(h_in, 1e-8)
    l0 = max(ext_dim_total * 0.5, 1e-8)

    sol = least_squares(
        residual,
        np.log([h0, l0]),
        max_nfev=2000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    h_eq = np.exp(sol.x[0])
    l_eq = np.exp(sol.x[1])

    D = np.array([
        KEX[m] * (l_eq ** 3) / (h_eq ** 3)
        for m in METAIS
    ])

    caq_eq = caq_in / (1.0 + D * AO)
    corg_eq = caq_in * D * AO / (1.0 + D * AO)

    return caq_eq, corg_eq, h_eq, l_eq


# ============================================================
# MODELO DE LAVAGEM
# ============================================================

def calcular_estagio_lavagem(
    corg_in,
    h_in,
    ext_monomer_total,
    AO
):
    """
    Lavagem da fase orgânica com solução ácida sem adição
    deliberada de terra rara.

    A fase aquosa nova entra inicialmente sem ETR.

    O orgânico é progressivamente descarregado.

    O H+ recebido pelo estágio é o H+ do estágio anterior.
    """

    ext_dim_total = ext_monomer_total / 2.0

    # Na lavagem não há nova saponificação.
    capacidade_neutralizacao = 0.0

    caq_in = np.zeros(len(METAIS))

    def residual(log_vars):

        h_eq = np.exp(log_vars[0])
        l_eq = np.exp(log_vars[1])

        D = np.array([
            KEX[m] * (l_eq ** 3) / (h_eq ** 3)
            for m in METAIS
        ])

        # Balanço de massa em cada metal.
        #
        # C_org,in + C_aq,in / AO
        # =
        # C_org,eq + C_aq,eq / AO
        #
        # Como C_org = D*C_aq:
        caq_eq = (
            corg_in + caq_in / AO
        ) / (
            D + 1.0 / AO
        )

        corg_eq = D * caq_eq

        ext_dim_calc = (
            ext_dim_total -
            3.0 * np.sum(corg_eq)
        )

        ext_dim_calc = max(ext_dim_calc, 1e-20)

        # Durante a lavagem, quando metal sai da orgânica,
        # ocorre consumo de H+.
        metal_transferido = np.sum(corg_in - corg_eq)

        h_esperado = h_in - 3.0 * metal_transferido

        h_esperado = max(h_esperado, 1e-20)

        r_h = np.log(h_eq / h_esperado)
        r_l = np.log(l_eq / ext_dim_calc)

        return [r_h, r_l]

    h0 = max(h_in, 1e-8)
    l0 = max(ext_dim_total * 0.8, 1e-8)

    sol = least_squares(
        residual,
        np.log([h0, l0]),
        max_nfev=2000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    h_eq = np.exp(sol.x[0])
    l_eq = np.exp(sol.x[1])

    D = np.array([
        KEX[m] * (l_eq ** 3) / (h_eq ** 3)
        for m in METAIS
    ])

    caq_eq = (
        corg_in + caq_in / AO
    ) / (
        D + 1.0 / AO
    )

    corg_eq = D * caq_eq

    return caq_eq, corg_eq, h_eq, l_eq


# ============================================================
# MODELO DE REEXTRAÇÃO
# ============================================================

def calcular_estagio_reextracao(
    corg_in,
    h_in,
    ext_monomer_total,
    primeiro_estagio
):
    """
    Reextração usando solução ácida.

    O ácido é informado diretamente como [H+].

    No primeiro estágio ocorre a neutralização da saponificação
    residual. Depois disso, o H+ recebido é o H+ do estágio anterior.
    """

    ext_dim_total = ext_monomer_total / 2.0

    if primeiro_estagio:
        # Aqui a quantidade de saponificação residual será definida
        # externamente pela interface.
        pass

    caq_in = np.zeros(len(METAIS))

    def residual(log_vars):

        h_eq = np.exp(log_vars[0])
        l_eq = np.exp(log_vars[1])

        D = np.array([
            KEX[m] * (l_eq ** 3) / (h_eq ** 3)
            for m in METAIS
        ])

        caq_eq = (
            corg_in + caq_in
        ) / (
            D + 1.0
        )

        corg_eq = D * caq_eq

        ext_dim_calc = (
            ext_dim_total -
            3.0 * np.sum(corg_eq)
        )

        ext_dim_calc = max(ext_dim_calc, 1e-20)

        metal_extraido = np.sum(
            corg_in - corg_eq
        )

        h_esperado = (
            h_in -
            3.0 * metal_extraido
        )

        h_esperado = max(
            h_esperado,
            1e-20
        )

        r_h = np.log(h_eq / h_esperado)
        r_l = np.log(l_eq / ext_dim_calc)

        return [r_h, r_l]

    h0 = max(h_in * 0.9, 1e-8)
    l0 = max(ext_dim_total * 0.9, 1e-8)

    sol = least_squares(
        residual,
        np.log([h0, l0]),
        max_nfev=2000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    h_eq = np.exp(sol.x[0])
    l_eq = np.exp(sol.x[1])

    D = np.array([
        KEX[m] * (l_eq ** 3) / (h_eq ** 3)
        for m in METAIS
    ])

    caq_eq = corg_in / (D + 1.0)
    corg_eq = D * caq_eq

    return caq_eq, corg_eq, h_eq, l_eq


# ============================================================
# MÉTRICAS DO ESTÁGIO
# ============================================================

def calcular_metricas(
    caq_in,
    caq_out,
    corg_out,
    caq_original,
    corte
):

    leves, pesados = obter_grupos(corte)

    idx_leves = [METAIS.index(m) for m in leves]
    idx_pesados = [METAIS.index(m) for m in pesados]

    metricas = {}

    # --------------------------------------------------------
    # EXTRAÇÃO INDIVIDUAL
    # --------------------------------------------------------

    extracao_estagio = np.zeros(len(METAIS))
    extracao_acumulada = np.zeros(len(METAIS))

    for i, m in enumerate(METAIS):

        # Quanto foi retirado do que entrou no estágio
        if caq_in[i] > 0:
            extracao_estagio[i] = (
                (caq_in[i] - caq_out[i])
                / caq_in[i]
                * 100
            )

        # Quanto foi retirado em relação ao feed original
        if caq_original[i] > 0:
            extracao_acumulada[i] = (
                (caq_original[i] - caq_out[i])
                / caq_original[i]
                * 100
            )

    metricas["Extracao_estagio"] = extracao_estagio
    metricas["Extracao_acumulada"] = extracao_acumulada

    # --------------------------------------------------------
    # DISTRIBUIÇÃO NA FASE AQUOSA
    # --------------------------------------------------------

    mol_aq_leves = np.sum(caq_out[idx_leves])
    mol_aq_pesados = np.sum(caq_out[idx_pesados])
    mol_aq_total = mol_aq_leves + mol_aq_pesados

    metricas["Aq_%_leves"] = percentual_seguro(
        mol_aq_leves,
        mol_aq_total
    )

    metricas["Aq_%_pesados"] = percentual_seguro(
        mol_aq_pesados,
        mol_aq_total
    )

    # --------------------------------------------------------
    # DISTRIBUIÇÃO NA FASE ORGÂNICA
    # --------------------------------------------------------

    mol_org_leves = np.sum(corg_out[idx_leves])
    mol_org_pesados = np.sum(corg_out[idx_pesados])
    mol_org_total = mol_org_leves + mol_org_pesados

    metricas["Org_%_leves"] = percentual_seguro(
        mol_org_leves,
        mol_org_total
    )

    metricas["Org_%_pesados"] = percentual_seguro(
        mol_org_pesados,
        mol_org_total
    )

    # --------------------------------------------------------
    # DISTRIBUIÇÃO DOS LEVES ENTRE FASES
    # --------------------------------------------------------

    total_leves = mol_aq_leves + mol_org_leves

    metricas["Leves_%_aq"] = percentual_seguro(
        mol_aq_leves,
        total_leves
    )

    metricas["Leves_%_org"] = percentual_seguro(
        mol_org_leves,
        total_leves
    )

    # --------------------------------------------------------
    # DISTRIBUIÇÃO DOS PESADOS ENTRE FASES
    # --------------------------------------------------------

    total_pesados = mol_aq_pesados + mol_org_pesados

    metricas["Pesados_%_aq"] = percentual_seguro(
        mol_aq_pesados,
        total_pesados
    )

    metricas["Pesados_%_org"] = percentual_seguro(
        mol_org_pesados,
        total_pesados
    )

    return metricas


# ============================================================
# INTERFACE
# ============================================================

abas = st.tabs([
    "1️⃣ Extração",
    "2️⃣ Lavagem",
    "3️⃣ Reextração"
])


# ============================================================
# ABA 1 — EXTRAÇÃO
# ============================================================

with abas[0]:

    st.header("Extração")

    st.subheader("Composição do licor de alimentação")

    col1, col2 = st.columns([3, 1])

    with col1:
        unidade_feed = st.selectbox(
            "Unidade das concentrações",
            ["mg/L", "g/L", "mmol/L", "mol/L"],
            key="feed_unit"
        )

    with col2:
        ph_inicial = st.number_input(
            "pH inicial",
            min_value=0.0,
            max_value=14.0,
            value=1.0,
            step=0.1
        )

    st.markdown("### Terras raras no licor")

    valores_feed = {}

    cols = st.columns(5)

    for i, metal in enumerate(METAIS):

        with cols[i % 5]:

            valores_feed[metal] = st.number_input(
                f"{metal} ({unidade_feed})",
                min_value=0.0,
                value=0.0,
                format="%.8g",
                key=f"feed_{metal}"
            )

    st.markdown("---")

    st.subheader("Parâmetros de extração")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        AO = st.number_input(
            "Razão A/O",
            min_value=0.001,
            value=1.0,
            step=0.1
        )

    with c2:
        ext_monomer = st.number_input(
            "Extratatante total (mol/L de monômero)",
            min_value=0.000001,
            value=1.0,
            step=0.1
        )

    with c3:
        saponificacao = st.number_input(
            "Saponificação (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0
        ) / 100.0

    with c4:
        n_estagios = st.number_input(
            "Número inicial de estágios",
            min_value=1,
            max_value=200,
            value=5,
            step=1
        )

    corte = st.selectbox(
        "Par de corte",
        [
            f"{METAIS[i]}/{METAIS[i+1]}"
            for i in range(len(METAIS)-1)
        ]
    )

    metal_corte_1, metal_corte_2 = corte.split("/")

    leves, pesados = obter_grupos(metal_corte_2)

    st.info(
        f"**Leves:** {', '.join(leves)}  \n"
        f"**Pesados:** {', '.join(pesados)}"
    )

    pureza_desejada = st.number_input(
        "Pureza desejada dos leves na fase aquosa (%)",
        min_value=0.0,
        max_value=100.0,
        value=99.9,
        step=0.1
    )

    st.markdown("---")

    if st.button(
        "▶️ Iniciar simulação de extração",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # CONVERSÃO DO FEED
        # ----------------------------------------------------

        caq_original = np.array([
            converter_para_mol_l(
                valores_feed[m],
                unidade_feed,
                m
            )
            for m in METAIS
        ])

        # ----------------------------------------------------
        # ESTADO INICIAL
        # ----------------------------------------------------

        caq_atual = caq_original.copy()
        h_atual = 10 ** (-ph_inicial)

        historico = []

        # Volume aquoso arbitrário de referência = 1 L
        # O AO determina o volume orgânico correspondente.
        volume_aq = 1.0
        volume_org = AO

        # ----------------------------------------------------
        # LOOP
        # ----------------------------------------------------

        for estagio in range(1, int(n_estagios) + 1):

            caq_out, corg_out, h_out, ext_livre = (
                calcular_estagio_extracao(
                    caq_atual,
                    h_atual,
                    ext_monomer,
                    saponificacao,
                    AO
                )
            )

            metricas = calcular_metricas(
                caq_atual,
                caq_out,
                corg_out,
                caq_original,
                metal_corte_2
            )

            linha = {
                "Estágio": estagio,
                "H+ (mol/L)": h_out,
                "pH": -np.log10(max(h_out, 1e-20)),
                "Extratatante livre dimérico (mol/L)": ext_livre
            }

            for i, m in enumerate(METAIS):

                linha[f"Aq {m} (mol/L)"] = caq_out[i]
                linha[f"Org {m} (mol/L)"] = corg_out[i]

                linha[f"Extração {m} (%)"] = (
                    metricas["Extracao_estagio"][i]
                )

                linha[f"Extração acumulada {m} (%)"] = (
                    metricas["Extracao_acumulada"][i]
                )

            linha["Aq — leves (%)"] = metricas["Aq_%_leves"]
            linha["Aq — pesados (%)"] = metricas["Aq_%_pesados"]

            linha["Org — leves (%)"] = metricas["Org_%_leves"]
            linha["Org — pesados (%)"] = metricas["Org_%_pesados"]

            linha["Leves — Aq (%)"] = metricas["Leves_%_aq"]
            linha["Leves — Org (%)"] = metricas["Leves_%_org"]

            linha["Pesados — Aq (%)"] = metricas["Pesados_%_aq"]
            linha["Pesados — Org (%)"] = metricas["Pesados_%_org"]

            historico.append(linha)

            caq_atual = caq_out
            h_atual = h_out

        df_extracao = pd.DataFrame(historico)

        st.session_state["df_extracao"] = df_extracao
        st.session_state["caq_original"] = caq_original
        st.session_state["corg_extracao_final"] = np.array([
            historico[-1][f"Org {m} (mol/L)"]
            for m in METAIS
        ])
        st.session_state["h_extracao_final"] = h_atual
        st.session_state["ext_monomer"] = ext_monomer
        st.session_state["AO"] = AO
        st.session_state["corte"] = metal_corte_2

        st.success(
            "Simulação concluída. Agora você pode escolher o que deseja visualizar."
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    if "df_extracao" in st.session_state:

        df = st.session_state["df_extracao"]

        st.markdown("---")
        st.subheader("Resultados da extração")

        opcoes_resultado = st.multiselect(
            "O que você deseja visualizar?",
            [
                "Concentração na fase aquosa",
                "Concentração na fase orgânica",
                "Extratação no estágio",
                "Extração acumulada",
                "H+ e pH",
                "Extratatante livre",
                "Distribuição molar na aquosa",
                "Distribuição molar na orgânica",
                "Distribuição dos leves entre fases",
                "Distribuição dos pesados entre fases"
            ],
            default=[
                "Concentração na fase aquosa",
                "Concentração na fase orgânica"
            ]
        )

        metais_visualizacao = st.multiselect(
            "Metais a visualizar",
            METAIS,
            default=[metal_corte_1, metal_corte_2]
        )

        tabela = pd.DataFrame()
        tabela["Estágio"] = df["Estágio"]

        # ----------------------------------------------------
        # CONCENTRAÇÕES AQUOSAS
        # ----------------------------------------------------

        if "Concentração na fase aquosa" in opcoes_resultado:

            unidade_resultado = st.selectbox(
                "Unidade das concentrações",
                ["mg/L", "g/L", "mmol/L", "mol/L"],
                key="unidade_resultado"
            )

            for m in metais_visualizacao:

                tabela[
                    f"Aq {m} ({unidade_resultado})"
                ] = df[
                    f"Aq {m} (mol/L)"
                ].apply(
                    lambda x, metal=m:
                    converter_de_mol_l(
                        x,
                        unidade_resultado,
                        metal
                    )
                )

        # ----------------------------------------------------
        # CONCENTRAÇÕES ORGÂNICAS
        # ----------------------------------------------------

        if "Concentração na fase orgânica" in opcoes_resultado:

            unidade_resultado_org = st.selectbox(
                "Unidade das concentrações orgânicas",
                ["mg/L", "g/L", "mmol/L", "mol/L"],
                key="unidade_resultado_org"
            )

            for m in metais_visualizacao:

                tabela[
                    f"Org {m} ({unidade_resultado_org})"
                ] = df[
                    f"Org {m} (mol/L)"
                ].apply(
                    lambda x, metal=m:
                    converter_de_mol_l(
                        x,
                        unidade_resultado_org,
                        metal
                    )
                )

        # ----------------------------------------------------
        # EXTRAÇÃO NO ESTÁGIO
        # ----------------------------------------------------

        if "Extratação no estágio" in opcoes_resultado:

            for m in metais_visualizacao:
                tabela[
                    f"Extração {m} — estágio (%)"
                ] = df[
                    f"Extração {m} (%)"
                ]

        # ----------------------------------------------------
        # EXTRAÇÃO ACUMULADA
        # ----------------------------------------------------

        if "Extração acumulada" in opcoes_resultado:

            for m in metais_visualizacao:
                tabela[
                    f"Extração {m} — acumulada (%)"
                ] = df[
                    f"Extração acumulada {m} (%)"
                ]

        # ----------------------------------------------------
        # H+
        # ----------------------------------------------------

        if "H+ e pH" in opcoes_resultado:

            tabela["H+ (mol/L)"] = df["H+ (mol/L)"]
            tabela["pH"] = df["pH"]

        # ----------------------------------------------------
        # EXTRATANTE
        # ----------------------------------------------------

        if "Extratatante livre" in opcoes_resultado:

            tabela[
                "Extratatante livre dimérico (mol/L)"
            ] = df[
                "Extratatante livre dimérico (mol/L)"
            ]

        # ----------------------------------------------------
        # DISTRIBUIÇÃO AQ
        # ----------------------------------------------------

        if "Distribuição molar na aquosa" in opcoes_resultado:

            tabela["Aq — leves (%)"] = df["Aq — leves (%)"]
            tabela["Aq — pesados (%)"] = df["Aq — pesados (%)"]

        # ----------------------------------------------------
        # DISTRIBUIÇÃO ORG
        # ----------------------------------------------------

        if "Distribuição molar na orgânica" in opcoes_resultado:

            tabela["Org — leves (%)"] = df["Org — leves (%)"]
            tabela["Org — pesados (%)"] = df["Org — pesados (%)"]

        # ----------------------------------------------------
        # LEVES ENTRE FASES
        # ----------------------------------------------------

        if "Distribuição dos leves entre fases" in opcoes_resultado:

            tabela["Leves — Aq (%)"] = df["Leves — Aq (%)"]
            tabela["Leves — Org (%)"] = df["Leves — Org (%)"]

        # ----------------------------------------------------
        # PESADOS ENTRE FASES
        # ----------------------------------------------------

        if "Distribuição dos pesados entre fases" in opcoes_resultado:

            tabela["Pesados — Aq (%)"] = df["Pesados — Aq (%)"]
            tabela["Pesados — Org (%)"] = df["Pesados — Org (%)"]

        st.dataframe(
            tabela,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # AVISO DE PUREZA
        # ====================================================

        st.markdown("---")

        st.subheader("🎯 Pureza do corte")

        purezas = df["Aq — leves (%)"].values

        indice = None

        for i, pureza in enumerate(purezas):

            if pureza >= pureza_desejada:
                indice = i
                break

        if pureza_desejada <= purezas[0]:

            st.info(
                f"A solução aquosa já apresenta "
                f"**{purezas[0]:.4f}% de leves no estágio 0** "
                f"para o corte {metal_corte_1}/{metal_corte_2}."
            )

        elif indice is not None:

            est = int(df.iloc[indice]["Estágio"])

            st.success(
                f"🎯 A pureza de **{pureza_desejada:.3f}% de leves "
                f"na fase aquosa** é atingida no estágio **{est}**."
            )

        else:

            st.warning(
                f"A pureza de {pureza_desejada:.3f}% ainda não foi "
                f"atingida nos {len(df)} estágios calculados."
            )

        # ====================================================
        # CONTROLE DE CONTINUAÇÃO
        # ====================================================

        st.markdown("---")

        st.subheader("Próximo passo")

        continuar = st.button(
            "➕ Adicionar mais um estágio",
            use_container_width=True
        )

        parar = st.button(
            "🛑 Parar extração e ir para lavagem",
            type="primary",
            use_container_width=True
        )

        if continuar:

            # Usa o último estado conhecido e calcula mais um estágio
            ultima_linha = df.iloc[-1]

            caq_atual = np.array([
                ultima_linha[f"Aq {m} (mol/L)"]
                for m in METAIS
            ])

            h_atual = ultima_linha["H+ (mol/L)"]

            caq_out, corg_out, h_out, ext_livre = (
                calcular_estagio_extracao(
                    caq_atual,
                    h_atual,
                    st.session_state["ext_monomer"],
                    saponificacao,
                    st.session_state["AO"]
                )
            )

            metricas = calcular_metricas(
                caq_atual,
                caq_out,
                corg_out,
                st.session_state["caq_original"],
                st.session_state["corte"]
            )

            estagio = int(ultima_linha["Estágio"]) + 1

            linha = {
                "Estágio": estagio,
                "H+ (mol/L)": h_out,
                "pH": -np.log10(max(h_out, 1e-20)),
                "Extratatante livre dimérico (mol/L)": ext_livre
            }

            for i, m in enumerate(METAIS):

                linha[f"Aq {m} (mol/L)"] = caq_out[i]
                linha[f"Org {m} (mol/L)"] = corg_out[i]

                linha[f"Extração {m} (%)"] = (
                    metricas["Extracao_estagio"][i]
                )

                linha[f"Extração acumulada {m} (%)"] = (
                    metricas["Extracao_acumulada"][i]
                )

            linha["Aq — leves (%)"] = metricas["Aq_%_leves"]
            linha["Aq — pesados (%)"] = metricas["Aq_%_pesados"]

            linha["Org — leves (%)"] = metricas["Org_%_leves"]
            linha["Org — pesados (%)"] = metricas["Org_%_pesados"]

            linha["Leves — Aq (%)"] = metricas["Leves_%_aq"]
            linha["Leves — Org (%)"] = metricas["Leves_%_org"]

            linha["Pesados — Aq (%)"] = metricas["Pesados_%_aq"]
            linha["Pesados — Org (%)"] = metricas["Pesados_%_org"]

            st.session_state["df_extracao"] = pd.concat(
                [df, pd.DataFrame([linha])],
                ignore_index=True
            )

            st.session_state["corg_extracao_final"] = corg_out
            st.session_state["h_extracao_final"] = h_out

            st.rerun()

        if parar:

            st.session_state["extracao_finalizada"] = True

            st.success(
                "Extração encerrada. A carga orgânica acumulada "
                "será encaminhada para a etapa de lavagem."
            )


# ============================================================
# ABA 2 — LAVAGEM
# ============================================================

with abas[1]:

    st.header("Lavagem")

    if "corg_extracao_final" not in st.session_state:

        st.warning(
            "Execute a etapa de extração primeiro."
        )

    else:

        st.info(
            "A lavagem recebe a **carga orgânica total acumulada da extração**. "
            "Não é adicionada nenhuma terra rara à solução de lavagem."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            ph_lavagem = st.number_input(
                "pH da solução de lavagem",
                min_value=0.0,
                max_value=7.0,
                value=1.0,
                step=0.1
            )

        with c2:
            AO_lav = st.number_input(
                "A/O da lavagem",
                min_value=0.001,
                value=1.0,
                step=0.1
            )

        with c3:
            n_lav = st.number_input(
                "Número de estágios de lavagem",
                min_value=1,
                max_value=100,
                value=5
            )

        pureza_lavagem = st.number_input(
            "Pureza desejada dos pesados na fase orgânica (%)",
            min_value=0.0,
            max_value=100.0,
            value=99.9,
            step=0.1
        )

        corte_lavagem = st.session_state.get(
            "corte",
            "Nd"
        )

        if st.button(
            "▶️ Iniciar lavagem",
            type="primary",
            use_container_width=True
        ):

            corg_atual = (
                st.session_state["corg_extracao_final"]
                .copy()
            )

            h_atual = 10 ** (-ph_lavagem)

            historico_lav = []

            for estagio in range(1, int(n_lav) + 1):

                caq_out, corg_out, h_out, ext_livre = (
                    calcular_estagio_lavagem(
                        corg_atual,
                        h_atual,
                        st.session_state["ext_monomer"],
                        AO_lav
                    )
                )

                leves, pesados = obter_grupos(
                    corte_lavagem
                )

                idx_l = [
                    METAIS.index(m)
                    for m in leves
                ]

                idx_p = [
                    METAIS.index(m)
                    for m in pesados
                ]

                mol_org_l = np.sum(corg_out[idx_l])
                mol_org_p = np.sum(corg_out[idx_p])
                mol_org_total = (
                    mol_org_l + mol_org_p
                )

                pureza_pesados = percentual_seguro(
                    mol_org_p,
                    mol_org_total
                )

                linha = {
                    "Estágio": estagio,
                    "H+ (mol/L)": h_out,
                    "pH": -np.log10(max(h_out, 1e-20)),
                    "Extratatante livre dimérico (mol/L)": ext_livre,
                    "Org — leves (%)": percentual_seguro(
                        mol_org_l,
                        mol_org_total
                    ),
                    "Org — pesados (%)": pureza_pesados
                }

                for i, m in enumerate(METAIS):

                    linha[f"Aq {m} (mol/L)"] = caq_out[i]
                    linha[f"Org {m} (mol/L)"] = corg_out[i]

                historico_lav.append(linha)

                corg_atual = corg_out
                h_atual = h_out

            df_lav = pd.DataFrame(
                historico_lav
            )

            st.session_state["df_lavagem"] = df_lav
            st.session_state["corg_lavagem_final"] = corg_atual
            st.session_state["h_lavagem_final"] = h_atual

        if "df_lavagem" in st.session_state:

            df = st.session_state["df_lavagem"]

            st.subheader("Resultados da lavagem")

            opcoes = st.multiselect(
                "Informações para exibir",
                [
                    "Concentração na aquosa",
                    "Concentração na orgânica",
                    "H+ e pH",
                    "Extratatante livre",
                    "Distribuição molar na orgânica"
                ],
                default=[
                    "Concentração na orgânica",
                    "Distribuição molar na orgânica"
                ],
                key="opcoes_lavagem"
            )

            metais_lav = st.multiselect(
                "Metais a visualizar",
                METAIS,
                default=[
                    st.session_state["corte"],
                    obter_grupos(
                        st.session_state["corte"]
                    )[1][0]
                ],
                key="metais_lav"
            )

            tabela = pd.DataFrame()
            tabela["Estágio"] = df["Estágio"]

            if "Concentração na aquosa" in opcoes:

                for m in metais_lav:

                    tabela[
                        f"Aq {m} (mol/L)"
                    ] = df[
                        f"Aq {m} (mol/L)"
                    ]

            if "Concentração na orgânica" in opcoes:

                for m in metais_lav:

                    tabela[
                        f"Org {m} (mol/L)"
                    ] = df[
                        f"Org {m} (mol/L)"
                    ]

            if "H+ e pH" in opcoes:

                tabela["H+ (mol/L)"] = df["H+ (mol/L)"]
                tabela["pH"] = df["pH"]

            if "Extratatante livre" in opcoes:

                tabela[
                    "Extratatante livre dimérico (mol/L)"
                ] = df[
                    "Extratatante livre dimérico (mol/L)"
                ]

            if "Distribuição molar na orgânica" in opcoes:

                tabela["Org — leves (%)"] = df[
                    "Org — leves (%)"
                ]

                tabela["Org — pesados (%)"] = df[
                    "Org — pesados (%)"
                ]

            st.dataframe(
                tabela,
                use_container_width=True,
                hide_index=True
            )

            purezas = df[
                "Org — pesados (%)"
            ].values

            idx = None

            for i, p in enumerate(purezas):

                if p >= pureza_lavagem:
                    idx = i
                    break

            if idx is not None:

                est = int(
                    df.iloc[idx]["Estágio"]
                )

                st.success(
                    f"🎯 A pureza de {pureza_lavagem:.3f}% "
                    f"de pesados na orgânica é atingida no "
                    f"estágio **{est}**."
                )

            else:

                st.warning(
                    "A pureza desejada ainda não foi atingida."
                )

            if st.button(
                "🛑 Encerrar lavagem → Reextração",
                type="primary",
                use_container_width=True
            ):

                st.session_state[
                    "lavagem_finalizada"
                ] = True

                st.success(
                    "Lavagem encerrada. A carga orgânica "
                    "está disponível para a reextração."
                )


# ============================================================
# ABA 3 — REEXTRAÇÃO
# ============================================================

with abas[2]:

    st.header("Reextração")

    if "corg_lavagem_final" not in st.session_state:

        st.warning(
            "Execute a etapa de lavagem primeiro."
        )

    else:

        st.info(
            "A reextração recebe a carga orgânica final da lavagem."
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            H_acido = st.number_input(
                "Concentração de H+ da solução ácida (mol/L)",
                min_value=0.000001,
                value=2.0,
                step=0.1
            )

        with c2:

            AO_reext = st.number_input(
                "A/O da reextração",
                min_value=0.001,
                value=1.0,
                step=0.1
            )

        with c3:

            n_reext = st.number_input(
                "Número de estágios de reextração",
                min_value=1,
                max_value=100,
                value=5
            )

        st.markdown(
            """
            O objetivo da reextração é retirar os metais da fase orgânica
            para a fase aquosa. O simulador acompanha a quantidade
            reextraída acumulada.
            """
        )

        if st.button(
            "▶️ Iniciar reextração",
            type="primary",
            use_container_width=True
        ):

            corg_atual = (
                st.session_state[
                    "corg_lavagem_final"
                ].copy()
            )

            h_atual = H_acido

            corg_inicial = corg_atual.copy()

            historico_reext = []

            for estagio in range(
                1,
                int(n_reext) + 1
            ):

                caq_out, corg_out, h_out, ext_livre = (
                    calcular_estagio_reextracao(
                        corg_atual,
                        h_atual,
                        st.session_state[
                            "ext_monomer"
                        ],
                        estagio == 1
                    )
                )

                linha = {
                    "Estágio": estagio,
                    "H+ (mol/L)": h_out,
                    "pH": -np.log10(
                        max(h_out, 1e-20)
                    ),
                    "Extratatante livre dimérico (mol/L)": ext_livre
                }

                for i, m in enumerate(METAIS):

                    linha[
                        f"Aq {m} (mol/L)"
                    ] = caq_out[i]

                    linha[
                        f"Org {m} (mol/L)"
                    ] = corg_out[i]

                    linha[
                        f"Reextração {m} (%)"
                    ] = percentual_seguro(
                        corg_atual[i] - corg_out[i],
                        corg_atual[i]
                    )

                    linha[
                        f"Reextração acumulada {m} (%)"
                    ] = percentual_seguro(
                        corg_inicial[i] - corg_out[i],
                        corg_inicial[i]
                    )

                historico_reext.append(
                    linha
                )

                corg_atual = corg_out
                h_atual = h_out

            df_reext = pd.DataFrame(
                historico_reext
            )

            st.session_state[
                "df_reextracao"
            ] = df_reext

        if "df_reextracao" in st.session_state:

            df = st.session_state[
                "df_reextracao"
            ]

            st.subheader(
                "Resultados da reextração"
            )

            opcoes_reext = st.multiselect(
                "Informações para exibir",
                [
                    "Concentração na aquosa",
                    "Concentração na orgânica",
                    "Reextração no estágio",
                    "Reextração acumulada",
                    "H+ e pH",
                    "Extratatante livre"
                ],
                default=[
                    "Concentração na aquosa",
                    "Reextração acumulada"
                ]
            )

            metais_reext = st.multiselect(
                "Metais a visualizar",
                METAIS,
                default=METAIS
            )

            tabela = pd.DataFrame()
            tabela["Estágio"] = df["Estágio"]

            if "Concentração na aquosa" in opcoes_reext:

                for m in metais_reext:

                    tabela[
                        f"Aq {m} (mol/L)"
                    ] = df[
                        f"Aq {m} (mol/L)"
                    ]

            if "Concentração na orgânica" in opcoes_reext:

                for m in metais_reext:

                    tabela[
                        f"Org {m} (mol/L)"
                    ] = df[
                        f"Org {m} (mol/L)"
                    ]

            if "Reextração no estágio" in opcoes_reext:

                for m in metais_reext:

                    tabela[
                        f"Reextração {m} — estágio (%)"
                    ] = df[
                        f"Reextração {m} (%)"
                    ]

            if "Reextração acumulada" in opcoes_reext:

                for m in metais_reext:

                    tabela[
                        f"Reextração {m} — acumulada (%)"
                    ] = df[
                        f"Reextração acumulada {m} (%)"
                    ]

            if "H+ e pH" in opcoes_reext:

                tabela["H+ (mol/L)"] = df[
                    "H+ (mol/L)"
                ]

                tabela["pH"] = df[
                    "pH"
                ]

            if "Extratatante livre" in opcoes_reext:

                tabela[
                    "Extratatante livre dimérico (mol/L)"
                ] = df[
                    "Extratatante livre dimérico (mol/L)"
                ]

            st.dataframe(
                tabela,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # CRITÉRIO DE 100%
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "🎯 Reextração completa"
            )

            max_reext = []

            for m in metais_reext:

                coluna = (
                    f"Reextração acumulada {m} (%)"
                )

                max_reext.append(
                    df[coluna].iloc[-1]
                )

            if len(max_reext) > 0:

                menor = min(max_reext)

                if menor >= 99.999:

                    st.success(
                        "🎯 Todos os metais selecionados "
                        "foram praticamente 100% reextraídos."
                    )

                else:

                    st.warning(
                        f"A menor reextração acumulada entre "
                        f"os metais selecionados foi "
                        f"{menor:.4f}%."
                    )
