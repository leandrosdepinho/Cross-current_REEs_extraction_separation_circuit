# ============================================================
# RARE EARTH ELEMENT SOLVENT EXTRACTION SIMULATOR
# Streamlit application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="REE Solvent Extraction Simulator",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

# Order deliberately chosen by the user:
# Y is placed between Ho and Er.
METALS = [
    "La", "Ce", "Pr", "Nd",
    "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Y", "Er", "Tm", "Yb", "Lu"
]

# Kex values associated individually with each metal.
# This prevents changing the position of Y from accidentally
# changing its extraction constant.
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
    "Ho": 1.52e0,
    "Y": 8.43e-1,
    "Er": 2.66e0,
    "Tm": 4.51e0,
    "Yb": 7.45e0,
    "Lu": 1.19e1
}

# Molar masses, g/mol
MOLAR_MASS = {
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


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "extraction_history": [],
    "washing_history": [],
    "stripping_history": [],
    "extraction_feed": None,
    "combined_organic": None,
    "washing_organic_feed": None,
    "stripping_organic_feed": None,
    "extraction_sap_remaining": None,
    "extraction_started": False,
    "washing_started": False,
    "stripping_started": False,
    "stop_extraction": False,
    "stop_washing": False,
    "stop_stripping": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def concentration_to_mol(value, unit, metal):
    """Convert user input concentration to mol/L."""

    if unit == "mol/L":
        return value

    if unit == "g/L":
        return value / MOLAR_MASS[metal]

    if unit == "mg/L":
        return (value / 1000.0) / MOLAR_MASS[metal]

    return 0.0


def mol_to_unit(value, unit, metal):
    """Convert mol/L to requested concentration unit."""

    if unit == "mol/L":
        return value

    if unit == "g/L":
        return value * MOLAR_MASS[metal]

    if unit == "mg/L":
        return value * MOLAR_MASS[metal] * 1000.0

    return value


def get_groups(cut_pair):
    """
    The first member of the selected cut pair belongs to LIGHT.
    The second member and everything after it belong to HEAVY.

    Example:
        Nd/Sm -> La Ce Pr Nd = light
                   Sm ... Lu = heavy
    """

    first, second = cut_pair.split("/")

    first_index = METALS.index(first)
    second_index = METALS.index(second)

    light = METALS[:second_index]
    heavy = METALS[second_index:]

    return light, heavy


def all_cut_pairs():
    return [
        f"{METALS[i]}/{METALS[i+1]}"
        for i in range(len(METALS) - 1)
    ]


def safe_percent(num, den):
    if abs(den) < 1e-30:
        return 0.0
    return 100.0 * num / den


def total_moles(concentrations, volume):
    return np.asarray(concentrations) * volume


# ============================================================
# EQUILIBRIUM MODEL
# ============================================================

def distribution_coefficients(h, free_extractant):
    """
    D_i = Kex_i * [E_free]^3 / [H+]^3

    Extractant concentration is expressed on a monomer basis.
    """

    if h <= 0:
        h = 1e-30

    if free_extractant <= 0:
        free_extractant = 1e-30

    return np.array([
        KEX[m] * free_extractant**3 / h**3
        for m in METALS
    ])


def solve_extraction_stage(
    caq_in,
    h_in,
    extractant_total,
    saponified_capacity
):
    """
    Solve one extraction stage.

    caq_in:
        aqueous concentration entering the stage [mol/L]

    h_in:
        H+ entering the stage [mol/L]

    extractant_total:
        total extractant, expressed as monomer concentration [mol/L]

    saponified_capacity:
        remaining neutralization capacity [mol H+/L]

    The model assumes:
        REE + 3E + 3H+ equilibrium relationship

    and:
        free extractant =
            total extractant - 3 * total extracted REE

    Saponification neutralizes generated H+ until its remaining
    capacity is exhausted.
    """

    total_ree_in = np.sum(caq_in)

    # Initial guesses
    h_guess = max(h_in, 1e-8)
    e_guess = max(
        extractant_total - 0.1 * 3 * total_ree_in,
        extractant_total * 0.5
    )

    def residual(log_vars):

        # Log transformation guarantees positivity.
        h_eq = np.exp(log_vars[0])
        e_free = np.exp(log_vars[1])

        D = distribution_coefficients(h_eq, e_free)

        caq_eq = caq_in / (1.0 + D)
        corg_eq = caq_in - caq_eq

        extracted = np.sum(corg_eq)

        # Extractant mass balance, monomer basis
        e_balance = e_free - (
            extractant_total - 3.0 * extracted
        )

        # H+ generated by extraction
        h_generated = 3.0 * extracted

        # Neutralization by saponified extractant
        neutralized = min(
            h_generated,
            saponified_capacity
        )

        h_expected = (
            h_in
            + h_generated
            - neutralized
        )

        h_balance = h_eq - h_expected

        # Scale equations so one does not dominate numerically.
        scale_e = max(extractant_total, 1e-6)
        scale_h = max(h_expected, 1e-8)

        return [
            e_balance / scale_e,
            h_balance / scale_h
        ]

    result = least_squares(
        residual,
        np.log([
            h_guess,
            e_guess
        ]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=5000
    )

    h_eq = np.exp(result.x[0])
    e_free = np.exp(result.x[1])

    D = distribution_coefficients(h_eq, e_free)

    caq_eq = caq_in / (1.0 + D)
    corg_eq = caq_in - caq_eq

    extracted = np.sum(corg_eq)
    h_generated = 3.0 * extracted

    neutralized = min(
        h_generated,
        saponified_capacity
    )

    sap_remaining = max(
        0.0,
        saponified_capacity - neutralized
    )

    return {
        "caq": caq_eq,
        "corg": corg_eq,
        "h": h_eq,
        "pH": -np.log10(max(h_eq, 1e-30)),
        "free_extractant": e_free,
        "D": D,
        "extracted": extracted,
        "h_generated": h_generated,
        "neutralized": neutralized,
        "sap_remaining": sap_remaining,
    }


# ============================================================
# WASHING MODEL
# ============================================================

def solve_washing_stage(
    corg_in,
    h_wash,
    extractant_total
):
    """
    Washing stage.

    A fresh acidic aqueous solution contacts the organic phase.

    No heavy REE is intentionally added to the washing solution.

    Organic and aqueous phase volumes are both represented explicitly.
    """

    def residual(log_vars):

        h_eq = np.exp(log_vars[0])
        e_free = np.exp(log_vars[1])

        D = distribution_coefficients(h_eq, e_free)

        # For O/A = 1 in the stage equations.
        # The actual phase volumes are handled externally.
        caq_eq = corg_in / D
        corg_eq = caq_eq * D

        # Since D = Corg/Caq, the above is mathematically
        # equivalent to solving the partitioning relationship.
        # H balance:
        metal_to_aq = np.sum(corg_in - corg_eq)

        h_expected = h_wash + 3.0 * metal_to_aq

        # Extractant released when REEs leave organic.
        # The wash does not add fresh extractant.
        e_expected = extractant_total + 3.0 * (
            np.sum(corg_in) - np.sum(corg_eq)
        )

        return [
            (h_eq - h_expected) / max(h_expected, 1e-8),
            (e_free - e_expected) / max(
                extractant_total,
                1e-6
            )
        ]

    # The simplified wash is most robustly solved directly by
    # equilibrium iteration.
    h_eq = max(h_wash, 1e-10)
    e_free = max(extractant_total, 1e-10)

    for _ in range(100):

        D = distribution_coefficients(
            h_eq,
            e_free
        )

        # For O/A = 1
        caq_eq = corg_in / (1.0 + D)
        corg_eq = corg_in - caq_eq

        metal_to_aq = np.sum(
            corg_in - corg_eq
        )

        h_new = max(
            1e-12,
            h_wash + 3.0 * metal_to_aq
        )

        e_new = max(
            1e-12,
            extractant_total + 3.0 * metal_to_aq
        )

        if (
            abs(h_new - h_eq) < 1e-12
            and abs(e_new - e_free) < 1e-12
        ):
            break

        h_eq = 0.5 * h_eq + 0.5 * h_new
        e_free = 0.5 * e_free + 0.5 * e_new

    D = distribution_coefficients(
        h_eq,
        e_free
    )

    caq_eq = corg_in / (1.0 + D)
    corg_eq = corg_in - caq_eq

    return {
        "caq": caq_eq,
        "corg": corg_eq,
        "h": h_eq,
        "pH": -np.log10(max(h_eq, 1e-30)),
        "free_extractant": e_free,
        "D": D
    }


# ============================================================
# STRIPPING MODEL
# ============================================================

def solve_stripping_stage(
    corg_in,
    h_acid
):
    """
    Acid stripping.

    Fresh acidic solution contacts the organic phase.

    No new REE is introduced.

    The stripping solution is treated as a fresh aqueous phase.
    """

    h_eq = max(h_acid, 1e-12)

    # Iterative equilibrium because the released metals influence
    # the H+ balance.
    for _ in range(200):

        # During stripping the extractant concentration approaches
        # the total organic extractant concentration. We estimate
        # the free fraction from the organic loading.
        loading = np.sum(corg_in)

        e_free = max(
            1e-10,
            0.5 - 3.0 * loading
        )

        D = distribution_coefficients(
            h_eq,
            e_free
        )

        caq_eq = corg_in / (
            D + 1.0
        )

        corg_eq = corg_in - caq_eq

        stripped = np.sum(
            corg_in - corg_eq
        )

        h_new = max(
            1e-12,
            h_acid - 3.0 * stripped
        )

        if abs(h_new - h_eq) < 1e-12:
            break

        h_eq = 0.5 * h_eq + 0.5 * h_new

    D = distribution_coefficients(
        h_eq,
        e_free
    )

    caq_eq = corg_in / (D + 1.0)
    corg_eq = corg_in - caq_eq

    return {
        "caq": caq_eq,
        "corg": corg_eq,
        "h": h_eq,
        "pH": -np.log10(max(h_eq, 1e-30)),
        "free_extractant": e_free,
        "D": D
    }


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def phase_composition(conc, group):
    total = np.sum(conc)

    if total <= 1e-30:
        return 0.0

    return 100.0 * np.sum([
        conc[METALS.index(m)]
        for m in group
    ]) / total


def group_phase_distribution(
    aq,
    org,
    group
):
    aq_total = np.sum([
        aq[METALS.index(m)]
        for m in group
    ])

    org_total = np.sum([
        org[METALS.index(m)]
        for m in group
    ])

    total = aq_total + org_total

    if total <= 1e-30:
        return 0.0, 0.0

    return (
        100.0 * aq_total / total,
        100.0 * org_total / total
    )


def create_extraction_row(
    stage,
    result,
    initial_aq,
    previous_aq,
    cut_pair
):

    light, heavy = get_groups(cut_pair)

    aq = result["caq"]
    org = result["corg"]

    row = {
        "Stage": stage,
        "pH": result["pH"],
        "H+ (mol/L)": result["h"],
        "Free extractant (mol/L)": result["free_extractant"],
        "Aqueous Light (%)": phase_composition(aq, light),
        "Aqueous Heavy (%)": phase_composition(aq, heavy),
        "Organic Light (%)": phase_composition(org, light),
        "Organic Heavy (%)": phase_composition(org, heavy),
    }

    light_aq, light_org = group_phase_distribution(
        aq,
        org,
        light
    )

    heavy_aq, heavy_org = group_phase_distribution(
        aq,
        org,
        heavy
    )

    row["Light: Aqueous (%)"] = light_aq
    row["Light: Organic (%)"] = light_org
    row["Heavy: Aqueous (%)"] = heavy_aq
    row["Heavy: Organic (%)"] = heavy_org

    for i, metal in enumerate(METALS):

        stage_extraction = safe_percent(
            previous_aq[i] - aq[i],
            previous_aq[i]
        )

        cumulative_extraction = safe_percent(
            initial_aq[i] - aq[i],
            initial_aq[i]
        )

        row[f"{metal} Aq (mol/L)"] = aq[i]
        row[f"{metal} Org (mol/L)"] = org[i]

        row[f"{metal} Stage Extraction (%)"] = stage_extraction
        row[f"{metal} Cumulative Extraction (%)"] = cumulative_extraction

    return row


# ============================================================
# GRAPH FUNCTIONS
# ============================================================

def plot_line_chart(
    df,
    columns,
    title,
    ylabel
):

    fig, ax = plt.subplots(figsize=(10, 5))

    for col in columns:
        ax.plot(
            df["Stage"],
            df[col],
            marker="o",
            label=col
        )

    ax.set_title(title)
    ax.set_xlabel("Stage")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()

    return fig


def plot_group_distribution(
    df,
    aq_column,
    org_column,
    title
):

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))

    aq = df[aq_column].values
    org = df[org_column].values

    ax.bar(
        x,
        aq,
        label="Aqueous"
    )

    ax.bar(
        x,
        org,
        bottom=aq,
        label="Organic"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Stage"].astype(str)
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Distribution (%)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    return fig


def plot_phase_composition(
    df,
    light_col,
    heavy_col,
    title
):

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))

    light = df[light_col].values
    heavy = df[heavy_col].values

    ax.bar(
        x,
        light,
        label="Light REEs"
    )

    ax.bar(
        x,
        heavy,
        bottom=light,
        label="Heavy REEs"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Stage"].astype(str)
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Molar composition (%)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    return fig


def plot_elemental_composition(
    df,
    phase,
    title
):

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(df))

    bottom = np.zeros(len(df))

    for metal in METALS:

        col = f"{metal} {phase} (mol/L)"

        if col not in df.columns:
            continue

        values = df[col].values

        # Convert concentration to relative composition
        totals = np.zeros(len(df))

        for m in METALS:
            c = f"{m} {phase} (mol/L)"
            if c in df.columns:
                totals += df[c].values

        percentages = np.divide(
            values,
            totals,
            out=np.zeros_like(values),
            where=totals > 0
        ) * 100.0

        ax.bar(
            x,
            percentages,
            bottom=bottom,
            label=metal
        )

        bottom += percentages

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Stage"].astype(str)
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Molar composition (%)")
    ax.set_title(title)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=2
    )

    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Simulation")

page = st.sidebar.radio(
    "Process",
    [
        "1 — Extraction",
        "2 — Washing",
        "3 — Re-extraction"
    ]
)


# ============================================================
# EXTRACTION PAGE
# ============================================================

if page == "1 — Extraction":

    st.title("REE Solvent Extraction")
    st.caption(
        "Counter-current separation is represented as sequential "
        "equilibrium contacts."
    )

    st.header("Feed composition")

    col1, col2, col3 = st.columns(3)

    with col1:
        concentration_unit = st.selectbox(
            "Concentration unit",
            [
                "mg/L",
                "g/L",
                "mol/L"
            ]
        )

    with col2:
        feed_volume = st.number_input(
            "Aqueous feed volume (L)",
            min_value=0.001,
            value=1.0,
            step=0.1
        )

    with col3:
        initial_pH = st.number_input(
            "Initial pH",
            min_value=-2.0,
            max_value=14.0,
            value=1.0,
            step=0.1
        )

    st.subheader("REE concentrations")

    feed_values = {}

    cols = st.columns(5)

    for i, metal in enumerate(METALS):

        with cols[i % 5]:

            feed_values[metal] = st.number_input(
                f"{metal} ({concentration_unit})",
                min_value=0.0,
                value=0.0,
                format="%.8g",
                key=f"feed_{metal}"
            )

    st.divider()

    st.header("Extraction parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cut_pair = st.selectbox(
            "REE cut",
            all_cut_pairs(),
            index=METALS.index("Nd")
            if "Nd" in METALS else 3
        )

    with col2:
        oa_ratio = st.number_input(
            "O/A ratio",
            min_value=0.001,
            value=1.0,
            step=0.1
        )

    with col3:
        extractant_total = st.number_input(
            "Total extractant concentration "
            "(mol/L, monomer basis)",
            min_value=0.000001,
            value=0.5,
            step=0.05
        )

    with col4:
        saponification = st.number_input(
            "Saponification (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=5.0
        )

    target_purity = st.number_input(
        "Target aqueous light-REE purity (%)",
        min_value=0.0,
        max_value=100.0,
        value=99.9,
        step=0.1
    )

    light, heavy = get_groups(cut_pair)

    st.info(
        f"Cut: **{cut_pair}**  |  "
        f"Light REEs: **{', '.join(light)}**  |  "
        f"Heavy REEs: **{', '.join(heavy)}**"
    )

    # --------------------------------------------------------
    # Initialize feed
    # --------------------------------------------------------

    if st.button(
        "Start / Reset Extraction",
        type="primary"
    ):

        initial_aq = np.array([
            concentration_to_mol(
                feed_values[m],
                concentration_unit,
                m
            )
            for m in METALS
        ])

        st.session_state.extraction_feed = initial_aq.copy()
        st.session_state.extraction_history = []
        st.session_state.extraction_sap_remaining = (
            extractant_total * saponification / 100.0
        )
        st.session_state.extraction_started = True
        st.session_state.stop_extraction = False

        st.rerun()

    # --------------------------------------------------------
    # Extraction stage calculation
    # --------------------------------------------------------

    if st.session_state.extraction_started:

        initial_aq = st.session_state.extraction_feed

        if len(st.session_state.extraction_history) == 0:

            current_aq = initial_aq.copy()
            h_in = 10 ** (-initial_pH)

        else:

            last = st.session_state.extraction_history[-1]

            current_aq = np.array([
                last[f"{m} Aq (mol/L)"]
                for m in METALS
            ])

            h_in = last["H+ (mol/L)"]

        if st.button(
            "Add extraction stage",
            type="secondary",
            disabled=st.session_state.stop_extraction
        ):

            result = solve_extraction_stage(
                current_aq,
                h_in,
                extractant_total,
                st.session_state.extraction_sap_remaining
            )

            stage = len(
                st.session_state.extraction_history
            ) + 1

            previous_aq = current_aq.copy()

            row = create_extraction_row(
                stage,
                result,
                initial_aq,
                previous_aq,
                cut_pair
            )

            st.session_state.extraction_history.append(
                row
            )

            st.session_state.extraction_sap_remaining = (
                result["sap_remaining"]
            )

            st.rerun()

        if st.session_state.extraction_history:

            df_ext = pd.DataFrame(
                st.session_state.extraction_history
            )

            st.subheader("Extraction results")

            st.dataframe(
                df_ext,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Extraction graphs")

            selected_metals = st.multiselect(
                "Metals to display",
                METALS,
                default=["Nd", "Sm"]
            )

            if selected_metals:

                c1, c2 = st.columns(2)

                with c1:

                    cols = [
                        f"{m} Cumulative Extraction (%)"
                        for m in selected_metals
                    ]

                    st.pyplot(
                        plot_line_chart(
                            df_ext,
                            cols,
                            "Cumulative extraction vs. stage",
                            "Cumulative extraction (%)"
                        ),
                        clear_figure=True
                    )

                with c2:

                    cols = [
                        f"{m} Stage Extraction (%)"
                        for m in selected_metals
                    ]

                    st.pyplot(
                        plot_line_chart(
                            df_ext,
                            cols,
                            "Stage extraction vs. stage",
                            "Stage extraction (%)"
                        ),
                        clear_figure=True
                    )

            c1, c2 = st.columns(2)

            with c1:

                st.pyplot(
                    plot_phase_composition(
                        df_ext,
                        "Aqueous Light (%)",
                        "Aqueous Heavy (%)",
                        "Aqueous phase composition"
                    ),
                    clear_figure=True
                )

            with c2:

                st.pyplot(
                    plot_phase_composition(
                        df_ext,
                        "Organic Light (%)",
                        "Organic Heavy (%)",
                        "Organic phase composition"
                    ),
                    clear_figure=True
                )

            c1, c2 = st.columns(2)

            with c1:

                st.pyplot(
                    plot_group_distribution(
                        df_ext,
                        "Light: Aqueous (%)",
                        "Light: Organic (%)",
                        "Light REE distribution between phases"
                    ),
                    clear_figure=True
                )

            with c2:

                st.pyplot(
                    plot_group_distribution(
                        df_ext,
                        "Heavy: Aqueous (%)",
                        "Heavy: Organic (%)",
                        "Heavy REE distribution between phases"
                    ),
                    clear_figure=True
                )

            st.subheader(
                "Elemental molar composition"
            )

            phase = st.radio(
                "Phase",
                ["Aqueous", "Organic"],
                horizontal=True,
                key="ext_element_phase"
            )

            st.pyplot(
                plot_elemental_composition(
                    df_ext,
                    phase,
                    f"{phase} elemental molar composition"
                ),
                clear_figure=True
            )

            # Current purity
            current_purity = df_ext.iloc[-1][
                "Aqueous Light (%)"
            ]

            if current_purity >= target_purity:

                st.success(
                    f"Target purity reached: "
                    f"{current_purity:.4f}% light REEs "
                    f"in the aqueous phase."
                )

            else:

                st.warning(
                    f"Current aqueous light-REE purity: "
                    f"{current_purity:.4f}%  |  "
                    f"Target: {target_purity:.4f}%"
                )

            if st.button(
                "Stop extraction and continue to washing",
                type="primary"
            ):

                # Build combined organic phase.
                #
                # Each extraction stage represents the same
                # organic volume multiplied according to O/A.
                #
                # For feed volume Vaq:
                # Vorg = Vaq / (O/A)
                #
                # All organic streams are then combined.

                organic_moles = np.zeros(len(METALS))
                total_organic_volume = 0.0

                stage_organic_volume = (
                    feed_volume / oa_ratio
                )

                for row in st.session_state.extraction_history:

                    c_org = np.array([
                        row[f"{m} Org (mol/L)"]
                        for m in METALS
                    ])

                    organic_moles += (
                        c_org * stage_organic_volume
                    )

                    total_organic_volume += (
                        stage_organic_volume
                    )

                combined_organic = (
                    organic_moles /
                    max(total_organic_volume, 1e-30)
                )

                st.session_state.combined_organic = {
                    "concentration": combined_organic,
                    "volume": total_organic_volume,
                    "moles": organic_moles
                }

                st.session_state.stop_extraction = True

                st.success(
                    "Extraction stopped. "
                    "The organic streams have been combined "
                    "and are ready for washing."
                )


# ============================================================
# WASHING PAGE
# ============================================================

elif page == "2 — Washing":

    st.title("Organic Phase Washing")

    if st.session_state.combined_organic is None:

        st.warning(
            "Complete and stop the extraction first."
        )

        st.stop()

    combined = st.session_state.combined_organic

    st.info(
        f"Combined organic phase: "
        f"**{combined['volume']:.4f} L**"
    )

    light, heavy = get_groups(
        st.session_state.get(
            "cut_pair",
            all_cut_pairs()[3]
        )
    )

    st.header("Washing parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        wash_pH = st.number_input(
            "Washing solution pH",
            min_value=-2.0,
            max_value=7.0,
            value=1.0,
            step=0.1
        )

    with col2:

        wash_ao = st.number_input(
            "Washing O/A ratio",
            min_value=0.001,
            value=1.0,
            step=0.1
        )

    with col3:

        washing_extractant = st.number_input(
            "Organic extractant concentration "
            "(mol/L, monomer basis)",
            min_value=0.000001,
            value=0.5,
            step=0.05
        )

    st.subheader("Current combined organic feed")

    feed_df = pd.DataFrame({
        "REE": METALS,
        "Organic concentration (mol/L)": combined[
            "concentration"
        ],
        "Organic amount (mol)": combined[
            "moles"
        ]
    })

    st.dataframe(
        feed_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Start / Reset Washing",
        type="primary"
    ):

        st.session_state.washing_history = []
        st.session_state.washing_started = True
        st.session_state.stop_washing = False
        st.session_state.washing_organic_feed = (
            combined["concentration"].copy()
        )

        st.rerun()

    if st.session_state.washing_started:

        if st.button(
            "Add washing stage",
            disabled=st.session_state.stop_washing
        ):

            corg_in = (
                st.session_state.washing_organic_feed
            )

            result = solve_washing_stage(
                corg_in,
                10 ** (-wash_pH),
                washing_extractant
            )

            stage = len(
                st.session_state.washing_history
            ) + 1

            row = {
                "Stage": stage,
                "pH": result["pH"],
                "H+ (mol/L)": result["h"],
                "Free extractant (mol/L)": result[
                    "free_extractant"
                ]
            }

            aq = result["caq"]
            org = result["corg"]

            initial = combined["concentration"]

            for i, metal in enumerate(METALS):

                removed_stage = safe_percent(
                    corg_in[i] - org[i],
                    corg_in[i]
                )

                removed_cumulative = safe_percent(
                    initial[i] - org[i],
                    initial[i]
                )

                remaining = safe_percent(
                    org[i],
                    initial[i]
                )

                row[
                    f"{metal} Aq (mol/L)"
                ] = aq[i]

                row[
                    f"{metal} Org (mol/L)"
                ] = org[i]

                row[
                    f"{metal} Stage Removed (%)"
                ] = removed_stage

                row[
                    f"{metal} Cumulative Removed (%)"
                ] = removed_cumulative

                row[
                    f"{metal} Remaining in Organic (%)"
                ] = remaining

            row["Organic Light (%)"] = phase_composition(
                org,
                light
            )

            row["Organic Heavy (%)"] = phase_composition(
                org,
                heavy
            )

            row["Aqueous Light (%)"] = phase_composition(
                aq,
                light
            )

            row["Aqueous Heavy (%)"] = phase_composition(
                aq,
                heavy
            )

            light_aq, light_org = group_phase_distribution(
                aq,
                org,
                light
            )

            heavy_aq, heavy_org = group_phase_distribution(
                aq,
                org,
                heavy
            )

            row["Light: Aqueous (%)"] = light_aq
            row["Light: Organic (%)"] = light_org
            row["Heavy: Aqueous (%)"] = heavy_aq
            row["Heavy: Organic (%)"] = heavy_org

            st.session_state.washing_history.append(row)

            st.session_state.washing_organic_feed = (
                org.copy()
            )

            st.rerun()

        if st.session_state.washing_history:

            df_wash = pd.DataFrame(
                st.session_state.washing_history
            )

            st.subheader("Washing results")

            st.dataframe(
                df_wash,
                use_container_width=True,
                hide_index=True
            )

            selected_metals = st.multiselect(
                "Metals to display",
                METALS,
                default=["Nd", "Sm"],
                key="wash_metals"
            )

            if selected_metals:

                cols = [
                    f"{m} Cumulative Removed (%)"
                    for m in selected_metals
                ]

                st.pyplot(
                    plot_line_chart(
                        df_wash,
                        cols,
                        "Cumulative REE removal during washing",
                        "Cumulative removed (%)"
                    ),
                    clear_figure=True
                )

            c1, c2 = st.columns(2)

            with c1:

                st.pyplot(
                    plot_phase_composition(
                        df_wash,
                        "Organic Light (%)",
                        "Organic Heavy (%)",
                        "Organic Light / Heavy composition"
                    ),
                    clear_figure=True
                )

            with c2:

                st.pyplot(
                    plot_phase_composition(
                        df_wash,
                        "Aqueous Light (%)",
                        "Aqueous Heavy (%)",
                        "Wash aqueous Light / Heavy composition"
                    ),
                    clear_figure=True
                )

            c1, c2 = st.columns(2)

            with c1:

                st.pyplot(
                    plot_group_distribution(
                        df_wash,
                        "Light: Aqueous (%)",
                        "Light: Organic (%)",
                        "Light REE distribution during washing"
                    ),
                    clear_figure=True
                )

            with c2:

                st.pyplot(
                    plot_group_distribution(
                        df_wash,
                        "Heavy: Aqueous (%)",
                        "Heavy: Organic (%)",
                        "Heavy REE distribution during washing"
                    ),
                    clear_figure=True
                )

            st.subheader(
                "Elemental composition of the washed organic"
            )

            st.pyplot(
                plot_elemental_composition(
                    df_wash,
                    "Organic",
                    "Washed organic elemental composition"
                ),
                clear_figure=True
            )

            if st.button(
                "Stop washing and continue to re-extraction",
                type="primary"
            ):

                final_org = (
                    st.session_state.washing_organic_feed
                )

                # Preserve the actual moles in the organic phase.
                final_moles = (
                    final_org * combined["volume"]
                )

                st.session_state.stripping_organic_feed = {
                    "concentration": final_org.copy(),
                    "volume": combined["volume"],
                    "moles": final_moles
                }

                st.session_state.stop_washing = True

                st.success(
                    "Washing stopped. The remaining organic "
                    "phase is ready for re-extraction."
                )


# ============================================================
# RE-EXTRACTION PAGE
# ============================================================

elif page == "3 — Re-extraction":

    st.title("REE Re-extraction / Stripping")

    if st.session_state.stripping_organic_feed is None:

        st.warning(
            "Complete and stop the washing first."
        )

        st.stop()

    feed = st.session_state.stripping_organic_feed

    st.info(
        f"Organic phase entering stripping: "
        f"**{feed['volume']:.4f} L**"
    )

    st.header("Stripping parameters")

    col1, col2 = st.columns(2)

    with col1:

        stripping_H = st.number_input(
            "Stripping solution H+ concentration (mol/L)",
            min_value=0.000001,
            value=2.0,
            step=0.1
        )

    with col2:

        stripping_ao = st.number_input(
            "Stripping O/A ratio",
            min_value=0.001,
            value=1.0,
            step=0.1
        )

    if st.button(
        "Start / Reset Re-extraction",
        type="primary"
    ):

        st.session_state.stripping_history = []
        st.session_state.stripping_started = True
        st.session_state.stop_stripping = False

        st.session_state.stripping_organic_feed = {
            **feed,
            "current_concentration":
                feed["concentration"].copy()
        }

        st.rerun()

    if st.session_state.stripping_started:

        if st.button(
            "Add re-extraction stage",
            disabled=st.session_state.stop_stripping
        ):

            current_org = (
                st.session_state
                .stripping_organic_feed[
                    "current_concentration"
                ]
            )

            result = solve_stripping_stage(
                current_org,
                stripping_H
            )

            stage = len(
                st.session_state.stripping_history
            ) + 1

            row = {
                "Stage": stage,
                "pH": result["pH"],
                "H+ (mol/L)": result["h"],
                "Free extractant (mol/L)": result[
                    "free_extractant"
                ]
            }

            aq = result["caq"]
            org = result["corg"]

            initial = feed["concentration"]

            for i, metal in enumerate(METALS):

                stage_reextraction = safe_percent(
                    current_org[i] - org[i],
                    current_org[i]
                )

                cumulative_reextraction = safe_percent(
                    initial[i] - org[i],
                    initial[i]
                )

                remaining = safe_percent(
                    org[i],
                    initial[i]
                )

                row[
                    f"{metal} Aq (mol/L)"
                ] = aq[i]

                row[
                    f"{metal} Org (mol/L)"
                ] = org[i]

                row[
                    f"{metal} Stage Re-extraction (%)"
                ] = stage_reextraction

                row[
                    f"{metal} Cumulative Re-extraction (%)"
                ] = cumulative_reextraction

                row[
                    f"{metal} Remaining in Organic (%)"
                ] = remaining

            st.session_state.stripping_history.append(
                row
            )

            st.session_state.stripping_organic_feed[
                "current_concentration"
            ] = org.copy()

            st.rerun()

        if st.session_state.stripping_history:

            df_strip = pd.DataFrame(
                st.session_state.stripping_history
            )

            st.subheader("Re-extraction results")

            st.dataframe(
                df_strip,
                use_container_width=True,
                hide_index=True
            )

            selected_metals = st.multiselect(
                "Metals to display",
                METALS,
                default=["Nd", "Sm"],
                key="strip_metals"
            )

            if selected_metals:

                c1, c2 = st.columns(2)

                with c1:

                    cols = [
                        f"{m} Cumulative Re-extraction (%)"
                        for m in selected_metals
                    ]

                    st.pyplot(
                        plot_line_chart(
                            df_strip,
                            cols,
                            "Cumulative re-extraction vs. stage",
                            "Cumulative re-extraction (%)"
                        ),
                        clear_figure=True
                    )

                with c2:

                    cols = [
                        f"{m} Stage Re-extraction (%)"
                        for m in selected_metals
                    ]

                    st.pyplot(
                        plot_line_chart(
                            df_strip,
                            cols,
                            "Stage re-extraction vs. stage",
                            "Stage re-extraction (%)"
                        ),
                        clear_figure=True
                    )

            c1, c2 = st.columns(2)

            with c1:

                st.pyplot(
                    plot_group_distribution(
                        df_strip,
                        "Light: Aqueous (%)"
                        if "Light: Aqueous (%)"
                        in df_strip.columns
                        else f"{selected_metals[0]} Aq (mol/L)",
                        "Light: Organic (%)"
                        if "Light: Organic (%)"
                        in df_strip.columns
                        else f"{selected_metals[0]} Org (mol/L)",
                        "Re-extraction distribution"
                    ),
                    clear_figure=True
                )

            st.subheader(
                "Remaining organic phase"
            )

            st.pyplot(
                plot_elemental_composition(
                    df_strip,
                    "Org",
                    "Remaining organic elemental composition"
                ),
                clear_figure=True
            )

            # 100% target check
            final_values = []

            for metal in METALS:

                final_values.append(
                    df_strip.iloc[-1][
                        f"{metal} Cumulative Re-extraction (%)"
                    ]
                )

            minimum_recovery = min(final_values)

            if minimum_recovery >= 99.999:

                st.success(
                    "All REEs have effectively reached "
                    "complete re-extraction."
                )

            else:

                st.warning(
                    f"Minimum cumulative re-extraction: "
                    f"{minimum_recovery:.4f}%"
                )

            if st.button(
                "Stop re-extraction",
                type="primary"
            ):

                st.session_state.stop_stripping = True

                st.success(
                    "Re-extraction stopped."
                )
