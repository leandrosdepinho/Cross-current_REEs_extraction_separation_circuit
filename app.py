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
    page_title="Cross-current REEs extraction separation circuit",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

METALS = [
    "La", "Ce", "Pr", "Nd",
    "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Y", "Er", "Tm", "Yb", "Lu"
]

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

    "extraction_started": False,
    "washing_started": False,
    "stripping_started": False,

    "stop_extraction": False,
    "stop_washing": False,
    "stop_stripping": False,

    "cut_pair": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def concentration_to_mol(value, unit, metal):

    if unit == "mol/L":
        return value

    if unit == "g/L":
        return value / MOLAR_MASS[metal]

    if unit == "mg/L":
        return (value / 1000.0) / MOLAR_MASS[metal]

    return 0.0


def mol_to_unit(value, unit, metal):

    if unit == "mol/L":
        return value

    if unit == "g/L":
        return value * MOLAR_MASS[metal]

    if unit == "mg/L":
        return value * MOLAR_MASS[metal] * 1000.0

    return value


def get_groups(cut_pair):

    first, second = cut_pair.split("/")

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

    h = max(h, 1e-30)
    free_extractant = max(free_extractant, 1e-30)

    return np.array([
        KEX[m] * free_extractant**3 / h**3
        for m in METALS
    ])


# ============================================================
# EXTRACTION
# ============================================================

def solve_extraction_stage(
    caq_in,
    h_in,
    extractant_total,
    saponified_capacity,
    oa_ratio
):

    # R = V_org / V_aq for this contact. Needed because the metal,
    # extractant and H+ balances live in different phases/volumes;
    # without R the code implicitly (and wrongly) assumed O/A = 1.
    R = max(oa_ratio, 1e-12)

    h_guess = max(h_in, 1e-8)

    e_guess = max(
        extractant_total - 0.3 * np.sum(caq_in),
        extractant_total * 0.5
    )

    def residual(log_vars):

        h_eq = np.exp(log_vars[0])
        e_free = np.exp(log_vars[1])

        D = distribution_coefficients(
            h_eq,
            e_free
        )

        # Mass balance across unequal phase volumes:
        # C_aq_in * V_aq = C_aq_eq * V_aq + C_org_eq * V_org
        caq_eq = caq_in / (
            1.0 + D * R
        )

        corg_eq = D * caq_eq

        extracted = np.sum(corg_eq)   # mol per L organic

        e_expected = (
            extractant_total
            - 3.0 * extracted
        )

        h_generated = (
            3.0 * extracted * R
        )

        neutralized = min(
            h_generated,
            saponified_capacity * R
        )

        h_expected = (
            h_in
            + h_generated
            - neutralized
        )

        scale_e = max(
            extractant_total,
            1e-6
        )

        scale_h = max(
            h_expected,
            1e-8
        )

        return [
            (
                e_free - e_expected
            ) / scale_e,

            (
                h_eq - h_expected
            ) / scale_h
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

    D = distribution_coefficients(
        h_eq,
        e_free
    )

    caq_eq = caq_in / (
        1.0 + D * R
    )

    corg_eq = D * caq_eq

    extracted = np.sum(corg_eq)

    h_generated = (
        3.0 * extracted * R
    )

    neutralized = min(
        h_generated,
        saponified_capacity * R
    )

    sap_remaining = max(
        0.0,
        saponified_capacity - neutralized / R
    )

    return {
        "caq": caq_eq,
        "corg": corg_eq,
        "h": h_eq,
        "pH": -np.log10(
            max(h_eq, 1e-30)
        ),
        "free_extractant": e_free,
        "D": D,
        "extracted": extracted,
        "h_generated": h_generated,
        "neutralized": neutralized,
        "sap_remaining": sap_remaining
    }


# ============================================================
# WASHING
# ============================================================

def solve_washing_stage(
    corg_in,
    h_wash,
    extractant_in,
    oa_ratio
):

    # R = V_org / V_aq for the wash contact.
    R = max(oa_ratio, 1e-12)

    h_guess = max(h_wash, 1e-10)
    e_guess = max(extractant_in, 1e-10)

    def residual(log_vars):

        h_eq = np.exp(log_vars[0])
        e_free = np.exp(log_vars[1])

        D = distribution_coefficients(
            h_eq,
            e_free
        )

        # Mass balance: C_org_in * V_org = C_aq_eq * V_aq + C_org_eq * V_org
        caq_eq = corg_in * R / (
            1.0 + D * R
        )

        corg_eq = D * caq_eq

        metal_to_aq = np.sum(
            corg_in - corg_eq
        )   # mol per L organic

        e_expected = (
            extractant_in
            + 3.0 * metal_to_aq
        )

        h_expected = max(
            1e-12,
            h_wash - 3.0 * metal_to_aq * R
        )

        scale_e = max(extractant_in, 1e-6)
        scale_h = max(h_expected, 1e-12)

        return [
            (e_free - e_expected) / scale_e,
            (h_eq - h_expected) / scale_h
        ]

    result = least_squares(
        residual,
        np.log([h_guess, e_guess]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=5000
    )

    h_eq = np.exp(result.x[0])
    e_free = np.exp(result.x[1])

    D = distribution_coefficients(
        h_eq,
        e_free
    )

    caq_eq = corg_in * R / (
        1.0 + D * R
    )

    corg_eq = D * caq_eq

    return {
        "caq": caq_eq,
        "corg": corg_eq,
        "h": h_eq,
        "pH": -np.log10(
            max(h_eq, 1e-30)
        ),
        "free_extractant": e_free,
        "D": D
    }


# ============================================================
# RE-EXTRACTION / STRIPPING
# ============================================================

def solve_stripping_stage(
    corg_in,
    h_acid,
    extractant_in,
    oa_ratio
):

    # R = V_org / V_aq for the stripping contact.
    R = max(oa_ratio, 1e-12)

    h_guess = max(h_acid, 1e-12)
    e_guess = max(extractant_in, 1e-12)

    def residual(log_vars):

        h_eq = np.exp(log_vars[0])
        e_free = np.exp(log_vars[1])

        D = distribution_coefficients(
            h_eq,
            e_free
        )

        caq_eq = corg_in * R / (
            1.0 + D * R
        )

        corg_eq = D * caq_eq

        stripped = np.sum(
            corg_in - corg_eq
        )   # mol per L organic

        e_expected = (
            extractant_in
            + 3.0 * stripped
        )

        h_expected = max(
            1e-12,
            h_acid - 3.0 * stripped * R
        )

        scale_e = max(extractant_in, 1e-6)
        scale_h = max(h_expected, 1e-12)

        return [
            (e_free - e_expected) / scale_e,
            (h_eq - h_expected) / scale_h
        ]

    result = least_squares(
        residual,
        np.log([h_guess, e_guess]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=5000
    )

    h_eq = np.exp(result.x[0])
    e_free = np.exp(result.x[1])

    D = distribution_coefficients(
        h_eq,
        e_free
    )

    caq_f = corg_in * R / (
        1.0 + D * R
    )

    corg_f = D * caq_f

    return {
        "caq": caq_f,
        "corg": corg_f,
        "h": h_eq,
        "pH": -np.log10(
            max(h_eq, 1e-30)
        ),
        "free_extractant": e_free,
        "D": D
    }


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def phase_composition(
    conc,
    group
):

    total = np.sum(conc)

    if total <= 1e-30:
        return 0.0

    group_total = sum(
        conc[METALS.index(m)]
        for m in group
    )

    return 100.0 * group_total / total


def group_phase_distribution(
    aq,
    org,
    group
):

    aq_total = sum(
        aq[METALS.index(m)]
        for m in group
    )

    org_total = sum(
        org[METALS.index(m)]
        for m in group
    )

    total = (
        aq_total
        + org_total
    )

    if total <= 1e-30:
        return 0.0, 0.0

    return (
        100.0 * aq_total / total,
        100.0 * org_total / total
    )


# ============================================================
# EXTRACTION TABLE ROW
# ============================================================

def create_extraction_row(
    stage,
    result,
    initial_aq,
    previous_aq,
    cut_pair
):

    light, heavy = get_groups(
        cut_pair
    )

    aq = result["caq"]
    org = result["corg"]

    row = {
        "Stage": stage,

        "pH": result["pH"],

        "H+ (mol/L)":
            result["h"],

        "Free extractant (mol/L)":
            result["free_extractant"],

        "Aqueous Light (%)":
            phase_composition(
                aq,
                light
            ),

        "Aqueous Heavy (%)":
            phase_composition(
                aq,
                heavy
            ),

        "Organic Light (%)":
            phase_composition(
                org,
                light
            ),

        "Organic Heavy (%)":
            phase_composition(
                org,
                heavy
            )
    }

    light_aq, light_org = (
        group_phase_distribution(
            aq,
            org,
            light
        )
    )

    heavy_aq, heavy_org = (
        group_phase_distribution(
            aq,
            org,
            heavy
        )
    )

    row[
        "Light: Aqueous (%)"
    ] = light_aq

    row[
        "Light: Organic (%)"
    ] = light_org

    row[
        "Heavy: Aqueous (%)"
    ] = heavy_aq

    row[
        "Heavy: Organic (%)"
    ] = heavy_org

    for i, metal in enumerate(
        METALS
    ):

        stage_extraction = safe_percent(
            previous_aq[i] - aq[i],
            previous_aq[i]
        )

        cumulative_extraction = safe_percent(
            initial_aq[i] - aq[i],
            initial_aq[i]
        )

        remaining = (
            100.0
            - cumulative_extraction
        )

        row[
            f"{metal} Aq (mol/L)"
        ] = aq[i]

        row[
            f"{metal} Org (mol/L)"
        ] = org[i]

        row[
            f"{metal} Stage Extraction (%)"
        ] = stage_extraction

        row[
            f"{metal} Cumulative Extraction (%)"
        ] = cumulative_extraction

        row[
            f"{metal} Remaining in Aqueous (%)"
        ] = remaining

    # ========================================================
    # GROUP EXTRACTION METRICS
    # ========================================================

    initial_light = sum(
        initial_aq[
            METALS.index(m)
        ]
        for m in light
    )

    previous_light = sum(
        previous_aq[
            METALS.index(m)
        ]
        for m in light
    )

    current_light = sum(
        aq[
            METALS.index(m)
        ]
        for m in light
    )

    initial_heavy = sum(
        initial_aq[
            METALS.index(m)
        ]
        for m in heavy
    )

    previous_heavy = sum(
        previous_aq[
            METALS.index(m)
        ]
        for m in heavy
    )

    current_heavy = sum(
        aq[
            METALS.index(m)
        ]
        for m in heavy
    )

    row[
        "Light Stage Extraction (%)"
    ] = safe_percent(
        previous_light - current_light,
        previous_light
    )

    row[
        "Light Cumulative Extraction (%)"
    ] = safe_percent(
        initial_light - current_light,
        initial_light
    )

    row[
        "Light Remaining in Aqueous (%)"
    ] = (
        100.0
        - row[
            "Light Cumulative Extraction (%)"
        ]
    )

    row[
        "Heavy Stage Extraction (%)"
    ] = safe_percent(
        previous_heavy - current_heavy,
        previous_heavy
    )

    row[
        "Heavy Cumulative Extraction (%)"
    ] = safe_percent(
        initial_heavy - current_heavy,
        initial_heavy
    )

    row[
        "Heavy Remaining in Aqueous (%)"
    ] = (
        100.0
        - row[
            "Heavy Cumulative Extraction (%)"
        ]
    )

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

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for col in columns:

        if col in df.columns:

            ax.plot(
                df["Stage"],
                df[col],
                marker="o",
                label=col
            )

    ax.set_title(title)
    ax.set_xlabel("Stage")
    ax.set_ylabel(ylabel)

    ax.grid(
        alpha=0.25
    )

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

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x = np.arange(
        len(df)
    )

    aq = df[
        aq_column
    ].values

    org = df[
        org_column
    ].values

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

    ax.set_ylim(
        0,
        100
    )

    ax.set_xlabel("Stage")
    ax.set_ylabel(
        "Distribution (%)"
    )

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

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x = np.arange(
        len(df)
    )

    light = df[
        light_col
    ].values

    heavy = df[
        heavy_col
    ].values

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

    ax.set_ylim(
        0,
        100
    )

    ax.set_xlabel("Stage")

    ax.set_ylabel(
        "Molar composition (%)"
    )

    ax.set_title(title)

    ax.legend()

    fig.tight_layout()

    return fig


# ============================================================
# ELEMENTAL COMPOSITION
# ============================================================

def plot_elemental_composition(
    df,
    phase_code,
    title
):

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    x = np.arange(
        len(df)
    )

    cmap = plt.get_cmap(
        "tab20"
    )

    bottom = np.zeros(
        len(df)
    )

    totals = np.zeros(
        len(df)
    )

    phase_columns = {}

    # IMPORTANT:
    # Extraction uses Aq / Org.
    # Washing and stripping also use Aq / Org.
    for metal in METALS:

        col = (
            f"{metal} "
            f"{phase_code} "
            f"(mol/L)"
        )

        if col in df.columns:

            values = (
                df[col]
                .astype(float)
                .values
            )

            phase_columns[
                metal
            ] = values

            totals += values

    for i, metal in enumerate(
        METALS
    ):

        if metal not in phase_columns:
            continue

        values = phase_columns[
            metal
        ]

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
            label=metal,
            color=cmap(i)
        )

        bottom += percentages

    ax.set_xticks(x)

    ax.set_xticklabels(
        df["Stage"].astype(str)
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_xlabel("Stage")

    ax.set_ylabel(
        "Molar composition (%)"
    )

    ax.set_title(title)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=2
    )

    fig.tight_layout()

    return fig


# ============================================================
# TABLE DISPLAY CONTROLS
# ============================================================

def select_table_columns(
    df,
    process
):

    st.subheader(
        "Table display options"
    )

    base_columns = [
        "Stage",
        "pH",
        "H+ (mol/L)",
        "Free extractant (mol/L)"
    ]

    selected = []

    selected.extend(
        st.multiselect(
            "General process information",
            [
                c for c in base_columns
                if c in df.columns
            ],
            default=base_columns,
            key=f"{process}_general_columns"
        )
    )

    if process == "Extraction":

        concentration_columns = [
            c for c in df.columns
            if (
                "Aq (mol/L)" in c
                or "Org (mol/L)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Metal concentrations",
                concentration_columns,
                default=concentration_columns,
                key="ext_concentration_columns"
            )
        )

        individual_columns = [
            c for c in df.columns
            if (
                "Stage Extraction (%)" in c
                or "Cumulative Extraction (%)" in c
                or "Remaining in Aqueous (%)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Individual-metal extraction",
                individual_columns,
                default=[],
                key="ext_individual_columns"
            )
        )

        group_columns = [
            c for c in df.columns
            if (
                c.startswith("Light ")
                or c.startswith("Heavy ")
            )
        ]

        selected.extend(
            st.multiselect(
                "Light / Heavy group extraction",
                group_columns,
                default=group_columns,
                key="ext_group_columns"
            )
        )

        composition_columns = [
            c for c in df.columns
            if (
                "Aqueous Light (%)" in c
                or "Aqueous Heavy (%)" in c
                or "Organic Light (%)" in c
                or "Organic Heavy (%)" in c
                or "Light: Aqueous (%)" in c
                or "Light: Organic (%)" in c
                or "Heavy: Aqueous (%)" in c
                or "Heavy: Organic (%)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Molar composition and phase distribution",
                composition_columns,
                default=composition_columns,
                key="ext_composition_columns"
            )
        )

    elif process == "Washing":

        concentration_columns = [
            c for c in df.columns
            if (
                "Aq (mol/L)" in c
                or "Org (mol/L)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Metal concentrations",
                concentration_columns,
                default=concentration_columns,
                key="wash_concentration_columns"
            )
        )

        individual_columns = [
            c for c in df.columns
            if (
                "Stage Removed (%)" in c
                or "Cumulative Removed (%)" in c
                or "Remaining in Organic (%)" in c
                or "Stage Re-extraction (%)" in c
                or "Cumulative Re-extraction (%)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Individual-metal removal / recovery",
                individual_columns,
                default=[],
                key="wash_individual_columns"
            )
        )

        group_columns = [
            c for c in df.columns
            if (
                c.startswith("Light ")
                or c.startswith("Heavy ")
            )
        ]

        selected.extend(
            st.multiselect(
                "Light / Heavy group metrics",
                group_columns,
                default=group_columns,
                key="wash_group_columns"
            )
        )

        composition_columns = [
            c for c in df.columns
            if (
                "Aqueous Light (%)" in c
                or "Aqueous Heavy (%)" in c
                or "Organic Light (%)" in c
                or "Organic Heavy (%)" in c
                or "Light: Aqueous (%)" in c
                or "Light: Organic (%)" in c
                or "Heavy: Aqueous (%)" in c
                or "Heavy: Organic (%)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Molar composition and phase distribution",
                composition_columns,
                default=composition_columns,
                key="wash_composition_columns"
            )
        )

    else:

        concentration_columns = [
            c for c in df.columns
            if (
                "Aq (mol/L)" in c
                or "Org (mol/L)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Metal concentrations",
                concentration_columns,
                default=concentration_columns,
                key="strip_concentration_columns"
            )
        )

        individual_columns = [
            c for c in df.columns
            if (
                "Stage Re-extraction (%)" in c
                or "Cumulative Re-extraction (%)" in c
                or "Remaining in Organic (%)" in c
            )
        ]

        selected.extend(
            st.multiselect(
                "Individual-metal re-extraction",
                individual_columns,
                default=[],
                key="strip_individual_columns"
            )
        )

    # Preserve original DataFrame order
    selected = [
        col for col in df.columns
        if col in selected
    ]

    return df[selected]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Simulation"
)

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

    st.title(
        "REE Solvent Extraction"
    )

    st.caption(
        "Sequential equilibrium contacts are used "
        "to represent the extraction process."
    )

    st.header(
        "Feed composition"
    )

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

    st.subheader(
        "REE concentrations"
    )

    feed_values = {}

    cols = st.columns(5)

    for i, metal in enumerate(
        METALS
    ):

        with cols[i % 5]:

            feed_values[metal] = st.number_input(
                f"{metal} ({concentration_unit})",
                min_value=0.0,
                value=0.0,
                format="%.8g",
                key=f"feed_{metal}"
            )

    st.divider()

    st.header(
        "Extraction parameters"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        cut_pair = st.selectbox(
            "REE cut",
            all_cut_pairs(),
            index=3
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
            "(mol/L, dimer basis, as (HA)\u2082)",
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

    st.session_state.cut_pair = cut_pair

    light, heavy = get_groups(
        cut_pair
    )

    st.info(
        f"Cut: **{cut_pair}**  |  "
        f"Light REEs: **{', '.join(light)}**  |  "
        f"Heavy REEs: **{', '.join(heavy)}**"
    )

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

        st.session_state.extraction_feed = (
            initial_aq.copy()
        )

        st.session_state.extraction_history = []

        st.session_state.extraction_started = True
        st.session_state.stop_extraction = False

        st.session_state.combined_organic = None

        st.rerun()

    if st.session_state.extraction_started:

        initial_aq = (
            st.session_state.extraction_feed
        )

        if len(
            st.session_state.extraction_history
        ) == 0:

            current_aq = initial_aq.copy()

            h_in = 10 ** (
                -initial_pH
            )

        else:

            last = (
                st.session_state
                .extraction_history[-1]
            )

            current_aq = np.array([
                last[
                    f"{m} Aq (mol/L)"
                ]
                for m in METALS
            ])

            h_in = last[
                "H+ (mol/L)"
            ]

        if st.button(
            "Add extraction stage",
            type="secondary",
            disabled=st.session_state.stop_extraction
        ):

            saponified_capacity = (
                extractant_total
                * saponification
                / 100.0
            )

            result = solve_extraction_stage(
                current_aq,
                h_in,
                extractant_total,
                saponified_capacity,
                oa_ratio
            )

            stage = (
                len(
                    st.session_state
                    .extraction_history
                ) + 1
            )

            previous_aq = (
                current_aq.copy()
            )

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

            st.rerun()

        if st.session_state.extraction_history:

            df_ext = pd.DataFrame(
                st.session_state.extraction_history
            )

            st.subheader(
                "Extraction results"
            )

            display_ext = select_table_columns(
                df_ext,
                "Extraction"
            )

            st.dataframe(
                display_ext,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "Extraction graphs"
            )

            selected_metals = st.multiselect(
                "Metals to display",
                METALS,
                default=["Nd", "Sm"],
                key="ext_metals"
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

            # =================================================
            # ELEMENTAL COMPOSITION — FIXED
            # =================================================

            st.subheader(
                "Elemental molar composition"
            )

            phase = st.radio(
                "Phase",
                [
                    "Aqueous",
                    "Organic"
                ],
                horizontal=True,
                key="ext_element_phase"
            )

            phase_code = (
                "Aq"
                if phase == "Aqueous"
                else "Org"
            )

            st.pyplot(
                plot_elemental_composition(
                    df_ext,
                    phase_code,
                    f"{phase} elemental molar composition"
                ),
                clear_figure=True
            )

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

                organic_moles = np.zeros(
                    len(METALS)
                )

                total_organic_volume = 0.0

                stage_organic_volume = (
                    feed_volume
                    * oa_ratio
                )

                for row in (
                    st.session_state
                    .extraction_history
                ):

                    c_org = np.array([
                        row[
                            f"{m} Org (mol/L)"
                        ]
                        for m in METALS
                    ])

                    organic_moles += (
                        c_org
                        * stage_organic_volume
                    )

                    total_organic_volume += (
                        stage_organic_volume
                    )

                combined_organic = (
                    organic_moles
                    / max(
                        total_organic_volume,
                        1e-30
                    )
                )

                extractant_moles = 0.0

                for row in (
                    st.session_state
                    .extraction_history
                ):

                    e_free = row[
                        "Free extractant (mol/L)"
                    ]

                    extractant_moles += (
                        e_free
                        * stage_organic_volume
                    )

                combined_extractant = (
                    extractant_moles
                    / max(
                        total_organic_volume,
                        1e-30
                    )
                )

                st.session_state.combined_organic = {

                    "concentration":
                        combined_organic,

                    "volume":
                        total_organic_volume,

                    "moles":
                        organic_moles,

                    "extractant_concentration":
                        combined_extractant,

                    "extractant_moles":
                        extractant_moles
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

    st.title(
        "Organic Phase Washing"
    )

    if st.session_state.combined_organic is None:

        st.warning(
            "Complete and stop the extraction first."
        )

        st.stop()

    combined = (
        st.session_state.combined_organic
    )

    st.info(
        f"Combined organic phase: "
        f"**{combined['volume']:.4f} L**"
    )

    st.info(
        f"Combined free extractant concentration: "
        f"**{combined['extractant_concentration']:.6f} "
        f"mol/L (dimer basis)**"
    )

    cut_pair = (
        st.session_state.cut_pair
    )

    if cut_pair is None:

        st.error(
            "No extraction cut was found."
        )

        st.stop()

    light, heavy = get_groups(
        cut_pair
    )

    st.header(
        "Washing parameters"
    )

    col1, col2 = st.columns(2)

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

    st.caption(
        "The washing operation uses the organic phase "
        "obtained from the extraction. Its extractant "
        "concentration is calculated automatically."
    )

    st.subheader(
        "Current combined organic feed"
    )

    feed_df = pd.DataFrame({
        "REE": METALS,

        "Organic concentration (mol/L)":
            combined["concentration"],

        "Organic amount (mol)":
            combined["moles"]
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

        st.session_state.washing_extractant_feed = (
            combined["extractant_concentration"]
        )

        st.rerun()

    if st.session_state.washing_started:

        if st.button(
            "Add washing stage",
            disabled=st.session_state.stop_washing
        ):

            corg_in = (
                st.session_state
                .washing_organic_feed
            )

            extractant_in = (
                st.session_state
                .washing_extractant_feed
            )

            result = solve_washing_stage(
                corg_in,
                10 ** (-wash_pH),
                extractant_in,
                wash_ao
            )

            stage = (
                len(
                    st.session_state
                    .washing_history
                ) + 1
            )

            row = {
                "Stage": stage,

                "pH": result["pH"],

                "H+ (mol/L)": result["h"],

                "Free extractant (mol/L)":
                    result["free_extractant"]
            }

            aq = result["caq"]
            org = result["corg"]

            initial = combined[
                "concentration"
            ]

            for i, metal in enumerate(
                METALS
            ):

                removed_stage = safe_percent(
                    corg_in[i] - org[i],
                    corg_in[i]
                )

                removed_cumulative = safe_percent(
                    initial[i] - org[i],
                    initial[i]
                )

                remaining = (
                    100.0
                    - removed_cumulative
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

            row[
                "Organic Light (%)"
            ] = phase_composition(
                org,
                light
            )

            row[
                "Organic Heavy (%)"
            ] = phase_composition(
                org,
                heavy
            )

            row[
                "Aqueous Light (%)"
            ] = phase_composition(
                aq,
                light
            )

            row[
                "Aqueous Heavy (%)"
            ] = phase_composition(
                aq,
                heavy
            )

            light_aq, light_org = (
                group_phase_distribution(
                    aq,
                    org,
                    light
                )
            )

            heavy_aq, heavy_org = (
                group_phase_distribution(
                    aq,
                    org,
                    heavy
                )
            )

            row[
                "Light: Aqueous (%)"
            ] = light_aq

            row[
                "Light: Organic (%)"
            ] = light_org

            row[
                "Heavy: Aqueous (%)"
            ] = heavy_aq

            row[
                "Heavy: Organic (%)"
            ] = heavy_org

            # =================================================
            # GROUP METRICS
            # =================================================

            initial_light = sum(
                initial[
                    METALS.index(m)
                ]
                for m in light
            )

            previous_light = sum(
                corg_in[
                    METALS.index(m)
                ]
                for m in light
            )

            current_light = sum(
                org[
                    METALS.index(m)
                ]
                for m in light
            )

            initial_heavy = sum(
                initial[
                    METALS.index(m)
                ]
                for m in heavy
            )

            previous_heavy = sum(
                corg_in[
                    METALS.index(m)
                ]
                for m in heavy
            )

            current_heavy = sum(
                org[
                    METALS.index(m)
                ]
                for m in heavy
            )

            row[
                "Light Stage Removed (%)"
            ] = safe_percent(
                previous_light - current_light,
                previous_light
            )

            row[
                "Light Cumulative Removed (%)"
            ] = safe_percent(
                initial_light - current_light,
                initial_light
            )

            row[
                "Light Remaining in Organic (%)"
            ] = (
                100.0
                - row[
                    "Light Cumulative Removed (%)"
                ]
            )

            row[
                "Heavy Stage Removed (%)"
            ] = safe_percent(
                previous_heavy - current_heavy,
                previous_heavy
            )

            row[
                "Heavy Cumulative Removed (%)"
            ] = safe_percent(
                initial_heavy - current_heavy,
                initial_heavy
            )

            row[
                "Heavy Remaining in Organic (%)"
            ] = (
                100.0
                - row[
                    "Heavy Cumulative Removed (%)"
                ]
            )

            st.session_state.washing_history.append(
                row
            )

            st.session_state.washing_organic_feed = (
                org.copy()
            )

            st.session_state.washing_extractant_feed = (
                result["free_extractant"]
            )

            st.rerun()

        if st.session_state.washing_history:

            df_wash = pd.DataFrame(
                st.session_state.washing_history
            )

            st.subheader(
                "Washing results"
            )

            display_wash = select_table_columns(
                df_wash,
                "Washing"
            )

            st.dataframe(
                display_wash,
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

                c1, c2 = st.columns(2)

                with c1:

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

                with c2:

                    cols = [
                        f"{m} Stage Removed (%)"
                        for m in selected_metals
                    ]

                    st.pyplot(
                        plot_line_chart(
                            df_wash,
                            cols,
                            "Stage REE removal during washing",
                            "Stage removed (%)"
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
                    "Org",
                    "Washed organic elemental composition"
                ),
                clear_figure=True
            )

            if st.button(
                "Stop washing and continue to re-extraction",
                type="primary"
            ):

                final_org = (
                    st.session_state
                    .washing_organic_feed
                )

                final_moles = (
                    final_org
                    * combined["volume"]
                )

                final_extractant = (
                    st.session_state
                    .washing_extractant_feed
                )

                final_extractant_moles = (
                    final_extractant
                    * combined["volume"]
                )

                st.session_state.stripping_organic_feed = {

                    "concentration":
                        final_org.copy(),

                    "volume":
                        combined["volume"],

                    "moles":
                        final_moles,

                    "extractant_concentration":
                        final_extractant,

                    "extractant_moles":
                        final_extractant_moles
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

    st.title(
        "REE Re-extraction / Stripping"
    )

    if (
        st.session_state
        .stripping_organic_feed is None
    ):

        st.warning(
            "Complete and stop the washing first."
        )

        st.stop()

    feed = (
        st.session_state
        .stripping_organic_feed
    )

    # ========================================================
    # FIX:
    # Define light/heavy here too.
    # This prevents the NameError that occurred when
    # the re-extraction page attempted to use them.
    # ========================================================

    cut_pair = (
        st.session_state.cut_pair
    )

    if cut_pair is None:

        st.error(
            "No extraction cut was found."
        )

        st.stop()

    light, heavy = get_groups(
        cut_pair
    )

    st.info(
        f"Organic phase entering stripping: "
        f"**{feed['volume']:.4f} L**"
    )

    st.info(
        f"Extractant concentration entering stripping: "
        f"**{feed['extractant_concentration']:.6f} "
        f"mol/L (dimer basis)**"
    )

    st.header(
        "Stripping parameters"
    )

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
                feed["concentration"].copy(),

            "current_extractant":
                feed["extractant_concentration"]
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

            current_extractant = (
                st.session_state
                .stripping_organic_feed[
                    "current_extractant"
                ]
            )

            result = solve_stripping_stage(
                current_org,
                stripping_H,
                current_extractant,
                stripping_ao
            )

            stage = (
                len(
                    st.session_state
                    .stripping_history
                ) + 1
            )

            row = {
                "Stage": stage,

                "pH": result["pH"],

                "H+ (mol/L)": result["h"],

                "Free extractant (mol/L)":
                    result["free_extractant"]
            }

            aq = result["caq"]
            org = result["corg"]

            initial = feed[
                "concentration"
            ]

            for i, metal in enumerate(
                METALS
            ):

                stage_reextraction = safe_percent(
                    current_org[i] - org[i],
                    current_org[i]
                )

                cumulative_reextraction = safe_percent(
                    initial[i] - org[i],
                    initial[i]
                )

                remaining = (
                    100.0
                    - cumulative_reextraction
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

            # =================================================
            # GROUP RE-EXTRACTION METRICS
            # =================================================

            initial_light = sum(
                initial[
                    METALS.index(m)
                ]
                for m in light
            )

            previous_light = sum(
                current_org[
                    METALS.index(m)
                ]
                for m in light
            )

            current_light = sum(
                org[
                    METALS.index(m)
                ]
                for m in light
            )

            initial_heavy = sum(
                initial[
                    METALS.index(m)
                ]
                for m in heavy
            )

            previous_heavy = sum(
                current_org[
                    METALS.index(m)
                ]
                for m in heavy
            )

            current_heavy = sum(
                org[
                    METALS.index(m)
                ]
                for m in heavy
            )

            row[
                "Light Stage Re-extraction (%)"
            ] = safe_percent(
                previous_light - current_light,
                previous_light
            )

            row[
                "Light Cumulative Re-extraction (%)"
            ] = safe_percent(
                initial_light - current_light,
                initial_light
            )

            row[
                "Light Remaining in Organic (%)"
            ] = (
                100.0
                - row[
                    "Light Cumulative Re-extraction (%)"
                ]
            )

            row[
                "Heavy Stage Re-extraction (%)"
            ] = safe_percent(
                previous_heavy - current_heavy,
                previous_heavy
            )

            row[
                "Heavy Cumulative Re-extraction (%)"
            ] = safe_percent(
                initial_heavy - current_heavy,
                initial_heavy
            )

            row[
                "Heavy Remaining in Organic (%)"
            ] = (
                100.0
                - row[
                    "Heavy Cumulative Re-extraction (%)"
                ]
            )

            # =================================================
            # PHASE COMPOSITION
            # =================================================

            row[
                "Organic Light (%)"
            ] = phase_composition(
                org,
                light
            )

            row[
                "Organic Heavy (%)"
            ] = phase_composition(
                org,
                heavy
            )

            row[
                "Aqueous Light (%)"
            ] = phase_composition(
                aq,
                light
            )

            row[
                "Aqueous Heavy (%)"
            ] = phase_composition(
                aq,
                heavy
            )

            light_aq, light_org = (
                group_phase_distribution(
                    aq,
                    org,
                    light
                )
            )

            heavy_aq, heavy_org = (
                group_phase_distribution(
                    aq,
                    org,
                    heavy
                )
            )

            row[
                "Light: Aqueous (%)"
            ] = light_aq

            row[
                "Light: Organic (%)"
            ] = light_org

            row[
                "Heavy: Aqueous (%)"
            ] = heavy_aq

            row[
                "Heavy: Organic (%)"
            ] = heavy_org

            st.session_state.stripping_history.append(
                row
            )

            st.session_state.stripping_organic_feed[
                "current_concentration"
            ] = org.copy()

            st.session_state.stripping_organic_feed[
                "current_extractant"
            ] = result[
                "free_extractant"
            ]

            st.rerun()

        if st.session_state.stripping_history:

            df_strip = pd.DataFrame(
                st.session_state.stripping_history
            )

            st.subheader(
                "Re-extraction results"
            )

            display_strip = select_table_columns(
                df_strip,
                "Re-extraction"
            )

            st.dataframe(
                display_strip,
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

            # =================================================
            # GROUP COMPOSITION GRAPHS
            # =================================================

            c1, c2 = st.columns(2)

            with c1:

                st.pyplot(
                    plot_phase_composition(
                        df_strip,
                        "Organic Light (%)",
                        "Organic Heavy (%)",
                        "Organic Light / Heavy composition"
                    ),
                    clear_figure=True
                )

            with c2:

                st.pyplot(
                    plot_phase_composition(
                        df_strip,
                        "Aqueous Light (%)",
                        "Aqueous Heavy (%)",
                        "Aqueous Light / Heavy composition"
                    ),
                    clear_figure=True
                )

            c1, c2 = st.columns(2)

            with c1:

                st.pyplot(
                    plot_group_distribution(
                        df_strip,
                        "Light: Aqueous (%)",
                        "Light: Organic (%)",
                        "Light REE distribution during re-extraction"
                    ),
                    clear_figure=True
                )

            with c2:

                st.pyplot(
                    plot_group_distribution(
                        df_strip,
                        "Heavy: Aqueous (%)",
                        "Heavy: Organic (%)",
                        "Heavy REE distribution during re-extraction"
                    ),
                    clear_figure=True
                )

            # =================================================
            # ELEMENTAL COMPOSITION
            # =================================================

            st.subheader(
                "Elemental composition"
            )

            st.pyplot(
                plot_elemental_composition(
                    df_strip,
                    "Org",
                    "Remaining organic elemental composition"
                ),
                clear_figure=True
            )

            final_values = []

            for metal in METALS:

                final_values.append(
                    df_strip.iloc[-1][
                        f"{metal} "
                        "Cumulative Re-extraction (%)"
                    ]
                )

            minimum_recovery = min(
                final_values
            )

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
