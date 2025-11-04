
import streamlit as st
import pandas as pd
import numpy as np
import io
import math
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Painel de Mortalidade por Leucemia (Nordeste x Sudeste)", layout="wide")

st.title("📊 Painel de Mortalidade por Leucemia — Nordeste x Sudeste")
st.caption("Cálculo de coeficientes brutos e padronizados por idade (método direto), análise descritiva por sexo e faixa etária, e tendências ao longo do tempo.")

# ------------------------------
# Utilidades
# ------------------------------

@st.cache_data
def read_csv_safely(path_or_buffer, **kwargs):
    try:
        return pd.read_csv(path_or_buffer, **kwargs)
    except Exception:
        try:
            return pd.read_csv(path_or_buffer, sep=';', decimal=',', **kwargs)
        except Exception:
            try:
                return pd.read_csv(path_or_buffer, sep=';', decimal='.', **kwargs)
            except Exception:
                return pd.read_csv(path_or_buffer, sep=',', decimal='.', **kwargs)

def normalize_col(s):
    return (
        s.strip()
         .lower()
         .replace("ã","a").replace("â","a").replace("á","a").replace("à","a")
         .replace("ê","e").replace("é","e").replace("è","e")
         .replace("î","i").replace("í","i").replace("ì","i")
         .replace("õ","o").replace("ô","o").replace("ó","o").replace("ò","o")
         .replace("û","u").replace("ú","u").replace("ù","u")
         .replace("ç","c")
         .replace("  "," ").replace("  "," ")
    )

def find_col(df, candidates):
    cols_norm = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        nc = normalize_col(cand)
        if nc in cols_norm:
            return cols_norm[nc]
    for c in df.columns:
        if any(token in normalize_col(c) for token in [normalize_col(x) for x in candidates]):
            return c
    return None

def coalesce_columns(df, name, candidate_lists, required=True):
    for candidates in candidate_lists:
        col = find_col(df, candidates)
        if col is not None:
            return col
    if required:
        raise ValueError(f"Não encontrei a coluna para '{name}'. Verifique os cabeçalhos.")
    return None

# WHO World Standard Population (2000-2025)
WHO_STD = pd.DataFrame({
    "AgeGroup": [
        "0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
        "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"
    ],
    "StdPop": [
        8800, 8700, 8600, 8600, 8800, 8800, 8800, 8700,
        8600, 8200, 8000, 6500, 5000, 4000, 3000, 2000, 1000
    ]
})

def harmonize_age_group(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    s = str(s).strip()
    s = s.replace(" anos", "").replace("anos", "")
    s = s.replace("anos ou mais", "+").replace("ou mais", "+")
    s = s.replace(" a ", "-").replace("–","-").replace("—","-").replace("−","-")
    s = s.replace(" ", "")
    if s in ["80+", "80mais", "80oumais"]:
        return "80+"
    parts = s.split("-")
    if len(parts) == 2:
        try:
            a = int(parts[0]); b = int(parts[1])
            return f"{a}-{b}"
        except Exception:
            pass
    return s

def direct_standardization(deaths_by_age, pop_by_age, std_df):
    df = pd.DataFrame({"Deaths": deaths_by_age, "Population": pop_by_age}).dropna()
    if df.empty:
        return np.nan
    df["Rate"] = df["Deaths"] / df["Population"]
    std = std_df.set_index("AgeGroup").reindex(df.index)
    std = std["StdPop"].fillna(0)
    if std.sum() == 0:
        return np.nan
    cmp_value = (df["Rate"] * std).sum() / std.sum() * 100000.0
    return cmp_value

# ------------------------------
# Carregamento de dados (prioriza diretório do app)
# ------------------------------

st.sidebar.header("Dados")
st.sidebar.caption("O app tentará carregar os CSVs do **mesmo diretório** do arquivo .py. Se não encontrar, você pode enviar abaixo.")

def candidate_paths(filename_variants):
    # tenta no diretório do arquivo e no diretório atual
    here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    paths = []
    for name in filename_variants:
        paths.append(os.path.join(here, name))
        paths.append(os.path.join(os.getcwd(), name))
    # remover duplicados mantendo ordem
    seen = set(); ordered = []
    for p in paths:
        if p not in seen:
            seen.add(p); ordered.append(p)
    return ordered

obitos_variants = ["Dados de Óbitos.csv", "Dados de Obitos.csv"]
pop_variants = ["Dados de População.csv", "Dados de Populacao.csv"]

df_ob, df_pop = None, None

# Tenta ler automaticamente
for p in candidate_paths(obitos_variants):
    if os.path.exists(p):
        try:
            df_ob = read_csv_safely(p)
            break
        except Exception:
            pass

for p in candidate_paths(pop_variants):
    if os.path.exists(p):
        try:
            df_pop = read_csv_safely(p)
            break
        except Exception:
            pass

# Se não conseguiu, oferece upload como fallback
uploaded_obitos = None
uploaded_pop = None

if df_ob is None:
    uploaded_obitos = st.sidebar.file_uploader("Enviar CSV de Óbitos", type=["csv"], key="obitos_up")

if df_pop is None:
    uploaded_pop = st.sidebar.file_uploader("Enviar CSV de População", type=["csv"], key="pop_up")

if df_ob is None and uploaded_obitos is not None:
    df_ob = read_csv_safely(uploaded_obitos)

if df_pop is None and uploaded_pop is not None:
    df_pop = read_csv_safely(uploaded_pop)

if df_ob is None or df_pop is None:
    st.error("Não foi possível localizar **ambos** os arquivos no diretório do app e nenhum upload foi fornecido. Verifique os nomes e tente novamente.")
    st.stop()

# ------------------------------
# Mapeamento de colunas
# ------------------------------

cols_ob = {
    "region": None,
    "year": None,
    "sex": None,
    "age": None,
    "deaths": None,
}

cols_pop = {
    "region": None,
    "year": None,
    "age": None,
    "population": None,
}

def map_columns(df_ob, df_pop):
    cols_ob = {
        "region": coalesce_columns(df_ob, "região", [["regiao","região","uf","region"]]),
        "year": coalesce_columns(df_ob, "ano", [["ano","year","periodo"]]),
        "sex": coalesce_columns(df_ob, "sexo", [["sexo","sex","genero","gênero"]], required=False),
        "age": coalesce_columns(df_ob, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup","age_group"]]),
        "deaths": coalesce_columns(df_ob, "óbitos", [["obitos","óbitos","mortes","deaths","count"]]),
    }
    cols_pop = {
        "region": coalesce_columns(df_pop, "região", [["regiao","região","uf","region"]]),
        "year": coalesce_columns(df_pop, "ano", [["ano","year"]]),
        "age": coalesce_columns(df_pop, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup","age_group"]]),
        "population": coalesce_columns(df_pop, "população", [["populacao","população","population","habitantes","estimativa"]]),
    }
    return cols_ob, cols_pop

cols_ob, cols_pop = map_columns(df_ob, df_pop)

# Harmonização
df_ob = df_ob.rename(columns={
    cols_ob["region"]:"Region", cols_ob["year"]:"Year", (cols_ob["sex"] or "Sex"):"Sex",
    cols_ob["age"]:"AgeGroup", cols_ob["deaths"]:"Deaths"
})
df_pop = df_pop.rename(columns={
    cols_pop["region"]:"Region", cols_pop["year"]:"Year", cols_pop["age"]:"AgeGroup", cols_pop["population"]:"Population"
})

df_ob["Sex"] = df_ob.get("Sex", "Todos")
df_ob["AgeGroup"] = df_ob["AgeGroup"].map(harmonize_age_group)
df_pop["AgeGroup"] = df_pop["AgeGroup"].map(harmonize_age_group)

# ------------------------------
# Filtros
# ------------------------------

regions = sorted(df_ob["Region"].dropna().unique().tolist())
years = sorted(pd.Series(df_ob["Year"].dropna().unique()).astype(int).tolist())

st.sidebar.subheader("Filtros")
sel_regions = st.sidebar.multiselect("Regiões", regions, default=regions)
sel_sex = st.sidebar.multiselect("Sexo", sorted(df_ob["Sex"].dropna().unique().tolist()), default=sorted(df_ob["Sex"].dropna().unique().tolist()))
sel_year_range = st.sidebar.slider("Período (anos)", min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))), step=1)

std_choice = st.sidebar.selectbox("População Padrão", ["WHO 2000-2025 (OMS)", "Arquivo CSV (AgeGroup, StdPop)"])
WHO_STD = WHO_STD  # already defined
std_df = WHO_STD.copy()
if std_choice == "Arquivo CSV (AgeGroup, StdPop)":
    up = st.sidebar.file_uploader("População Padrão (CSV)", type=["csv"], key="stdpop")
    if up is not None:
        tmp = read_csv_safely(up)
        cand_age = find_col(tmp, ["agegroup","age_group","faixa etaria","faixa_etaria","idade","grupo etario","grupo etário"])
        cand_std = find_col(tmp, ["stdpop","populacao padrao","populacao_padrao","standard","peso","peso padrao"])
        if cand_age and cand_std:
            tmp = tmp.rename(columns={cand_age:"AgeGroup", cand_std:"StdPop"})[["AgeGroup","StdPop"]]
            tmp["AgeGroup"] = tmp["AgeGroup"].map(harmonize_age_group)
            tmp["StdPop"] = pd.to_numeric(tmp["StdPop"], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                std_df = tmp

# ------------------------------
# Cálculos CMB/CMP
# ------------------------------

mask_ob = (df_ob["Region"].isin(sel_regions)) & (df_ob["Sex"].isin(sel_sex)) & (df_ob["Year"].between(sel_year_range[0], sel_year_range[1]))
mask_pop = (df_pop["Region"].isin(sel_regions)) & (df_pop["Year"].between(sel_year_range[0], sel_year_range[1]))

ob = df_ob.loc[mask_ob].copy()
pop = df_pop.loc[mask_pop].copy()

mid_year = (sel_year_range[0] + sel_year_range[1]) // 2
pop_mid = pop[pop["Year"] == mid_year].groupby(["Region"], as_index=False)["Population"].sum()
deaths_period = ob.groupby(["Region"], as_index=False)["Deaths"].sum()
cmb = pd.merge(deaths_period, pop_mid, on="Region", how="left")
cmb["CMB (óbitos/100.000)"] = np.where(cmb["Population"]>0, (cmb["Deaths"] / cmb["Population"]) * 100000.0, np.nan)

# CMP
deaths_age = ob.groupby(["Region","AgeGroup"], as_index=False)["Deaths"].sum()
pop_mid_age = pop[pop["Year"] == mid_year].groupby(["Region","AgeGroup"], as_index=False)["Population"].sum()
cmp_rows = []
age_order = std_df["AgeGroup"].tolist()
for region in sorted(deaths_age["Region"].unique().tolist()):
    d = deaths_age[deaths_age["Region"]==region].set_index("AgeGroup")["Deaths"]
    p = pop_mid_age[pop_mid_age["Region"]==region].set_index("AgeGroup")["Population"]
    d = d.reindex(age_order).fillna(0)
    p = p.reindex(age_order).fillna(0)
    cmp_val = direct_standardization(d, p, std_df)
    cmp_rows.append({"Region": region, "CMP (óbitos/100.000)": cmp_val})
cmp = pd.DataFrame(cmp_rows)

# ------------------------------
# Exibição
# ------------------------------

st.subheader("Coeficiente de Mortalidade Bruto (CMB)")
st.dataframe(cmb, use_container_width=True)

st.subheader("Coeficiente de Mortalidade Padronizado (CMP)")
st.dataframe(cmp, use_container_width=True)

# Tendências (opcional — rápidas)
st.markdown("---")
st.subheader("Tendências ao longo do tempo")

series_rows = []
for region in sel_regions:
    for year in range(sel_year_range[0], sel_year_range[1]+1):
        ob_y = ob[(ob["Region"]==region) & (ob["Year"]==year)]
        pop_y = pop[(pop["Region"]==region) & (pop["Year"]==year)]
        deaths_total = ob_y["Deaths"].sum()
        pop_total = pop_y["Population"].sum()
        cmb_y = (deaths_total / pop_total) * 100000.0 if pop_total>0 else np.nan
        d_age = ob_y.groupby("AgeGroup")["Deaths"].sum()
        p_age = pop_y.groupby("AgeGroup")["Population"].sum()
        d_age = d_age.reindex(age_order).fillna(0)
        p_age = p_age.reindex(age_order).fillna(0)
        cmp_y = direct_standardization(d_age, p_age, std_df)
        series_rows.append({"Region":region, "Year":year, "CMB":cmb_y, "CMP":cmp_y})

series = pd.DataFrame(series_rows)

if not series.empty:
    fig1, ax1 = plt.subplots()
    for region in sel_regions:
        s = series[series["Region"]==region].sort_values("Year")
        ax1.plot(s["Year"], s["CMB"], label=str(region))
    ax1.set_xlabel("Ano"); ax1.set_ylabel("CMB (óbitos por 100.000)"); ax1.set_title("Tendência do CMB"); ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    for region in sel_regions:
        s = series[series["Region"]==region].sort_values("Year")
        ax2.plot(s["Year"], s["CMP"], label=str(region))
    ax2.set_xlabel("Ano"); ax2.set_ylabel("CMP (óbitos por 100.000)"); ax2.set_title("Tendência do CMP (padronizado)"); ax2.legend()
    st.pyplot(fig2)

# Ética
st.markdown("---")
st.subheader("Considerações Éticas")
st.markdown("Dados públicos e anonimizados. Conforme a **Resolução CNS nº 510/2016**, o estudo dispensa submissão ao CEP.")
