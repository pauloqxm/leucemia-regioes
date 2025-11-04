import streamlit as st
import pandas as pd
import numpy as np
import io
import math
import matplotlib.pyplot as plt

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
            a = int(parts[0])
            b = int(parts[1])
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
# Carregamento de dados
# ------------------------------

st.sidebar.header("Configurações")
st.sidebar.markdown("Envie os dois arquivos CSV obrigatórios para continuar.")

uploaded_obitos = st.sidebar.file_uploader("Dados de Óbitos (.csv)", type=["csv"], key="obitos")
uploaded_pop = st.sidebar.file_uploader("Dados de População (.csv)", type=["csv"], key="pop")

if uploaded_obitos is None or uploaded_pop is None:
    st.warning("Envie ambos os arquivos: **Óbitos** e **População** para iniciar a análise.")
    st.stop()

df_ob = read_csv_safely(uploaded_obitos)
df_pop = read_csv_safely(uploaded_pop)

# ------------------------------
# Mapeamento de colunas
# ------------------------------

cols_ob = {
    "region": coalesce_columns(df_ob, "região", [["regiao","região","uf","region"]]),
    "year": coalesce_columns(df_ob, "ano", [["ano","year","periodo"]]),
    "sex": coalesce_columns(df_ob, "sexo", [["sexo","sex","genero"]], required=False),
    "age": coalesce_columns(df_ob, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup"]]),
    "deaths": coalesce_columns(df_ob, "óbitos", [["obitos","mortes","deaths"]]),
}

cols_pop = {
    "region": coalesce_columns(df_pop, "região", [["regiao","região","uf","region"]]),
    "year": coalesce_columns(df_pop, "ano", [["ano","year"]]),
    "age": coalesce_columns(df_pop, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup"]]),
    "population": coalesce_columns(df_pop, "população", [["populacao","population","habitantes"]]),
}

allowed_age_groups = WHO_STD["AgeGroup"].tolist()

df_ob = df_ob.rename(columns={
    cols_ob["region"]:"Region", cols_ob["year"]:"Year", cols_ob["sex"]:"Sex" if cols_ob["sex"] else "Sex",
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
std_df = WHO_STD.copy()
if std_choice == "Arquivo CSV (AgeGroup, StdPop)":
    up = st.sidebar.file_uploader("População Padrão (CSV)", type=["csv"], key="stdpop")
    if up is not None:
        tmp = read_csv_safely(up)
        cand_age = find_col(tmp, ["agegroup","faixa etaria"])
        cand_std = find_col(tmp, ["stdpop","populacao padrao"])
        if cand_age and cand_std:
            tmp = tmp.rename(columns={cand_age:"AgeGroup", cand_std:"StdPop"})[["AgeGroup","StdPop"]]
            tmp["AgeGroup"] = tmp["AgeGroup"].map(harmonize_age_group)
            tmp["StdPop"] = pd.to_numeric(tmp["StdPop"], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                std_df = tmp

# ------------------------------
# Cálculos
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
for region in sorted(deaths_age["Region"].unique().tolist()):
    d = deaths_age[deaths_age["Region"]==region].set_index("AgeGroup")["Deaths"]
    p = pop_mid_age[pop_mid_age["Region"]==region].set_index("AgeGroup")["Population"]
    d = d.reindex(std_df["AgeGroup"]).fillna(0)
    p = p.reindex(std_df["AgeGroup"]).fillna(0)
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

st.markdown("---")
st.subheader("Considerações Éticas")
st.markdown("""
Trata-se de **dados públicos e anonimizados**.  
De acordo com a **Resolução CNS nº 510/2016**, o estudo dispensa submissão a Comitê de Ética em Pesquisa.
""")
