
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
    # fallback: try contains
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

# WHO World Standard Population (2000-2025) (5-year groups) — weights out of 100,000
# Source: Ahmad OB, Boschi-Pinto C, Lopez AD, et al. Age Standardization of Rates: A New WHO Standard.
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
    s = s.replace("80e+", "80+").replace("80oumais","80+")
    # common forms
    if s in ["80+", "80mais", "80oumais", "80+"]:
        return "80+"
    if s.lower() in ["ignorado","desconhecido","na","n/a","semidade"]:
        return None
    # pad like 0-4, 5-9 etc
    # try parse "0-04", "05-09"
    parts = s.split("-")
    if len(parts) == 2:
        try:
            a = int(parts[0])
            b = int(parts[1])
            return f"{a}-{b}"
        except Exception:
            pass
    return s

def align_age_groups(df_age, age_col, allowed_groups):
    df = df_age.copy()
    df[age_col] = df[age_col].map(harmonize_age_group)
    df = df[df[age_col].isin(allowed_groups)]
    return df

def compute_mid_year(years):
    try:
        years = sorted(int(y) for y in years if pd.notnull(y))
        if not years:
            return None
        return years[len(years)//2]
    except Exception:
        return None

def direct_standardization(deaths_by_age, pop_by_age, std_df):
    # deaths_by_age, pop_by_age indexed by AgeGroup with numeric values
    df = pd.DataFrame({
        "Deaths": deaths_by_age,
        "Population": pop_by_age
    }).dropna()
    if df.empty:
        return np.nan
    # Age-specific rates
    df["Rate"] = df["Deaths"] / df["Population"]
    std = std_df.set_index("AgeGroup").reindex(df.index)
    std = std["StdPop"].fillna(0)
    if std.sum() == 0:
        return np.nan
    # Direct standardized rate:
    cmp_value = (df["Rate"] * std).sum() / std.sum() * 100000.0
    return cmp_value

# ------------------------------
# Carregamento de dados
# ------------------------------

st.sidebar.header("Configurações")
st.sidebar.markdown("Carregue os arquivos ou use os pré-carregados.")

default_obitos_path = "/mnt/data/Dados de Obitos.csv"
default_pop_path = "/mnt/data/Dados de População.csv"

uploaded_obitos = st.sidebar.file_uploader("Dados de Óbitos (.csv)", type=["csv"], key="obitos")
uploaded_pop = st.sidebar.file_uploader("Dados de População (.csv)", type=["csv"], key="pop")

if uploaded_obitos is not None:
    df_ob = read_csv_safely(uploaded_obitos)
elif os.path.exists(default_obitos_path):
    df_ob = read_csv_safely(default_obitos_path)
else:
    df_ob = pd.DataFrame()

if uploaded_pop is not None:
    df_pop = read_csv_safely(uploaded_pop)
elif os.path.exists(default_pop_path):
    df_pop = read_csv_safely(default_pop_path)
else:
    df_pop = pd.DataFrame()

if df_ob.empty or df_pop.empty:
    st.warning("Envie os dois arquivos CSV de **Óbitos** e **População** para continuar.")
    st.stop()

# ------------------------------
# Mapeamento de colunas
# ------------------------------

# Esperados (com tolerância): Região, Ano/Periodo, Sexo, Faixa Etária, Óbitos (contagem)
# População: Região, Ano, Faixa Etária, População

cols_ob = {
    "region": coalesce_columns(df_ob, "região", [["regiao","região","regiao/uf","uf","macroregiao","macrorregiao","regiao (nordeste/sudeste)","region"]]),
    "year": coalesce_columns(df_ob, "ano/período", [["ano","periodo","período","ano do obito","ano do óbito","year"]]),
    "sex": coalesce_columns(df_ob, "sexo", [["sexo","sex","genero","gênero"]], required=False),
    "age": coalesce_columns(df_ob, "faixa etária", [["faixa etaria","faixa-etaria","faixa_etaria","idade","grupo etario","grupo etário","agegroup","age_group","idade (faixas)"]]),
    "deaths": coalesce_columns(df_ob, "óbitos", [["obitos","óbitos","mortes","mortalidade","deaths","count"]]),
}

cols_pop = {
    "region": coalesce_columns(df_pop, "região", [["regiao","região","regiao/uf","uf","macroregiao","macrorregiao","regiao (nordeste/sudeste)","region"]]),
    "year": coalesce_columns(df_pop, "ano", [["ano","year"]]),
    "age": coalesce_columns(df_pop, "faixa etária", [["faixa etaria","faixa-etaria","faixa_etaria","idade","grupo etario","grupo etário","agegroup","age_group","idade (faixas)"]]),
    "population": coalesce_columns(df_pop, "população", [["populacao","população","pop","population","estimativa populacional","habitantes"]]),
}

# Harmonizar faixas etárias conforme padrão WHO
allowed_age_groups = WHO_STD["AgeGroup"].tolist()

df_ob = df_ob.rename(columns={
    cols_ob["region"]:"Region", cols_ob["year"]:"Year", cols_ob["sex"]:"Sex" if cols_ob["sex"] else "Sex",
    cols_ob["age"]:"AgeGroup", cols_ob["deaths"]:"Deaths"
})
df_pop = df_pop.rename(columns={
    cols_pop["region"]:"Region", cols_pop["year"]:"Year", cols_pop["age"]:"AgeGroup", cols_pop["population"]:"Population"
})

# Forçar tipos
for col in ["Year"]:
    for d in (df_ob, df_pop):
        try:
            d[col] = pd.to_numeric(d[col], errors="coerce").astype("Int64")
        except Exception:
            d[col] = pd.to_numeric(d[col], errors="coerce")

for col in ["Deaths","Population"]:
    if col in df_ob.columns:
        if col=="Deaths":
            df_ob[col] = pd.to_numeric(df_ob[col], errors="coerce")
    if col in df_pop.columns:
        df_pop[col] = pd.to_numeric(df_pop[col], errors="coerce")

if "Sex" not in df_ob.columns:
    df_ob["Sex"] = "Todos"

# Harmonizar faixas e filtrar para grupos WHO
df_ob["AgeGroup"] = df_ob["AgeGroup"].map(lambda x: harmonize_age_group(x))
df_pop["AgeGroup"] = df_pop["AgeGroup"].map(lambda x: harmonize_age_group(x))
df_ob = df_ob[df_ob["AgeGroup"].isin(allowed_age_groups)]
df_pop = df_pop[df_pop["AgeGroup"].isin(allowed_age_groups)]

# ------------------------------
# Filtros
# ------------------------------

regions = sorted(df_ob["Region"].dropna().unique().tolist())
years = sorted(pd.Series(df_ob["Year"].dropna().unique()).astype(int).tolist())
min_year, max_year = (min(years), max(years)) if years else (None, None)

st.sidebar.subheader("Filtros")
sel_regions = st.sidebar.multiselect("Regiões", regions, default=regions)
sel_sex = st.sidebar.multiselect("Sexo", sorted(df_ob["Sex"].dropna().unique().tolist()), default=sorted(df_ob["Sex"].dropna().unique().tolist()))
sel_year_range = st.sidebar.slider("Período (anos)", min_value=int(min_year), max_value=int(max_year), value=(int(min_year), int(max_year)) if (min_year and max_year) else (2000, 2020), step=1)

std_choice = st.sidebar.selectbox("População Padrão para Padronização Direta", ["WHO 2000-2025 (padrão OMS)", "Arquivo CSV (AgeGroup, StdPop)"])
std_df = WHO_STD.copy()
user_std = None
if std_choice == "Arquivo CSV (AgeGroup, StdPop)":
    up = st.sidebar.file_uploader("Envie a População Padrão", type=["csv"], key="stdpop")
    if up is not None:
        tmp = read_csv_safely(up)
        cand_age = find_col(tmp, ["agegroup","age_group","faixa etaria","faixa_etaria","idade","grupo etario","grupo etário","faixa etária"])
        cand_std = find_col(tmp, ["stdpop","populacao padrao","populacao_padrao","standard","peso","peso padrao"])
        if cand_age and cand_std:
            user_std = tmp.rename(columns={cand_age:"AgeGroup", cand_std:"StdPop"})[["AgeGroup","StdPop"]].copy()
            user_std["AgeGroup"] = user_std["AgeGroup"].map(harmonize_age_group)
            user_std = user_std[user_std["AgeGroup"].isin(allowed_age_groups)]
            user_std["StdPop"] = pd.to_numeric(user_std["StdPop"], errors="coerce")
            user_std = user_std.dropna()
            if not user_std.empty:
                std_df = user_std

# Aplicar filtros básicos
mask_ob = (
    df_ob["Region"].isin(sel_regions) &
    df_ob["Sex"].isin(sel_sex) &
    df_ob["Year"].between(sel_year_range[0], sel_year_range[1])
)
mask_pop = (
    df_pop["Region"].isin(sel_regions) &
    df_pop["Year"].between(sel_year_range[0], sel_year_range[1])
)

ob = df_ob.loc[mask_ob].copy()
pop = df_pop.loc[mask_pop].copy()

# ------------------------------
# Cálculo CMB e CMP por Região e Período
# ------------------------------

# CMB: (total de óbitos no período) / (população no ponto médio do período) * 100000
mid_year = (sel_year_range[0] + sel_year_range[1]) // 2
pop_mid = pop[pop["Year"] == mid_year].groupby(["Region"], as_index=False)["Population"].sum()

deaths_period = ob.groupby(["Region"], as_index=False)["Deaths"].sum()
cmb = pd.merge(deaths_period, pop_mid, on="Region", how="left")
cmb["CMB (óbitos/100.000)"] = np.where(cmb["Population"]>0, (cmb["Deaths"] / cmb["Population"]) * 100000.0, np.nan)
cmb = cmb[["Region","Deaths","Population","CMB (óbitos/100.000)"]]

# CMP: padronização direta
# Precisamos das taxas específicas por idade: para cada Região, no período, somar óbitos por faixa etária e somar população do mid-year por faixa etária
deaths_age = ob.groupby(["Region","AgeGroup"], as_index=False)["Deaths"].sum()
pop_mid_age = pop[pop["Year"] == mid_year].groupby(["Region","AgeGroup"], as_index=False)["Population"].sum()

cmp_rows = []
for region in sorted(deaths_age["Region"].unique().tolist()):
    d = deaths_age[deaths_age["Region"]==region].set_index("AgeGroup")["Deaths"]
    p = pop_mid_age[pop_mid_age["Region"]==region].set_index("AgeGroup")["Population"]
    # alinhar aos grupos permitidos:
    d = d.reindex(std_df["AgeGroup"]).fillna(0)
    p = p.reindex(std_df["AgeGroup"]).fillna(0)
    cmp_val = direct_standardization(d, p, std_df)
    cmp_rows.append({"Region": region, "CMP padronizado (óbitos/100.000)": cmp_val})

cmp = pd.DataFrame(cmp_rows)

# Tabelas por sexo e faixa etária (descritivo)
desc_sex = ob.groupby(["Region","Sex"], as_index=False)["Deaths"].sum().rename(columns={"Deaths":"Óbitos"})
desc_age = ob.groupby(["Region","AgeGroup"], as_index=False)["Deaths"].sum().rename(columns={"Deaths":"Óbitos"})

# ------------------------------
# Apresentação
# ------------------------------

left, right = st.columns([1,1])
with left:
    st.subheader("Coeficiente de Mortalidade Bruto (CMB) — Período Selecionado")
    st.markdown(f"Ponto médio do período: **{mid_year}**.")
    st.dataframe(cmb, use_container_width=True)
with right:
    st.subheader("Coeficiente de Mortalidade Padronizado (CMP) — Método Direto")
    if user_std is None:
        st.caption("Padrão: WHO 2000–2025")
    else:
        st.caption("Padrão: arquivo enviado pelo usuário")
    st.dataframe(cmp, use_container_width=True)

st.markdown("---")
st.subheader("Análise Descritiva de Óbitos")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Por Sexo**")
    st.dataframe(desc_sex, use_container_width=True)
with c2:
    st.markdown("**Por Faixa Etária**")
    st.dataframe(desc_age.sort_values(["Region","AgeGroup"]), use_container_width=True)

# ------------------------------
# Tendências (gráficos de linha)
# ------------------------------

st.markdown("---")
st.subheader("Tendências ao longo do tempo")

# Preparar séries anuais: CMB e CMP por ano (rolling period = ano único usando população do próprio ano como mid-year simplificado)
series_rows = []

for region in sel_regions:
    for year in range(sel_year_range[0], sel_year_range[1]+1):
        ob_y = ob[(ob["Region"]==region) & (ob["Year"]==year)]
        pop_y = pop[(pop["Region"]==region) & (pop["Year"]==year)]
        deaths_total = ob_y["Deaths"].sum()
        pop_total = pop_y["Population"].sum()
        cmb_y = (deaths_total / pop_total) * 100000.0 if pop_total>0 else np.nan

        # CMP ano a ano
        d_age = ob_y.groupby("AgeGroup")["Deaths"].sum()
        p_age = pop_y.groupby("AgeGroup")["Population"].sum()
        # reindex to WHO groups
        d_age = d_age.reindex(WHO_STD["AgeGroup"]).fillna(0)
        p_age = p_age.reindex(WHO_STD["AgeGroup"]).fillna(0)
        cmp_y = direct_standardization(d_age, p_age, std_df)
        series_rows.append({"Region":region, "Year":year, "CMB":cmb_y, "CMP":cmp_y})

series = pd.DataFrame(series_rows)

if not series.empty:
    # Gráfico CMB
    fig1, ax1 = plt.subplots()
    for region in sel_regions:
        s = series[series["Region"]==region].sort_values("Year")
        ax1.plot(s["Year"], s["CMB"], label=str(region))
    ax1.set_xlabel("Ano")
    ax1.set_ylabel("CMB (óbitos por 100.000)")
    ax1.set_title("Tendência do CMB")
    ax1.legend()
    st.pyplot(fig1)

    # Gráfico CMP
    fig2, ax2 = plt.subplots()
    for region in sel_regions:
        s = series[series["Region"]==region].sort_values("Year")
        ax2.plot(s["Year"], s["CMP"], label=str(region))
    ax2.set_xlabel("Ano")
    ax2.set_ylabel("CMP (óbitos por 100.000)")
    ax2.set_title("Tendência do CMP (padronizado por idade)")
    ax2.legend()
    st.pyplot(fig2)

# ------------------------------
# Downloads
# ------------------------------

st.markdown("---")
st.subheader("Exportar tabelas")
exp_cmb = cmb.to_csv(index=False).encode("utf-8")
exp_cmp = cmp.to_csv(index=False).encode("utf-8")
exp_desc_sex = desc_sex.to_csv(index=False).encode("utf-8")
exp_desc_age = desc_age.to_csv(index=False).encode("utf-8")

st.download_button("Baixar CMB (CSV)", exp_cmb, file_name="cmb_periodo.csv", mime="text/csv")
st.download_button("Baixar CMP (CSV)", exp_cmp, file_name="cmp_periodo.csv", mime="text/csv")
st.download_button("Baixar Óbitos por Sexo (CSV)", exp_desc_sex, file_name="obitos_por_sexo.csv", mime="text/csv")
st.download_button("Baixar Óbitos por Faixa Etária (CSV)", exp_desc_age, file_name="obitos_por_faixa_etaria.csv", mime="text/csv")

# ------------------------------
# Metodologia e Considerações Éticas
# ------------------------------

with st.expander("📑 Metodologia e Considerações Éticas"):
    st.markdown("""
**Cálculos**  
• **Coeficiente de Mortalidade Bruto (CMB)** = (Óbitos no período / População no ponto médio do período) × 100.000.  
• **Padronização por idade (método direto)**: calculada a partir das taxas específicas por faixa etária aplicadas a uma **população padrão** (WHO 2000–2025 por padrão ou arquivo enviado pelo usuário).  
• **Tendências anuais**: séries de CMB e CMP por ano, com o próprio ano usado como aproximação de ponto médio.

**Organização dos dados**  
Os dados de óbitos e de população devem conter, no mínimo: Região, Ano, Faixa Etária, e (para óbitos) uma contagem de óbitos, (para população) uma contagem populacional. Colunas com nomes diferentes são aceitas — o sistema faz a detecção automática.

**Limitações**  
• A exatidão da padronização depende do alinhamento das faixas etárias com a população padrão. Este painel usa grupos 0–4, 5–9, ..., 80+.  
• Quando algum ano ou faixa não possui dados, as taxas podem ficar instáveis.  
• Modelos de regressão (ex.: Joinpoint) não são aplicados aqui, mas o painel permite exportar as séries para análise externa.

**Aspectos Éticos**  
Trata-se de **dados secundários, públicos e anonimizados**. Conforme a Resolução **CNS nº 510/2016**, o estudo **dispensa** submissão a Comitê de Ética em Pesquisa. Recomenda-se registrar essa informação na seção de Metodologia do TCC.
    """)

