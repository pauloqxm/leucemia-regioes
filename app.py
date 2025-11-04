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
    """
    Tenta ler um CSV com diferentes codificações e separadores
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    separators = [',', ';', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                # Reset do buffer para o início
                if hasattr(path_or_buffer, 'seek'):
                    path_or_buffer.seek(0)
                
                df = pd.read_csv(path_or_buffer, sep=sep, encoding=encoding, **kwargs)
                if not df.empty and len(df.columns) > 1:
                    st.success(f"Arquivo lido com sucesso! Encoding: {encoding}, Separador: '{sep}'")
                    return df
            except Exception as e:
                continue
    
    # Última tentativa com parâmetros padrão
    try:
        if hasattr(path_or_buffer, 'seek'):
            path_or_buffer.seek(0)
        df = pd.read_csv(path_or_buffer)
        st.success("Arquivo lido com parâmetros padrão do pandas")
        return df
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {str(e)}")
        # Tentativa como string literal para debug
        if hasattr(path_or_buffer, 'seek'):
            path_or_buffer.seek(0)
        content = path_or_buffer.read().decode('latin-1')
        st.text_area("Conteúdo do arquivo (primeiras 1000 caracteres):", content[:1000], height=200)
        return pd.DataFrame()

def normalize_col(s):
    if pd.isna(s):
        return ""
    return (
        str(s).strip()
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
    if df.empty:
        return None
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
        st.error(f"Não encontrei a coluna para '{name}'. Colunas disponíveis: {list(df.columns)}")
        return None
    return None

# WHO World Standard Population (2000-2025)
WHO_STD = pd.DataFrame({
    "AgeGroup": [
        "0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
        "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"
    ],
    "StdPop": [
        8800, 8700, 8600, 8500, 8000, 7500, 7000, 6500,
        6000, 5500, 5000, 4000, 2500, 1500, 800, 200, 100
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
    if s in ["80+", "80mais", "80oumais", "80 e mais"]:
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

# Mostrar informações dos arquivos
st.sidebar.subheader("Informações dos Arquivos")
if uploaded_obitos:
    st.sidebar.write(f"Óbitos: {uploaded_obitos.name} ({uploaded_obitos.size} bytes)")
if uploaded_pop:
    st.sidebar.write(f"População: {uploaded_pop.name} ({uploaded_pop.size} bytes)")

# Carregar dados com tratamento de erro melhorado
with st.spinner("Carregando arquivo de óbitos..."):
    df_ob = read_csv_safely(uploaded_obitos)

with st.spinner("Carregando arquivo de população..."):
    df_pop = read_csv_safely(uploaded_pop)

# Verificar se os DataFrames não estão vazios
if df_ob.empty or df_pop.empty:
    st.error("Erro ao carregar os arquivos. Verifique o formato dos arquivos CSV.")
    st.stop()

# Mostrar preview dos dados
st.sidebar.subheader("Preview dos Dados")
if st.sidebar.checkbox("Mostrar preview dos dados carregados"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dados de Óbitos:")
        st.dataframe(df_ob.head(3))
    with col2:
        st.write("Dados de População:")
        st.dataframe(df_pop.head(3))

# ------------------------------
# Mapeamento de colunas
# ------------------------------

st.subheader("Mapeamento de Colunas")

cols_ob = {
    "region": coalesce_columns(df_ob, "região", [["regiao","região","uf","region","estado","uf_regiao"]]),
    "year": coalesce_columns(df_ob, "ano", [["ano","year","periodo","anodeobito","ano_obito"]]),
    "sex": coalesce_columns(df_ob, "sexo", [["sexo","sex","genero","sexobiologico"]], required=False),
    "age": coalesce_columns(df_ob, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup","faixaetaria","faixa_etaria"]]),
    "deaths": coalesce_columns(df_ob, "óbitos", [["obitos","mortes","deaths","numeroobitos","n_obitos"]]),
}

cols_pop = {
    "region": coalesce_columns(df_pop, "região", [["regiao","região","uf","region","estado","uf_regiao"]]),
    "year": coalesce_columns(df_pop, "ano", [["ano","year","periodo","anoreferencia"]]),
    "age": coalesce_columns(df_pop, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup","faixaetaria","faixa_etaria"]]),
    "population": coalesce_columns(df_pop, "população", [["populacao","population","habitantes","pop","populacaoresidente"]]),
}

# Verificar se todas as colunas obrigatórias foram encontradas
required_ob_columns = ["region", "year", "age", "deaths"]
required_pop_columns = ["region", "year", "age", "population"]

missing_columns = []
for col in required_ob_columns:
    if cols_ob[col] is None:
        missing_columns.append(f"Óbitos: {col}")

for col in required_pop_columns:
    if cols_pop[col] is None:
        missing_columns.append(f"População: {col}")

if missing_columns:
    st.error("Colunas obrigatórias não encontradas:")
    for missing in missing_columns:
        st.write(f"- {missing}")
    st.write("Colunas disponíveis no arquivo de Óbitos:", list(df_ob.columns))
    st.write("Colunas disponíveis no arquivo de População:", list(df_pop.columns))
    st.stop()

# Renomear colunas
df_ob = df_ob.rename(columns={
    cols_ob["region"]: "Region", 
    cols_ob["year"]: "Year", 
    cols_ob["age"]: "AgeGroup", 
    cols_ob["deaths"]: "Deaths"
})

df_pop = df_pop.rename(columns={
    cols_pop["region"]: "Region", 
    cols_pop["year"]: "Year", 
    cols_pop["age"]: "AgeGroup", 
    cols_pop["population"]: "Population"
})

# Adicionar coluna Sex se não existir
if cols_ob["sex"]:
    df_ob = df_ob.rename(columns={cols_ob["sex"]: "Sex"})
else:
    df_ob["Sex"] = "Todos"

# Processar dados
df_ob["AgeGroup"] = df_ob["AgeGroup"].astype(str).map(harmonize_age_group)
df_pop["AgeGroup"] = df_pop["AgeGroup"].astype(str).map(harmonize_age_group)

# Converter colunas numéricas
df_ob["Deaths"] = pd.to_numeric(df_ob["Deaths"], errors='coerce').fillna(0)
df_pop["Population"] = pd.to_numeric(df_pop["Population"], errors='coerce').fillna(0)
df_ob["Year"] = pd.to_numeric(df_ob["Year"], errors='coerce').dropna().astype(int)
df_pop["Year"] = pd.to_numeric(df_pop["Year"], errors='coerce').dropna().astype(int)

# ------------------------------
# Filtros
# ------------------------------

st.sidebar.subheader("Filtros")

regions = sorted(df_ob["Region"].dropna().unique().tolist())
years_ob = sorted(df_ob["Year"].dropna().unique().tolist())
years_pop = sorted(df_pop["Year"].dropna().unique().tolist())

available_years = sorted(list(set(years_ob) & set(years_pop)))
if not available_years:
    st.error("Não há anos comuns entre os datasets de óbitos e população.")
    st.stop()

sel_regions = st.sidebar.multiselect("Regiões", regions, default=regions[:2] if len(regions) >= 2 else regions)
sel_sex = st.sidebar.multiselect("Sexo", sorted(df_ob["Sex"].dropna().unique().tolist()), 
                               default=sorted(df_ob["Sex"].dropna().unique().tolist()))

sel_year_range = st.sidebar.slider("Período (anos)", 
                                  min_value=int(min(available_years)), 
                                  max_value=int(max(available_years)), 
                                  value=(int(min(available_years)), int(max(available_years))), 
                                  step=1)

std_choice = st.sidebar.selectbox("População Padrão", ["WHO 2000-2025 (OMS)", "Arquivo CSV (AgeGroup, StdPop)"])
std_df = WHO_STD.copy()

if std_choice == "Arquivo CSV (AgeGroup, StdPop)":
    up = st.sidebar.file_uploader("População Padrão (CSV)", type=["csv"], key="stdpop")
    if up is not None:
        tmp = read_csv_safely(up)
        cand_age = find_col(tmp, ["agegroup","faixa etaria"])
        cand_std = find_col(tmp, ["stdpop","populacao padrao"])
        if cand_age and cand_std:
            tmp = tmp.rename(columns={cand_age: "AgeGroup", cand_std: "StdPop"})[["AgeGroup", "StdPop"]]
            tmp["AgeGroup"] = tmp["AgeGroup"].astype(str).map(harmonize_age_group)
            tmp["StdPop"] = pd.to_numeric(tmp["StdPop"], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                std_df = tmp

# ------------------------------
# Cálculos
# ------------------------------

# Aplicar filtros
mask_ob = (df_ob["Region"].isin(sel_regions)) & (df_ob["Sex"].isin(sel_sex)) & (df_ob["Year"].between(sel_year_range[0], sel_year_range[1]))
mask_pop = (df_pop["Region"].isin(sel_regions)) & (df_pop["Year"].between(sel_year_range[0], sel_year_range[1]))

ob = df_ob.loc[mask_ob].copy()
pop = df_pop.loc[mask_pop].copy()

if ob.empty or pop.empty:
    st.error("Não há dados disponíveis para os filtros selecionados.")
    st.stop()

# Cálculo do CMB
mid_year = (sel_year_range[0] + sel_year_range[1]) // 2
pop_mid = pop[pop["Year"] == mid_year].groupby(["Region"], as_index=False)["Population"].sum()
deaths_period = ob.groupby(["Region"], as_index=False)["Deaths"].sum()

cmb = pd.merge(deaths_period, pop_mid, on="Region", how="left")
cmb["CMB (óbitos/100.000)"] = np.where(
    cmb["Population"] > 0, 
    (cmb["Deaths"] / cmb["Population"]) * 100000.0, 
    np.nan
)

# Cálculo do CMP
deaths_age = ob.groupby(["Region", "AgeGroup"], as_index=False)["Deaths"].sum()
pop_mid_age = pop[pop["Year"] == mid_year].groupby(["Region", "AgeGroup"], as_index=False)["Population"].sum()

cmp_rows = []
for region in sorted(deaths_age["Region"].unique().tolist()):
    d = deaths_age[deaths_age["Region"] == region].set_index("AgeGroup")["Deaths"]
    p = pop_mid_age[pop_mid_age["Region"] == region].set_index("AgeGroup")["Population"]
    
    # Garantir alinhamento com std_df
    index_std = std_df["AgeGroup"].values
    d_aligned = d.reindex(index_std).fillna(0)
    p_aligned = p.reindex(index_std).fillna(0)
    
    cmp_val = direct_standardization(d_aligned, p_aligned, std_df)
    cmp_rows.append({"Region": region, "CMP (óbitos/100.000)": cmp_val})

cmp = pd.DataFrame(cmp_rows)

# ------------------------------
# Exibição
# ------------------------------

st.subheader("Coeficiente de Mortalidade Bruto (CMB)")
st.dataframe(cmb, use_container_width=True)

st.subheader("Coeficiente de Mortalidade Padronizado (CMP)")
st.dataframe(cmp, use_container_width=True)

# Visualizações gráficas
st.subheader("Visualização Gráfica")

col1, col2 = st.columns(2)

with col1:
    if not cmb.empty and not cmb["CMB (óbitos/100.000)"].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_data = cmb.dropna(subset=["CMB (óbitos/100.000)"])
        if not valid_data.empty:
            bars = ax.bar(valid_data["Region"], valid_data["CMB (óbitos/100.000)"])
            ax.set_title("Coeficiente de Mortalidade Bruto (CMB)")
            ax.set_ylabel("Óbitos por 100.000 habitantes")
            plt.xticks(rotation=45)
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom')
            st.pyplot(fig)

with col2:
    if not cmp.empty and not cmp["CMP (óbitos/100.000)"].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_data = cmp.dropna(subset=["CMP (óbitos/100.000)"])
        if not valid_data.empty:
            bars = ax.bar(valid_data["Region"], valid_data["CMP (óbitos/100.000)"])
            ax.set_title("Coeficiente de Mortalidade Padronizado (CMP)")
            ax.set_ylabel("Óbitos por 100.000 habitantes")
            plt.xticks(rotation=45)
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom')
            st.pyplot(fig)

st.markdown("---")
st.subheader("Considerações Éticas")
st.markdown("""
Trata-se de **dados públicos e anonimizados**.  
De acordo com a **Resolução CNS nº 510/2016**, o estudo dispensa submissão a Comitê de Ética em Pesquisa.
""")
