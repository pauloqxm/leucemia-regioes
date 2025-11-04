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
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
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
        return pd.DataFrame()

def normalize_col(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    # Primeiro, tentar normalizar caracteres especiais
    replacements = {
        'ã': 'a', 'â': 'a', 'á': 'a', 'à': 'a',
        'ê': 'e', 'é': 'e', 'è': 'e',
        'î': 'i', 'í': 'i', 'ì': 'i',
        'õ': 'o', 'ô': 'o', 'ó': 'o', 'ò': 'o',
        'û': 'u', 'ú': 'u', 'ù': 'u',
        'ç': 'c', 'þ': 'c', 'Þ': 'c',  # Tratamento específico para o caractere problemático
        '  ': ' ', '  ': ' '
    }
    
    result = ""
    for char in s.lower():
        result += replacements.get(char, char)
    
    return result

def find_col(df, candidates):
    if df.empty:
        return None
    
    # Primeiro: busca exata após normalização
    cols_norm = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        nc = normalize_col(cand)
        if nc in cols_norm:
            return cols_norm[nc]
    
    # Segundo: busca por substring
    for col in df.columns:
        col_norm = normalize_col(col)
        for cand in candidates:
            if normalize_col(cand) in col_norm:
                return col
    
    # Terceiro: busca por tokens
    for col in df.columns:
        col_norm = normalize_col(col)
        col_tokens = set(col_norm.split())
        for cand in candidates:
            cand_tokens = set(normalize_col(cand).split())
            if cand_tokens.intersection(col_tokens):
                return col
    
    return None

def coalesce_columns(df, name, candidate_lists, required=True):
    for candidates in candidate_lists:
        col = find_col(df, candidates)
        if col is not None:
            st.success(f"Coluna '{name}' encontrada: '{col}'")
            return col
    
    if required:
        st.error(f"Não encontrei a coluna para '{name}'. Colunas disponíveis: {list(df.columns)}")
        st.info("Tentando identificar colunas automaticamente...")
        
        # Tentativa automática baseada no conteúdo
        if name == "região":
            # Procurar por colunas que contenham valores como "Nordeste", "Sudeste", etc.
            region_keywords = ['nordeste', 'sudeste', 'norte', 'sul', 'centro', 'regiao', 'uf']
            for col in df.columns:
                col_norm = normalize_col(col)
                if any(keyword in col_norm for keyword in region_keywords):
                    st.success(f"Coluna '{name}' identificada automaticamente: '{col}'")
                    return col
        
        elif name == "faixa etária":
            # Procurar por colunas que contenham faixas etárias
            age_keywords = ['idade', 'faixa', 'ano', 'anos', 'etaria', 'group']
            for col in df.columns:
                col_norm = normalize_col(col)
                if any(keyword in col_norm for keyword in age_keywords):
                    st.success(f"Coluna '{name}' identificada automaticamente: '{col}'")
                    return col
        
        elif name == "óbitos":
            # Procurar por colunas numéricas que possam ser óbitos
            death_keywords = ['obito', 'morte', 'death', 'numero', 'total', 'quantidade']
            for col in df.columns:
                col_norm = normalize_col(col)
                if any(keyword in col_norm for keyword in death_keywords):
                    # Verificar se a coluna tem valores numéricos
                    if pd.to_numeric(df[col], errors='coerce').notna().any():
                        st.success(f"Coluna '{name}' identificada automaticamente: '{col}'")
                        return col
        
        elif name == "população":
            # Procurar por colunas numéricas que possam ser população
            pop_keywords = ['populacao', 'population', 'habitante', 'residente']
            for col in df.columns:
                col_norm = normalize_col(col)
                if any(keyword in col_norm for keyword in pop_keywords):
                    # Verificar se a coluna tem valores numéricos
                    if pd.to_numeric(df[col], errors='coerce').notna().any():
                        st.success(f"Coluna '{name}' identificada automaticamente: '{col}'")
                        return col
        
        elif name == "ano":
            # Procurar por colunas que contenham anos
            year_keywords = ['ano', 'year', 'periodo', 'data']
            for col in df.columns:
                col_norm = normalize_col(col)
                if any(keyword in col_norm for keyword in year_keywords):
                    # Verificar se a coluna tem valores que parecem anos
                    sample_values = df[col].dropna().head(10).astype(str)
                    if any(len(str(val)) == 4 and str(val).isdigit() for val in sample_values):
                        st.success(f"Coluna '{name}' identificada automaticamente: '{col}'")
                        return col
        
        return None
    return None

# WHO World Standard Population (2000-2025)
WHO_STD = pd.DataFrame({
    "AgeGroup": [
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
        "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
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
    s = s.replace(" a ", "-").replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace(" ", "")
    
    # Mapeamento específico para os grupos do seu arquivo
    age_mapping = {
        "menor1ano": "0-1",
        "menor 1 ano": "0-1", 
        "1a4anos": "1-4",
        "1 a 4 anos": "1-4",
        "5a9anos": "5-9", 
        "5 a 9 anos": "5-9",
        "10a14anos": "10-14",
        "10 a 14 anos": "10-14",
        "15a19anos": "15-19",
        "15 a 19 anos": "15-19",
        "20a29anos": "20-29", 
        "20 a 29 anos": "20-29",
        "30a39anos": "30-39",
        "30 a 39 anos": "30-39",
        "40a49anos": "40-49",
        "40 a 49 anos": "40-49",
        "50a59anos": "50-59",
        "50 a 59 anos": "50-59",
        "60a69anos": "60-69",
        "60 a 69 anos": "60-69",
        "70a79anos": "70-79",
        "70 a 79 anos": "70-79",
        "80anose mais": "80+",
        "80 anos e mais": "80+",
        "80mais": "80+",
        "idadeignorada": "ignorado",
        "idade ignorada": "ignorado"
    }
    
    s_normalized = s.lower().replace(" ", "")
    if s_normalized in age_mapping:
        return age_mapping[s_normalized]
    
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

# Carregar dados
with st.spinner("Carregando arquivo de óbitos..."):
    df_ob = read_csv_safely(uploaded_obitos)

with st.spinner("Carregando arquivo de população..."):
    df_pop = read_csv_safely(uploaded_pop)

if df_ob.empty or df_pop.empty:
    st.error("Erro ao carregar os arquivos. Verifique o formato dos arquivos CSV.")
    st.stop()

# ------------------------------
# Mapeamento de colunas - CORRIGIDO
# ------------------------------

st.subheader("Mapeamento de Colunas")

# Para o arquivo de óbitos com a estrutura que você mostrou
st.info("Analisando a estrutura do arquivo de óbitos...")

# Baseado nas colunas que você mostrou: ['RegiaÞo', 'ClassificaçaÞo', 'Ano', 'Menor 1 ano', ...]
# Vamos mapear manualmente baseado no padrão observado

cols_ob = {
    "region": "RegiaÞo",  # Coluna de região
    "year": "Ano",        # Coluna de ano
    "age": None,          # Não há uma coluna única de faixa etária - as faixas estão nas colunas
    "deaths": None,       # As mortes estão distribuídas por colunas de faixa etária
}

cols_pop = {
    "region": None,
    "year": None, 
    "age": None,
    "population": None,
}

# Verificar se podemos usar o mapeamento manual
if "RegiaÞo" in df_ob.columns and "Ano" in df_ob.columns:
    st.success("✅ Estrutura do arquivo de óbitos identificada!")
    
    # Lista de colunas que são faixas etárias (excluindo colunas de metadados)
    age_columns = [col for col in df_ob.columns if col not in ['RegiaÞo', 'ClassificaçaÞo', 'Ano', 'Total', 'Idade ignorada']]
    
    st.write(f"**Colunas de faixa etária identificadas:** {age_columns}")
    
    # Transformar o formato wide para long
    df_ob_long = pd.melt(
        df_ob, 
        id_vars=['RegiaÞo', 'Ano'],
        value_vars=age_columns,
        var_name='AgeGroup',
        value_name='Deaths'
    )
    
    # Renomear colunas
    df_ob_long = df_ob_long.rename(columns={
        'RegiaÞo': 'Region',
        'Ano': 'Year'
    })
    
    # Processar faixas etárias
    df_ob_long['AgeGroup'] = df_ob_long['AgeGroup'].map(harmonize_age_group)
    df_ob_long['Deaths'] = pd.to_numeric(df_ob_long['Deaths'], errors='coerce').fillna(0)
    df_ob_long['Year'] = pd.to_numeric(df_ob_long['Year'], errors='coerce').dropna().astype(int)
    
    df_ob = df_ob_long
    st.success("✅ Dados de óbitos transformados para formato longo!")
    
else:
    # Se o mapeamento automático falhar, tentar o método original
    cols_ob = {
        "region": coalesce_columns(df_ob, "região", [["regiao","região","uf","region","estado","uf_regiao"]]),
        "year": coalesce_columns(df_ob, "ano", [["ano","year","periodo","anodeobito","ano_obito"]]),
        "age": coalesce_columns(df_ob, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup","faixaetaria","faixa_etaria"]]),
        "deaths": coalesce_columns(df_ob, "óbitos", [["obitos","mortes","deaths","numeroobitos","n_obitos"]]),
    }
    
    # Renomear colunas se encontradas
    if all(cols_ob.values()):
        df_ob = df_ob.rename(columns={
            cols_ob["region"]: "Region", 
            cols_ob["year"]: "Year", 
            cols_ob["age"]: "AgeGroup", 
            cols_ob["deaths"]: "Deaths"
        })

# Processar arquivo de população da mesma forma
st.info("Analisando a estrutura do arquivo de população...")

if "RegiaÞo" in df_pop.columns and "Ano" in df_pop.columns:
    st.success("✅ Estrutura do arquivo de população identificada!")
    
    # Lista de colunas que são faixas etárias
    age_columns_pop = [col for col in df_pop.columns if col not in ['RegiaÞo', 'ClassificaçaÞo', 'Ano', 'Total', 'Idade ignorada']]
    
    st.write(f"**Colunas de faixa etária identificadas:** {age_columns_pop}")
    
    # Transformar o formato wide para long
    df_pop_long = pd.melt(
        df_pop, 
        id_vars=['RegiaÞo', 'Ano'],
        value_vars=age_columns_pop,
        var_name='AgeGroup',
        value_name='Population'
    )
    
    # Renomear colunas
    df_pop_long = df_pop_long.rename(columns={
        'RegiaÞo': 'Region',
        'Ano': 'Year'
    })
    
    # Processar faixas etárias
    df_pop_long['AgeGroup'] = df_pop_long['AgeGroup'].map(harmonize_age_group)
    df_pop_long['Population'] = pd.to_numeric(df_pop_long['Population'], errors='coerce').fillna(0)
    df_pop_long['Year'] = pd.to_numeric(df_pop_long['Year'], errors='coerce').dropna().astype(int)
    
    df_pop = df_pop_long
    st.success("✅ Dados de população transformados para formato longo!")
    
else:
    # Método original para população
    cols_pop = {
        "region": coalesce_columns(df_pop, "região", [["regiao","região","uf","region","estado","uf_regiao"]]),
        "year": coalesce_columns(df_pop, "ano", [["ano","year","periodo","anoreferencia"]]),
        "age": coalesce_columns(df_pop, "faixa etária", [["faixa etaria","idade","grupo etario","agegroup","faixaetaria","faixa_etaria"]]),
        "population": coalesce_columns(df_pop, "população", [["populacao","population","habitantes","pop","populacaoresidente"]]),
    }
    
    if all(cols_pop.values()):
        df_pop = df_pop.rename(columns={
            cols_pop["region"]: "Region", 
            cols_pop["year"]: "Year", 
            cols_pop["age"]: "AgeGroup", 
            cols_pop["population"]: "Population"
        })

# Adicionar coluna Sex se não existir
if "Sex" not in df_ob.columns:
    df_ob["Sex"] = "Todos"

# Mostrar preview dos dados processados
st.subheader("Preview dos Dados Processados")
col1, col2 = st.columns(2)

with col1:
    st.write("Dados de Óbitos (primeiras 10 linhas):")
    st.dataframe(df_ob.head(10))

with col2:
    st.write("Dados de População (primeiras 10 linhas):")
    st.dataframe(df_pop.head(10))

# Verificar se temos os dados necessários
required_cols_ob = ["Region", "Year", "AgeGroup", "Deaths"]
required_cols_pop = ["Region", "Year", "AgeGroup", "Population"]

missing_ob = [col for col in required_cols_ob if col not in df_ob.columns]
missing_pop = [col for col in required_cols_pop if col not in df_pop.columns]

if missing_ob or missing_pop:
    st.error("Colunas obrigatórias não encontradas após processamento:")
    if missing_ob:
        st.write(f"Óbitos: {missing_ob}")
    if missing_pop:
        st.write(f"População: {missing_pop}")
    st.stop()

# ------------------------------
# Resto do código permanece igual...
# ------------------------------

# [O restante do código dos filtros e cálculos permanece igual...]

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
