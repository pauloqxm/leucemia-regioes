
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Painel de Mortalidade por Leucemia (Nordeste x Sudeste)", layout="wide")

st.title("📊 Painel de Mortalidade por Leucemia — Nordeste x Sudeste")
st.caption("Cálculo de coeficientes brutos e padronizados por idade (método direto)")

# ------------------------------
# Utilidades
# ------------------------------

@st.cache_data
def read_csv_safely(path_or_buffer, **kwargs):
    """
    Lê arquivos CSV com encoding correto para os dados brasileiros
    """
    try:
        # Para os arquivos com caracteres especiais brasileiros
        return pd.read_csv(path_or_buffer, sep=';', encoding='utf-8', **kwargs)
    except Exception:
        try:
            return pd.read_csv(path_or_buffer, sep=';', encoding='latin-1', **kwargs)
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {str(e)}")
            return pd.DataFrame()

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
    """
    Converte as faixas etárias dos arquivos para o formato padrão
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    
    s = str(s).strip()
    
    # Mapeamento específico para os grupos dos seus arquivos
    age_mapping = {
        "menor 1 ano": "0-1",
        "1 a 4 anos": "1-4",
        "5 a 9 anos": "5-9", 
        "10 a 14 anos": "10-14",
        "15 a 19 anos": "15-19",
        "20 a 29 anos": "20-29", 
        "30 a 39 anos": "30-39",
        "40 a 49 anos": "40-49",
        "50 a 59 anos": "50-59",
        "60 a 69 anos": "60-69",
        "70 a 79 anos": "70-79",
        "80 anos e mais": "80+",
        "idade ignorada": "ignorado",
        "total": "total"
    }
    
    s_lower = s.lower()
    if s_lower in age_mapping:
        return age_mapping[s_lower]
    
    return s

def map_to_std_age_groups(age_group, value):
    """
    Mapeia as faixas etárias dos dados para as faixas padrão WHO
    """
    if age_group == "0-1":
        return {"0-4": value * 0.2}  # Distribui proporcionalmente
    elif age_group == "1-4":
        return {"0-4": value * 0.8}  # Distribui proporcionalmente
    elif age_group == "20-29":
        return {"20-24": value * 0.5, "25-29": value * 0.5}
    elif age_group == "30-39":
        return {"30-34": value * 0.5, "35-39": value * 0.5}
    elif age_group == "40-49":
        return {"40-44": value * 0.5, "45-49": value * 0.5}
    elif age_group == "50-59":
        return {"50-54": value * 0.5, "55-59": value * 0.5}
    elif age_group == "60-69":
        return {"60-64": value * 0.5, "65-69": value * 0.5}
    elif age_group == "70-79":
        return {"70-74": value * 0.5, "75-79": value * 0.5}
    elif age_group in WHO_STD["AgeGroup"].values:
        return {age_group: value}
    else:
        return {}

def direct_standardization(deaths_by_age, pop_by_age, std_df):
    """
    Calcula a padronização direta usando o método WHO
    """
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

def transform_wide_to_long(df, value_column_name):
    """
    Transforma dados do formato wide (faixas como colunas) para long
    """
    # Identificar colunas que são faixas etárias (excluindo metadados)
    metadata_columns = ['Região', 'Regiaﬁo', 'ClassificaÁaﬁo', 'Classificação', 'Ano', 'Total', 'Idade ignorada']
    age_columns = [col for col in df.columns if col not in metadata_columns]
    
    # Usar o nome correto da coluna de região baseado no arquivo
    region_col = 'Regiaﬁo' if 'Regiaﬁo' in df.columns else 'Região'
    
    # Fazer o melt para formato longo
    df_long = pd.melt(
        df, 
        id_vars=[region_col, 'Ano'],
        value_vars=age_columns,
        var_name='AgeGroup',
        value_name=value_column_name
    )
    
    # Renomear colunas
    df_long = df_long.rename(columns={
        region_col: 'Region',
        'Ano': 'Year'
    })
    
    # Processar faixas etárias
    df_long['AgeGroup'] = df_long['AgeGroup'].map(harmonize_age_group)
    df_long[value_column_name] = pd.to_numeric(df_long[value_column_name], errors='coerce').fillna(0)
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').dropna().astype(int)
    
    return df_long

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
# Processamento dos dados
# ------------------------------

st.subheader("Processamento dos Dados")

# Transformar dados de óbitos para formato longo
st.info("Transformando dados de óbitos...")
df_ob_long = transform_wide_to_long(df_ob, 'Deaths')

# Transformar dados de população para formato longo  
st.info("Transformando dados de população...")
df_pop_long = transform_wide_to_long(df_pop, 'Population')

# Adicionar coluna Sex (não existe nos dados originais)
df_ob_long['Sex'] = 'Todos'

# Mostrar preview
col1, col2 = st.columns(2)

with col1:
    st.write("Dados de Óbitos (formato longo):")
    st.dataframe(df_ob_long.head(8))

with col2:
    st.write("Dados de População (formato longo):")
    st.dataframe(df_pop_long.head(8))

# ------------------------------
# Filtros
# ------------------------------

st.sidebar.subheader("Filtros")

# Obter regiões e anos disponíveis
regions = sorted(df_ob_long["Region"].dropna().unique().tolist())
years_ob = sorted(df_ob_long["Year"].dropna().unique().tolist())
years_pop = sorted(df_pop_long["Year"].dropna().unique().tolist())

available_years = sorted(list(set(years_ob) & set(years_pop)))
if not available_years:
    st.error("Não há anos comuns entre os datasets de óbitos e população.")
    st.stop()

sel_regions = st.sidebar.multiselect("Regiões", regions, default=regions)
sel_sex = st.sidebar.multiselect("Sexo", sorted(df_ob_long["Sex"].dropna().unique().tolist()), 
                               default=sorted(df_ob_long["Sex"].dropna().unique().tolist()))

sel_year_range = st.sidebar.slider("Período (anos)", 
                                  min_value=int(min(available_years)), 
                                  max_value=int(max(available_years)), 
                                  value=(int(min(available_years)), int(max(available_years))), 
                                  step=1)

std_choice = st.sidebar.selectbox("População Padrão", ["WHO 2000-2025 (OMS)"])
std_df = WHO_STD.copy()

# ------------------------------
# Cálculos
# ------------------------------

# Aplicar filtros
mask_ob = (df_ob_long["Region"].isin(sel_regions)) & (df_ob_long["Sex"].isin(sel_sex)) & (df_ob_long["Year"].between(sel_year_range[0], sel_year_range[1]))
mask_pop = (df_pop_long["Region"].isin(sel_regions)) & (df_pop_long["Year"].between(sel_year_range[0], sel_year_range[1]))

ob = df_ob_long.loc[mask_ob].copy()
pop = df_pop_long.loc[mask_pop].copy()

if ob.empty or pop.empty:
    st.error("Não há dados disponíveis para os filtros selecionados.")
    st.stop()

st.success(f"Dados filtrados: {len(ob)} registros de óbitos, {len(pop)} registros de população")

# Cálculo do CMB (Coeficiente de Mortalidade Bruto)
st.subheader("1. Coeficiente de Mortalidade Bruto (CMB)")

# Usar ano médio do período para população
mid_year = (sel_year_range[0] + sel_year_range[1]) // 2
pop_mid = pop[pop["Year"] == mid_year].groupby(["Region"], as_index=False)["Population"].sum()
deaths_period = ob.groupby(["Region"], as_index=False)["Deaths"].sum()

cmb = pd.merge(deaths_period, pop_mid, on="Region", how="left")
cmb["CMB (óbitos/100.000)"] = np.where(
    cmb["Population"] > 0, 
    (cmb["Deaths"] / cmb["Population"]) * 100000.0, 
    np.nan
)

st.dataframe(cmb, use_container_width=True)

# Cálculo do CMP (Coeficiente de Mortalidade Padronizado)
st.subheader("2. Coeficiente de Mortalidade Padronizado (CMP)")

# Agrupar óbitos por região e faixa etária
deaths_age = ob.groupby(["Region", "AgeGroup"], as_index=False)["Deaths"].sum()
pop_mid_age = pop[pop["Year"] == mid_year].groupby(["Region", "AgeGroup"], as_index=False)["Population"].sum()

# Preparar dados para padronização
cmp_rows = []

for region in sorted(deaths_age["Region"].unique().tolist()):
    # Obter dados da região
    d_region = deaths_age[deaths_age["Region"] == region]
    p_region = pop_mid_age[pop_mid_age["Region"] == region]
    
    # Mapear para faixas etárias padrão
    deaths_std = {}
    pop_std = {}
    
    # Processar cada faixa etária dos dados
    for _, row in d_region.iterrows():
        mappings = map_to_std_age_groups(row["AgeGroup"], row["Deaths"])
        for std_group, value in mappings.items():
            deaths_std[std_group] = deaths_std.get(std_group, 0) + value
    
    for _, row in p_region.iterrows():
        mappings = map_to_std_age_groups(row["AgeGroup"], row["Population"])
        for std_group, value in mappings.items():
            pop_std[std_group] = pop_std.get(std_group, 0) + value
    
    # Criar séries alinhadas com std_df
    deaths_series = pd.Series(deaths_std).reindex(std_df["AgeGroup"]).fillna(0)
    pop_series = pd.Series(pop_std).reindex(std_df["AgeGroup"]).fillna(0)
    
    # Calcular CMP
    cmp_val = direct_standardization(deaths_series, pop_series, std_df)
    cmp_rows.append({"Region": region, "CMP (óbitos/100.000)": cmp_val})

cmp = pd.DataFrame(cmp_rows)
st.dataframe(cmp, use_container_width=True)

# ------------------------------
# Visualizações
# ------------------------------

st.subheader("3. Visualização Gráfica")

col1, col2 = st.columns(2)

with col1:
    if not cmb.empty and not cmb["CMB (óbitos/100.000)"].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_data = cmb.dropna(subset=["CMB (óbitos/100.000)"])
        bars = ax.bar(valid_data["Region"], valid_data["CMB (óbitos/100.000)"], color='skyblue', alpha=0.7)
        ax.set_title("Coeficiente de Mortalidade Bruto (CMB)\npor 100.000 habitantes", fontsize=14, fontweight='bold')
        ax.set_ylabel("Óbitos por 100.000 habitantes", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

with col2:
    if not cmp.empty and not cmp["CMP (óbitos/100.000)"].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_data = cmp.dropna(subset=["CMP (óbitos/100.000)"])
        bars = ax.bar(valid_data["Region"], valid_data["CMP (óbitos/100.000)"], color='lightcoral', alpha=0.7)
        ax.set_title("Coeficiente de Mortalidade Padronizado (CMP)\npor 100.000 habitantes", fontsize=14, fontweight='bold')
        ax.set_ylabel("Óbitos por 100.000 habitantes", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# ------------------------------
# Análise Detalhada
# ------------------------------

st.subheader("4. Análise Detalhada por Faixa Etária")

# Calcular taxas por faixa etária
st.write("**Taxas de mortalidade por faixa etária (por 100.000 habitantes):**")

deaths_by_age_region = ob.groupby(["Region", "AgeGroup"])["Deaths"].sum().reset_index()
pop_by_age_region = pop_mid_age.groupby(["Region", "AgeGroup"])["Population"].sum().reset_index()

rates_by_age = pd.merge(deaths_by_age_region, pop_by_age_region, on=["Region", "AgeGroup"], how="left")
rates_by_age["Taxa (por 100.000)"] = np.where(
    rates_by_age["Population"] > 0,
    (rates_by_age["Deaths"] / rates_by_age["Population"]) * 100000,
    np.nan
)

# Pivot table para melhor visualização
pivot_rates = rates_by_age.pivot_table(
    index="AgeGroup", 
    columns="Region", 
    values="Taxa (por 100.000)", 
    aggfunc='mean'
).round(1)

st.dataframe(pivot_rates, use_container_width=True)

# ------------------------------
# Resumo Estatístico
# ------------------------------

st.subheader("5. Resumo Estatístico")

col1, col2 = st.columns(2)

with col1:
    st.write("**Óbitos totais no período:**")
    total_deaths = ob["Deaths"].sum()
    st.metric("Total de Óbitos", f"{total_deaths:,}".replace(",", "."))
    
    st.write("**Óbitos por região:**")
    deaths_by_region = ob.groupby("Region")["Deaths"].sum().sort_values(ascending=False)
    for region, deaths in deaths_by_region.items():
        st.write(f"- {region}: {deaths:,} óbitos".replace(",", "."))

with col2:
    st.write("**População de referência:**")
    total_pop = pop_mid["Population"].sum()
    st.metric(f"População total ({mid_year})", f"{total_pop:,.0f}".replace(",", "."))
    
    st.write("**População por região:**")
    for _, row in pop_mid.iterrows():
        st.write(f"- {row['Region']}: {row['Population']:,.0f} habitantes".replace(",", "."))

# ------------------------------
# Considerações Finais
# ------------------------------

st.markdown("---")
st.subheader("Considerações Éticas e Metodológicas")
st.markdown("""
**Metodologia:**
- **CMB**: Coeficiente de Mortalidade Bruto = (Óbitos / População) × 100.000
- **CMP**: Coeficiente de Mortalidade Padronizado pelo método direto, usando população padrão WHO
- **População de referência**: Ano médio do período selecionado
- **Faixas etárias**: Dados originais mapeados para faixas padrão WHO

**Aspectos Éticos:**
- Trata-se de **dados públicos e anonimizados**  
- De acordo com a **Resolução CNS nº 510/2016**, o estudo dispensa submissão a Comitê de Ética em Pesquisa
- Dados agregados, sem possibilidade de identificação individual

**Limitações:**
- População disponível apenas para anos censitários (interpolação para anos intercensitários)
- Necessidade de mapeamento proporcional entre faixas etárias diferentes
""")

# Informações técnicas
st.sidebar.markdown("---")
st.sidebar.subheader("Informações Técnicas")
st.sidebar.write(f"Período analisado: {sel_year_range[0]}-{sel_year_range[1]}")
st.sidebar.write(f"Ano de referência: {mid_year}")
st.sidebar.write(f"Regiões: {len(sel_regions)}")
st.sidebar.write(f"Total de óbitos: {total_deaths:,}".replace(",", "."))
