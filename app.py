import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(page_title="Análise de Mortalidade por Leucemia", layout="wide")

# Título principal
st.title("📊 Análise de Mortalidade por Leucemia - Nordeste vs Sudeste")
st.markdown("---")

@st.cache_data
def carregar_dados():
    """Carrega e prepara os dados dos arquivos CSV"""
    
    # Carregar dados de óbitos
    obitos_df = pd.read_csv("Obitos_regiões.csv")
    
    # Carregar dados populacionais regionais
    pop_regioes_df = pd.read_csv("Pop_regiões.csv")
    
    # Carregar população padrão do Brasil
    pop_br_df = pd.read_csv("Pop_BR.csv")
    
    return obitos_df, pop_regioes_df, pop_br_df

# Carregar dados
obitos_df, pop_regioes_df, pop_br_df = carregar_dados()

# Sidebar para navegação
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Selecione a análise:", 
                         ["Visão Geral", 
                          "Coeficiente de Mortalidade Bruto", 
                          "Taxa de Mortalidade por Idade",
                          "Padronização por Idade",
                          "Análise de Tendência"])

# Funções de cálculo
def calcular_cmb(obitos_df, pop_regioes_df):
    """Calcula o Coeficiente de Mortalidade Bruto"""
    
    # Agrupar óbitos totais por região e ano
    obitos_totais = obitos_df.groupby(['Região', 'Ano'])['Obitos'].sum().reset_index()
    obitos_totais.rename(columns={'Obitos': 'Obitos_Totais'}, inplace=True)
    
    # Calcular população total por região e ano
    pop_total = pop_regioes_df.groupby(['Região', 'Ano'])['População'].sum().reset_index()
    pop_total.rename(columns={'População': 'Pop_Total'}, inplace=True)
    
    # Combinar dados
    cmb_df = pd.merge(obitos_totais, pop_total, on=['Região', 'Ano'])
    
    # Calcular CMB
    cmb_df['CMB'] = (cmb_df['Obitos_Totais'] / cmb_df['Pop_Total']) * 100000
    
    return cmb_df

def calcular_tmi(obitos_df, pop_regioes_df):
    """Calcula a Taxa de Mortalidade por Idade (TMI)"""
    
    # Combinar dados de óbitos e população por faixa etária
    tmi_df = pd.merge(obitos_df, pop_regioes_df, 
                     on=['Região', 'Ano', 'Faixa_Etaria'])
    
    # Calcular TMI
    tmi_df['TMI'] = (tmi_df['Obitos'] / tmi_df['População']) * 100000
    
    return tmi_df

def padronizar_mortalidade(tmi_df, pop_br_df):
    """Realiza padronização direta por idade"""
    
    # Preparar população padrão
    pop_padrao = pop_br_df.copy()
    pop_padrao.rename(columns={'Pop_Padrao_BR_2010': 'Pop_Padrao'}, inplace=True)
    
    # Combinar TMI com população padrão
    padronizado_df = pd.merge(tmi_df, pop_padrao, on='Faixa_Etaria')
    
    # Calcular óbitos esperados
    padronizado_df['Obitos_Esperados'] = (padronizado_df['TMI'] * padronizado_df['Pop_Padrao']) / 100000
    
    # Calcular taxa padronizada
    padronizado_agg = padronizado_df.groupby(['Região', 'Ano']).agg({
        'Obitos_Esperados': 'sum',
        'Pop_Padrao': 'sum'
    }).reset_index()
    
    padronizado_agg['Taxa_Padronizada'] = (padronizado_agg['Obitos_Esperados'] / padronizado_agg['Pop_Padrao']) * 100000
    
    return padronizado_agg

def analise_tendencia(cmb_df, padronizado_df):
    """Realiza análise de tendência usando regressão linear"""
    
    resultados = {}
    
    for regiao in ['Nordeste', 'Sudeste']:
        # Dados CMB
        dados_cmb = cmb_df[cmb_df['Região'] == regiao].copy()
        dados_cmb = dados_cmb.sort_values('Ano')
        
        # Dados padronizados
        dados_pad = padronizado_df[padronizado_df['Região'] == regiao].copy()
        dados_pad = dados_pad.sort_values('Ano')
        
        # Regressão para CMB
        if len(dados_cmb) > 1:
            slope_cmb, intercept_cmb, r_value_cmb, p_value_cmb, std_err_cmb = stats.linregress(
                dados_cmb['Ano'], dados_cmb['CMB']
            )
        else:
            slope_cmb = intercept_cmb = r_value_cmb = p_value_cmb = std_err_cmb = np.nan
        
        # Regressão para taxas padronizadas
        if len(dados_pad) > 1:
            slope_pad, intercept_pad, r_value_pad, p_value_pad, std_err_pad = stats.linregress(
                dados_pad['Ano'], dados_pad['Taxa_Padronizada']
            )
        else:
            slope_pad = intercept_pad = r_value_pad = p_value_pad = std_err_pad = np.nan
        
        resultados[regiao] = {
            'CMB': {
                'slope': slope_cmb,
                'intercept': intercept_cmb,
                'r_squared': r_value_cmb**2,
                'p_value': p_value_cmb,
                'std_err': std_err_cmb
            },
            'Padronizada': {
                'slope': slope_pad,
                'intercept': intercept_pad,
                'r_squared': r_value_pad**2,
                'p_value': p_value_pad,
                'std_err': std_err_pad
            }
        }
    
    return resultados

# Calcular métricas
cmb_df = calcular_cmb(obitos_df, pop_regioes_df)
tmi_df = calcular_tmi(obitos_df, pop_regioes_df)
padronizado_df = padronizar_mortalidade(tmi_df, pop_br_df)
tendencia_resultados = analise_tendencia(cmb_df, padronizado_df)

# Página: Visão Geral
if pagina == "Visão Geral":
    st.header("📈 Visão Geral dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dados de Óbitos")
        st.dataframe(obitos_df.head(10), use_container_width=True)
        st.write(f"Total de registros de óbitos: {len(obitos_df):,}")
        st.write(f"Período: {obitos_df['Ano'].min()} - {obitos_df['Ano'].max()}")
        
        # Estatísticas descritivas
        st.subheader("Estatísticas Descritivas - Óbitos")
        st.write(obitos_df.groupby('Região')['Obitos'].describe())
    
    with col2:
        st.subheader("Dados Populacionais")
        st.dataframe(pop_regioes_df.head(10), use_container_width=True)
        st.write(f"Total de registros populacionais: {len(pop_regioes_df):,}")
        st.write(f"Anos disponíveis: {sorted(pop_regioes_df['Ano'].unique())}")
        
        st.subheader("População Padrão Brasil 2010")
        st.dataframe(pop_br_df, use_container_width=True)
    
    # Gráfico de óbitos totais por ano e região
    st.subheader("Evolução dos Óbitos Totais por Leucemia")
    
    obitos_totais_ano = obitos_df.groupby(['Ano', 'Região'])['Obitos'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for regiao in obitos_totais_ano['Região'].unique():
        dados_regiao = obitos_totais_ano[obitos_totais_ano['Região'] == regiao]
        ax.plot(dados_regiao['Ano'], dados_regiao['Obitos'], 
                marker='o', label=regiao, linewidth=2)
    
    ax.set_xlabel('Ano')
    ax.set_ylabel('Número de Óbitos')
    ax.set_title('Evolução dos Óbitos por Leucemia (1979-2022)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Distribuição por faixa etária
    st.subheader("Distribuição de Óbitos por Faixa Etária")
    
    obitos_faixa = obitos_df.groupby(['Faixa_Etaria', 'Região'])['Obitos'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ordem_faixas = [
        'Menor 1 ano', '1 a 4 anos', '5 a 9 anos', '10 a 14 anos', 
        '15 a 19 anos', '20 a 29 anos', '30 a 39 anos', '40 a 49 anos',
        '50 a 59 anos', '60 a 69 anos', '70 a 79 anos', '80 anos e mais'
    ]
    
    obitos_faixa_pivot = obitos_faixa.pivot_table(values='Obitos', index='Faixa_Etaria', columns='Região').reindex(ordem_faixas)
    
    obitos_faixa_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Faixa Etária')
    ax.set_ylabel('Total de Óbitos')
    ax.set_title('Distribuição de Óbitos por Faixa Etária (1979-2022)')
    ax.legend(title='Região')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Página: Coeficiente de Mortalidade Bruto
elif pagina == "Coeficiente de Mortalidade Bruto":
    st.header("📊 Coeficiente de Mortalidade Bruto (CMB)")
    st.latex(r"CMB = \left( \frac{\text{Número total de óbitos por leucemia no período}}{\text{População total da região no ponto médio do período}} \right) \times 100.000")
    
    st.info("💡 **Nota:** Os dados populacionais estão disponíveis apenas para os anos 1980, 1991, 2000 e 2010. O CMB é calculado apenas para esses anos.")
    
    # Tabela CMB
    st.subheader("Tabela - Coeficiente de Mortalidade Bruto")
    cmb_pivot = cmb_df.pivot_table(values='CMB', index='Ano', columns='Região').reset_index()
    st.dataframe(cmb_pivot.round(2), use_container_width=True)
    
    # Gráfico CMB
    st.subheader("Evolução do Coeficiente de Mortalidade Bruto")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for regiao in cmb_df['Região'].unique():
        dados_regiao = cmb_df[cmb_df['Região'] == regiao]
        ax.plot(dados_regiao['Ano'], dados_regiao['CMB'], 
                marker='o', label=regiao, linewidth=2, markersize=8)
    
    ax.set_xlabel('Ano')
    ax.set_ylabel('CMB (óbitos por 100.000 habitantes)')
    ax.set_title('Coeficiente de Mortalidade Bruto por Leucemia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Análise comparativa
    st.subheader("Análise Comparativa")
    col1, col2 = st.columns(2)
    
    with col1:
        cmb_medio = cmb_df.groupby('Região')['CMB'].mean().round(2)
        st.metric("CMB Médio - Nordeste", f"{cmb_medio['Nordeste']}")
        st.metric("CMB Médio - Sudeste", f"{cmb_medio['Sudeste']}")
    
    with col2:
        cmb_ultimo = cmb_df[cmb_df['Ano'] == 2010].set_index('Região')['CMB']
        st.metric("CMB 2010 - Nordeste", f"{cmb_ultimo['Nordeste']:.2f}")
        st.metric("CMB 2010 - Sudeste", f"{cmb_ultimo['Sudeste']:.2f}")

# Página: Taxa de Mortalidade por Idade
elif pagina == "Taxa de Mortalidade por Idade":
    st.header("👥 Taxa de Mortalidade por Idade (TMI)")
    st.latex(r"TMI_i = \left( \frac{\text{Número total de óbitos na faixa etária i}}{\text{População total na faixa etária i}} \right) \times 100.000")
    
    st.info("💡 **Nota:** Os dados populacionais estão disponíveis apenas para os anos 1980, 1991, 2000 e 2010.")
    
    # Selecionar ano para análise
    ano_selecionado = st.selectbox("Selecione o ano para análise:", sorted(tmi_df['Ano'].unique()))
    
    # Filtrar dados
    tmi_filtrado = tmi_df[tmi_df['Ano'] == ano_selecionado]
    
    # Gráfico de TMI por faixa etária
    st.subheader(f"Taxa de Mortalidade por Idade - {ano_selecionado}")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Ordem das faixas etárias
    ordem_faixas = [
        'Menor 1 ano', '1 a 4 anos', '5 a 9 anos', '10 a 14 anos', 
        '15 a 19 anos', '20 a 29 anos', '30 a 39 anos', '40 a 49 anos',
        '50 a 59 anos', '60 a 69 anos', '70 a 79 anos', '80 anos e mais'
    ]
    
    for regiao in tmi_filtrado['Região'].unique():
        dados_regiao = tmi_filtrado[tmi_filtrado['Região'] == regiao]
        dados_regiao = dados_regiao.set_index('Faixa_Etaria').reindex(ordem_faixas).reset_index()
        ax.plot(dados_regiao['Faixa_Etaria'], dados_regiao['TMI'], 
                marker='o', label=regiao, linewidth=2, markersize=6)
    
    ax.set_xlabel('Faixa Etária')
    ax.set_ylabel('TMI (óbitos por 100.000 habitantes)')
    ax.set_title(f'Taxa de Mortalidade por Idade - {ano_selecionado}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Tabela de TMI
    st.subheader("Tabela - Taxas de Mortalidade por Idade")
    tmi_pivot = tmi_filtrado.pivot_table(values='TMI', index='Faixa_Etaria', columns='Região').reindex(ordem_faixas)
    st.dataframe(tmi_pivot.round(2), use_container_width=True)

# Página: Padronização por Idade
elif pagina == "Padronização por Idade":
    st.header("⚖️ Padronização por Idade")
    st.markdown("""
    **Método Direto de Padronização:**
    - Usa a população padrão do Brasil (2010)
    - Elimina o efeito das diferenças na estrutura etária
    - Permite comparação mais justa entre regiões
    """)
    
    # Tabela comparativa
    st.subheader("Comparação: CMB vs Taxa Padronizada")
    
    comparativo_df = pd.merge(
        cmb_df[['Região', 'Ano', 'CMB']],
        padronizado_df[['Região', 'Ano', 'Taxa_Padronizada']],
        on=['Região', 'Ano']
    )
    
    st.dataframe(comparativo_df.round(2), use_container_width=True)
    
    # Gráfico comparativo
    st.subheader("Evolução: CMB vs Taxa Padronizada")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # CMB
    for regiao in comparativo_df['Região'].unique():
        dados_regiao = comparativo_df[comparativo_df['Região'] == regiao]
        ax1.plot(dados_regiao['Ano'], dados_regiao['CMB'], 
                 marker='o', label=regiao, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('CMB (óbitos por 100.000 habitantes)')
    ax1.set_title('Coeficiente de Mortalidade Bruto')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Taxa Padronizada
    for regiao in comparativo_df['Região'].unique():
        dados_regiao = comparativo_df[comparativo_df['Região'] == regiao]
        ax2.plot(dados_regiao['Ano'], dados_regiao['Taxa_Padronizada'], 
                 marker='s', label=regiao, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Taxa Padronizada (óbitos por 100.000 habitantes)')
    ax2.set_title('Taxa de Mortalidade Padronizada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Análise de diferenças
    st.subheader("Análise das Diferenças")
    
    comparativo_df['Diferenca'] = comparativo_df['Taxa_Padronizada'] - comparativo_df['CMB']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Diferenças por Ano:**")
        for ano in comparativo_df['Ano'].unique():
            dados_ano = comparativo_df[comparativo_df['Ano'] == ano]
            st.write(f"**{ano}:**")
            for _, row in dados_ano.iterrows():
                st.write(f"{row['Região']}: {row['Diferenca']:.3f}")
    
    with col2:
        st.write("**Interpretação:**")
        st.write("""
        - **Diferença positiva:** Estrutura etária mais jovem na região
        - **Diferença negativa:** Estrutura etária mais envelhecida na região  
        - **Valores próximos de zero:** Estrutura etária similar à padrão
        
        *A padronização remove o efeito da estrutura etária, permitindo comparações mais válidas entre regiões.*
        """)

# Página: Análise de Tendência
elif pagina == "Análise de Tendência":
    st.header("📈 Análise de Tendência Temporal")
    st.markdown("""
    **Análise de regressão linear** para identificar tendências significativas 
    na mortalidade por leucemia ao longo do tempo.
    """)
    
    st.info("💡 **Nota:** A análise considera apenas os anos com dados populacionais disponíveis (1980, 1991, 2000, 2010).")
    
    # Resultados da análise de tendência
    st.subheader("Resultados da Regressão Linear")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🇧🇷 Nordeste**")
        ne_cmb = tendencia_resultados['Nordeste']['CMB']
        ne_pad = tendencia_resultados['Nordeste']['Padronizada']
        
        st.metric("Tendência CMB", f"{ne_cmb['slope']:.4f} por ano", 
                 delta=f"{ne_cmb['slope']*10:.2f} por década" if not np.isnan(ne_cmb['slope']) else "N/A")
        st.metric("R² CMB", f"{ne_cmb['r_squared']:.3f}")
        st.metric("p-valor CMB", f"{ne_cmb['p_value']:.4f}")
        
        st.metric("Tendência Padronizada", f"{ne_pad['slope']:.4f} por ano",
                 delta=f"{ne_pad['slope']*10:.2f} por década" if not np.isnan(ne_pad['slope']) else "N/A")
        st.metric("R² Padronizada", f"{ne_pad['r_squared']:.3f}")
        st.metric("p-valor Padronizada", f"{ne_pad['p_value']:.4f}")
    
    with col2:
        st.write("**🇧🇷 Sudeste**")
        se_cmb = tendencia_resultados['Sudeste']['CMB']
        se_pad = tendencia_resultados['Sudeste']['Padronizada']
        
        st.metric("Tendência CMB", f"{se_cmb['slope']:.4f} por ano",
                 delta=f"{se_cmb['slope']*10:.2f} por década" if not np.isnan(se_cmb['slope']) else "N/A")
        st.metric("R² CMB", f"{se_cmb['r_squared']:.3f}")
        st.metric("p-valor CMB", f"{se_cmb['p_value']:.4f}")
        
        st.metric("Tendência Padronizada", f"{se_pad['slope']:.4f} por ano",
                 delta=f"{se_pad['slope']*10:.2f} por década" if not np.isnan(se_pad['slope']) else "N/A")
        st.metric("R² Padronizada", f"{se_pad['r_squared']:.3f}")
        st.metric("p-valor Padronizada", f"{se_pad['p_value']:.4f}")
    
    # Gráficos com linhas de tendência
    st.subheader("Gráficos com Linhas de Tendência")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # CMB com tendência
    for regiao in cmb_df['Região'].unique():
        dados_regiao = cmb_df[cmb_df['Região'] == regiao].sort_values('Ano')
        ax1.scatter(dados_regiao['Ano'], dados_regiao['CMB'], 
                   label=f'{regiao} (dados)', alpha=0.7, s=80)
        
        # Linha de tendência
        tendencia = tendencia_resultados[regiao]['CMB']
        if not np.isnan(tendencia['slope']):
            y_pred = tendencia['intercept'] + tendencia['slope'] * dados_regiao['Ano']
            ax1.plot(dados_regiao['Ano'], y_pred, 
                     label=f'{regiao} (tendência)', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('CMB (óbitos por 100.000 habitantes)')
    ax1.set_title('Coeficiente de Mortalidade Bruto com Tendência')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Taxas padronizadas com tendência
    for regiao in padronizado_df['Região'].unique():
        dados_regiao = padronizado_df[padronizado_df['Região'] == regiao].sort_values('Ano')
        ax2.scatter(dados_regiao['Ano'], dados_regiao['Taxa_Padronizada'], 
                   label=f'{regiao} (dados)', alpha=0.7, s=80)
        
        # Linha de tendência
        tendencia = tendencia_resultados[regiao]['Padronizada']
        if not np.isnan(tendencia['slope']):
            y_pred = tendencia['intercept'] + tendencia['slope'] * dados_regiao['Ano']
            ax2.plot(dados_regiao['Ano'], y_pred, 
                     label=f'{regiao} (tendência)', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Taxa Padronizada (óbitos por 100.000 habitantes)')
    ax2.set_title('Taxa de Mortalidade Padronizada com Tendência')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Interpretação
    st.subheader("📋 Interpretação dos Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **Significância estatística (p-valor):**
        - 🔴 p < 0.05: Tendência estatisticamente significativa
        - 🟡 p ≥ 0.05: Tendência não significativa
        
        **Direção da tendência (coeficiente):**
        - 📈 Positivo: Aumento na mortalidade ao longo do tempo
        - 📉 Negativo: Redução na mortalidade ao longo do tempo
        """)
    
    with col2:
        st.write("""
        **Força da relação (R²):**
        - 🟢 0.8-1.0: Forte relação linear
        - 🟡 0.5-0.8: Relação moderada
        - 🔴 0.0-0.5: Fraca relação linear
        
        **Limitações:**
        - Poucos pontos temporais (4 anos)
        - Dados populacionais limitados
        """)

# Rodapé
st.markdown("---")
st.markdown(
    "**Desenvolvido para análise epidemiológica de mortalidade por leucemia** | "
    "Dados: 1979-2022 | Regiões: Nordeste e Sudeste | "
    "População Padrão: Brasil 2010"
)
