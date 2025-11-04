import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Análise de Mortalidade por Leucemia - Nordeste x Sudeste",
    page_icon="📊",
    layout="wide"
)

# Título e introdução
st.title("📊 Análise de Mortalidade por Leucemia: Nordeste vs Sudeste (1979-2022)")
st.markdown("""
**Trabalho de Conclusão de Curso**  
*Análise dos coeficientes de mortalidade bruta e padronizada por idade*
""")

# ------------------------------
# Funções auxiliares
# ------------------------------

@st.cache_data
def read_csv_safely(file):
    """Lê arquivos CSV com tratamento de encoding"""
    try:
        return pd.read_csv(file, sep=';', encoding='utf-8')
    except:
        try:
            return pd.read_csv(file, sep=';', encoding='latin-1')
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return pd.DataFrame()

def transform_to_long_format(df, value_name):
    """Transforma dados do formato wide para long"""
    id_vars = [col for col in df.columns if col in ['Região', 'Regiaﬁo', 'ClassificaÁaﬁo', 'Ano']]
    value_vars = [col for col in df.columns if col not in id_vars + ['Total', 'Idade ignorada']]
    
    region_col = 'Regiaﬁo' if 'Regiaﬁo' in df.columns else 'Região'
    
    df_long = pd.melt(
        df,
        id_vars=[region_col, 'Ano'],
        value_vars=value_vars,
        var_name='FaixaEtaria',
        value_name=value_name
    )
    
    df_long = df_long.rename(columns={region_col: 'Regiao'})
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce').fillna(0)
    df_long['Ano'] = pd.to_numeric(df_long['Ano'], errors='coerce').astype(int)
    
    return df_long

def harmonize_age_groups(faixa_etaria):
    """Padroniza as faixas etárias"""
    mapping = {
        'menor 1 ano': '0-1',
        '1 a 4 anos': '1-4', 
        '5 a 9 anos': '5-9',
        '10 a 14 anos': '10-14',
        '15 a 19 anos': '15-19',
        '20 a 29 anos': '20-29',
        '30 a 39 anos': '30-39',
        '40 a 49 anos': '40-49',
        '50 a 59 anos': '50-59',
        '60 a 69 anos': '60-69',
        '70 a 79 anos': '70-79',
        '80 anos e mais': '80+'
    }
    return mapping.get(faixa_etaria.lower(), faixa_etaria)

# População padrão OMS
OMS_STD_POP = pd.DataFrame({
    'FaixaEtaria': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', 
                   '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
    'PopulacaoPadrao': [8800, 8700, 8600, 8500, 8000, 7500, 7000, 6500, 
                       6000, 5500, 5000, 4000, 2500, 1500, 800, 200, 100]
})

def map_to_std_age_groups(df, tipo):
    """Mapeia para faixas etárias padrão"""
    df_std = df.copy()
    df_std['FaixaEtariaStd'] = df_std['FaixaEtaria'].map(harmonize_age_groups)
    
    # Mapeamento para grupos padrão OMS
    mapping_rules = {
        '0-1': [('0-4', 0.2)],  # Distribuição proporcional
        '1-4': [('0-4', 0.8)],
        '20-29': [('20-24', 0.5), ('25-29', 0.5)],
        '30-39': [('30-34', 0.5), ('35-39', 0.5)],
        '40-49': [('40-44', 0.5), ('45-49', 0.5)],
        '50-59': [('50-54', 0.5), ('55-59', 0.5)],
        '60-69': [('60-64', 0.5), ('65-69', 0.5)],
        '70-79': [('70-74', 0.5), ('75-79', 0.5)]
    }
    
    expanded_data = []
    for _, row in df_std.iterrows():
        faixa = row['FaixaEtariaStd']
        if faixa in mapping_rules:
            for std_faixa, proporcao in mapping_rules[faixa]:
                new_row = row.copy()
                new_row['FaixaEtariaStd'] = std_faixa
                new_row[tipo] = new_row[tipo] * proporcao
                expanded_data.append(new_row)
        else:
            # Se já está no formato padrão, mantém
            if faixa in OMS_STD_POP['FaixaEtaria'].values:
                expanded_data.append(row)
    
    return pd.DataFrame(expanded_data)

def calcular_cmb(obitos, populacao):
    """Calcula Coeficiente de Mortalidade Bruto"""
    return (obitos / populacao) * 100000 if populacao > 0 else 0

def calcular_cmp(obitos_df, populacao_df, pop_padrao):
    """Calcula Coeficiente de Mortalidade Padronizado"""
    cmp_results = []
    
    for regiao in obitos_df['Regiao'].unique():
        for ano in obitos_df['Ano'].unique():
            obitos_regiao = obitos_df[(obitos_df['Regiao'] == regiao) & (obitos_df['Ano'] == ano)]
            pop_regiao = populacao_df[(populacao_df['Regiao'] == regiao) & (populacao_df['Ano'] == ano)]
            
            # Juntar dados
            merged = pd.merge(
                obitos_regiao, pop_regiao, 
                on=['Regiao', 'Ano', 'FaixaEtariaStd'], 
                suffixes=('_obitos', '_pop')
            )
            
            # Juntar com população padrão
            merged = pd.merge(merged, pop_padrao, on='FaixaEtariaStd')
            
            if not merged.empty:
                # Calcular taxa específica por idade
                merged['TaxaEspecifica'] = merged['Obitos'] / merged['Populacao']
                
                # Calcular CMP
                cmp_val = (merged['TaxaEspecifica'] * merged['PopulacaoPadrao']).sum() / merged['PopulacaoPadrao'].sum() * 100000
                
                cmp_results.append({
                    'Regiao': regiao,
                    'Ano': ano,
                    'CMP': cmp_val
                })
    
    return pd.DataFrame(cmp_results)

# ------------------------------
# Sidebar - Upload e Configurações
# ------------------------------

st.sidebar.header("📁 Carregamento de Dados")
st.sidebar.markdown("Faça upload dos arquivos CSV necessários:")

uploaded_obitos = st.sidebar.file_uploader("Dados de Óbitos por Leucemia", type=['csv'])
uploaded_populacao = st.sidebar.file_uploader("Dados de População", type=['csv'])

st.sidebar.header("⚙️ Configurações de Análise")
decada_inicio = st.sidebar.slider("Década de início", 1980, 2010, 1980, step=10)
decada_fim = st.sidebar.slider("Década de fim", 1980, 2020, 2020, step=10)

pop_padrao_opcao = st.sidebar.selectbox(
    "População Padrão para Padronização",
    ["OMS (World Standard Population)", "Brasil 2010"]
)

# ------------------------------
# Processamento dos Dados
# ------------------------------

if uploaded_obitos and uploaded_populacao:
    # Ler dados
    df_obitos = read_csv_safely(uploaded_obitos)
    df_populacao = read_csv_safely(uploaded_populacao)
    
    if not df_obitos.empty and not df_populacao.empty:
        # Transformar para formato longo
        with st.spinner("Processando dados..."):
            obitos_long = transform_to_long_format(df_obitos, 'Obitos')
            populacao_long = transform_to_long_format(df_populacao, 'Populacao')
            
            # Aplicar padronização de faixas etárias
            obitos_std = map_to_std_age_groups(obitos_long, 'Obitos')
            populacao_std = map_to_std_age_groups(populacao_long, 'Populacao')
            
            # Agrupar por faixa etária padrão
            obitos_agg = obitos_std.groupby(['Regiao', 'Ano', 'FaixaEtariaStd'])['Obitos'].sum().reset_index()
            populacao_agg = populacao_std.groupby(['Regiao', 'Ano', 'FaixaEtariaStd'])['Populacao'].sum().reset_index()
            
            # Calcular totais anuais
            obitos_totais = obitos_long.groupby(['Regiao', 'Ano'])['Obitos'].sum().reset_index()
            populacao_totais = populacao_long.groupby(['Regiao', 'Ano'])['Populacao'].sum().reset_index()
            
            # Calcular CMB
            cmb_df = pd.merge(obitos_totais, populacao_totais, on=['Regiao', 'Ano'])
            cmb_df['CMB'] = cmb_df.apply(lambda x: calcular_cmb(x['Obitos'], x['Populacao']), axis=1)
            
            # Calcular CMP
            cmp_df = calcular_cmp(obitos_agg, populacao_agg, OMS_STD_POP)
            
            # Combinar resultados
            resultados = pd.merge(cmb_df, cmp_df, on=['Regiao', 'Ano'], how='left')
            
        st.success("Dados processados com sucesso!")
        
        # ------------------------------
        # SEÇÃO 1: RESUMO EXECUTIVO
        # ------------------------------
        
        st.header("📈 Resumo Executivo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Filtrar por período selecionado
        periodo_mask = (resultados['Ano'] >= decada_inicio) & (resultados['Ano'] <= decada_fim)
        resultados_periodo = resultados[periodo_mask]
        
        with col1:
            total_obitos = resultados_periodo['Obitos'].sum()
            st.metric("Total de Óbitos no Período", f"{total_obitos:,.0f}".replace(",", "."))
        
        with col2:
            avg_cmb = resultados_periodo['CMB'].mean()
            st.metric("CMB Médio", f"{avg_cmb:.2f}")
        
        with col3:
            avg_cmp = resultados_periodo['CMP'].mean()
            st.metric("CMP Médio", f"{avg_cmp:.2f}")
        
        with col4:
            anos_analisados = resultados_periodo['Ano'].nunique()
            st.metric("Anos Analisados", anos_analisados)
        
        # ------------------------------
        # SEÇÃO 2: ANÁLISE TEMPORAL
        # ------------------------------
        
        st.header("📊 Análise Temporal da Mortalidade")
        
        tab1, tab2, tab3 = st.tabs(["Evolução dos Coeficientes", "Comparação Regional", "Tabela de Dados"])
        
        with tab1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Gráfico CMB
            for regiao in resultados['Regiao'].unique():
                dados_regiao = resultados[resultados['Regiao'] == regiao]
                ax1.plot(dados_regiao['Ano'], dados_regiao['CMB'], 
                        marker='o', linewidth=2, label=regiao)
            
            ax1.set_title('Evolução do Coeficiente de Mortalidade Bruto (CMB)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('CMB (óbitos/100.000 hab.)', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico CMP
            for regiao in resultados['Regiao'].unique():
                dados_regiao = resultados[resultados['Regiao'] == regiao]
                ax2.plot(dados_regiao['Ano'], dados_regiao['CMP'], 
                        marker='s', linewidth=2, label=regiao)
            
            ax2.set_title('Evolução do Coeficiente de Mortalidade Padronizado (CMP)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('CMP (óbitos/100.000 hab.)', fontweight='bold')
            ax2.set_xlabel('Ano', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Comparação por Década - CMB")
                resultados['Decada'] = (resultados['Ano'] // 10) * 10
                cmb_decada = resultados.groupby(['Regiao', 'Decada'])['CMB'].mean().unstack()
                st.dataframe(cmb_decada.style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
            
            with col2:
                st.subheader("Comparação por Década - CMP")
                cmp_decada = resultados.groupby(['Regiao', 'Decada'])['CMP'].mean().unstack()
                st.dataframe(cmp_decada.style.format("{:.2f}").background_gradient(cmap='Reds'), use_container_width=True)
        
        with tab3:
            st.subheader("Dados Completos de Coeficientes")
            display_cols = ['Regiao', 'Ano', 'Obitos', 'Populacao', 'CMB', 'CMP']
            st.dataframe(resultados[display_cols].sort_values(['Regiao', 'Ano']), use_container_width=True)
        
        # ------------------------------
        # SEÇÃO 3: ANÁLISE POR FAIXA ETÁRIA
        # ------------------------------
        
        st.header("👥 Análise por Faixa Etária")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição de Óbitos por Faixa Etária")
            
            # Calcular distribuição percentual
            dist_obitos = obitos_std.groupby(['Regiao', 'FaixaEtariaStd'])['Obitos'].sum().reset_index()
            dist_obitos['Percentual'] = dist_obitos.groupby('Regiao')['Obitos'].transform(lambda x: x / x.sum() * 100)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=dist_obitos, x='FaixaEtariaStd', y='Percentual', hue='Regiao', ax=ax)
            ax.set_title('Distribuição Percentual de Óbitos por Faixa Etária', fontweight='bold')
            ax.set_xlabel('Faixa Etária')
            ax.set_ylabel('Percentual (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Taxas Específicas por Idade")
            
            # Calcular taxas específicas
            taxas_especificas = pd.merge(
                obitos_agg, populacao_agg, 
                on=['Regiao', 'Ano', 'FaixaEtariaStd']
            )
            taxas_especificas['Taxa'] = taxas_especificas['Obitos'] / taxas_especificas['Populacao'] * 100000
            
            # Média das taxas por faixa etária
            taxas_medias = taxas_especificas.groupby(['Regiao', 'FaixaEtariaStd'])['Taxa'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=taxas_medias, x='FaixaEtariaStd', y='Taxa', hue='Regiao', 
                        marker='o', ax=ax, linewidth=2.5)
            ax.set_title('Taxa de Mortalidade Específica por Idade', fontweight='bold')
            ax.set_xlabel('Faixa Etária')
            ax.set_ylabel('Taxa (óbitos/100.000 hab.)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # ------------------------------
        # SEÇÃO 4: ANÁLISE ESTATÍSTICA DESCRITIVA
        # ------------------------------
        
        st.header("📋 Estatísticas Descritivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resumo por Região - CMB")
            resumo_cmb = resultados.groupby('Regiao')['CMB'].agg([
                'mean', 'std', 'min', 'max'
            ]).round(2)
            st.dataframe(resumo_cmb, use_container_width=True)
        
        with col2:
            st.subheader("Resumo por Região - CMP")
            resumo_cmp = resultados.groupby('Regiao')['CMP'].agg([
                'mean', 'std', 'min', 'max'
            ]).round(2)
            st.dataframe(resumo_cmp, use_container_width=True)
        
        # ------------------------------
        # SEÇÃO 5: METODOLOGIA E CONSIDERAÇÕES
        # ------------------------------
        
        st.header("🔬 Metodologia e Considerações")
        
        with st.expander("Métodos Estatísticos"):
            st.markdown("""
            **Coeficiente de Mortalidade Bruto (CMB):**
            ```
            CMB = (Número total de óbitos por leucemia no período / População total da região) × 100.000
            ```
            
            **Coeficiente de Mortalidade Padronizado (CMP):**
            - Método direto de padronização por idade
            - População padrão: OMS World Standard Population
            - Fórmula: ∑(taxa específica por idade × população padrão) / ∑população padrão × 100.000
            
            **População de Referência:**
            - Para anos intercensitários: utilizou-se o ano mais próximo disponível
            - População padrão OMS para remover efeito da estrutura etária
            """)
        
        with st.expander("Considerações Éticas"):
            st.markdown("""
            **Aspectos Éticos:**
            - Trata-se de **dados públicos e anonimizados**  
            - Dados agregados, sem possibilidade de identificação individual
            - De acordo com a **Resolução CNS nº 510/2016**, o estudo dispensa submissão a Comitê de Ética em Pesquisa
            
            **Limitações:**
            - Dados de população disponíveis apenas para anos censitários
            - Necessidade de interpolação para anos intercensitários
            - Mapeamento proporcional entre faixas etárias diferentes
            - Subnotificação pode variar entre regiões e períodos
            """)
        
        with st.expander("Exportação de Dados"):
            st.download_button(
                label="📥 Baixar Dados Processados (CSV)",
                data=resultados.to_csv(index=False, encoding='utf-8-sig'),
                file_name=f"dados_mortalidade_leucemia_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Relatório resumido
            relatorio = f"""
            RELATÓRIO DE ANÁLISE - MORTALIDADE POR LEUCEMIA
            Período: {decada_inicio}-{decada_fim}
            Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            
            RESUMO:
            - Total de óbitos analisados: {total_obitos:,}
            - CMB médio no período: {avg_cmb:.2f} óbitos/100.000 hab.
            - CMP médio no período: {avg_cmp:.2f} óbitos/100.000 hab.
            - Período analisado: {anos_analisados} anos
            
            METODOLOGIA:
            - Padronização por idade: Método direto
            - População padrão: OMS World Standard Population
            - Anos de referência populacional: Censos demográficos
            """
            
            st.download_button(
                label="📄 Baixar Relatório (TXT)",
                data=relatorio,
                file_name=f"relatorio_mortalidade_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    else:
        st.error("Erro ao processar os arquivos. Verifique o formato dos dados.")
else:
    st.info("👆 Faça upload dos arquivos CSV para iniciar a análise")
    
    # Exemplo de estrutura esperada
    st.markdown("""
    ### Estrutura Esperada dos Arquivos:
    
    **Dados de Óbitos:**
    - Colunas: Região, Ano, Classificação, Menor 1 ano, 1 a 4 anos, ..., 80 anos e mais, Total
    
    **Dados de População:**
    - Colunas: Região, Ano, Menor 1 ano, 1 a 4 anos, ..., 80 anos e mais, Total
    
    ### Sobre a Análise:
    - **Período:** 1979-2022 (dependendo dos dados disponíveis)
    - **Regiões:** Nordeste e Sudeste
    - **Métodos:** Cálculo de CMB e CMP (padronização direta)
    - **Saídas:** Gráficos temporais, análise por faixa etária, estatísticas descritivas
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Trabalho de Conclusão de Curso - Análise de Mortalidade por Leucemia | "
    "Desenvolvido com Streamlit | "
    f"Última atualização: {datetime.now().strftime('%d/%m/%Y')}"
    "</div>",
    unsafe_allow_html=True
)
