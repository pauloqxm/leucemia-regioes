import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import os

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
def read_csv_safely(file_path):
    """Lê arquivos CSV com tratamento robusto de encoding e separadores"""
    try:
        # Tentar diferentes encodings e separadores
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
        separators = [';', ',']
        
        for encoding in encodings:
            for separator in separators:
                try:
                    df = pd.read_csv(file_path, sep=separator, encoding=encoding)
                    if not df.empty:
                        st.success(f"Arquivo lido com encoding: {encoding}, separador: '{separator}'")
                        return df
                except:
                    continue
        
        # Última tentativa com parâmetros padrão
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Erro ao ler arquivo {file_path}: {str(e)}")
        return pd.DataFrame()

def detect_column_names(df):
    """Detecta e corrige nomes de colunas com caracteres especiais"""
    if df.empty:
        return df
    
    # Mapeamento de correção para os caracteres problemáticos
    correction_map = {
        'RegiaÞo': 'Regiao',
        'Regiaﬁo': 'Regiao', 
        'ClassificaÁaﬁo': 'Classificacao',
        'ClassificaçaÞo': 'Classificacao'
    }
    
    # Renomear colunas
    new_columns = []
    for col in df.columns:
        if col in correction_map:
            new_columns.append(correction_map[col])
        else:
            new_columns.append(col)
    
    df.columns = new_columns
    return df

def transform_to_long_format(df, value_name):
    """Transforma dados do formato wide para long"""
    # Identificar colunas de metadados
    meta_columns = ['Regiao', 'Classificacao', 'Ano', 'Total', 'Idade ignorada']
    available_meta = [col for col in meta_columns if col in df.columns]
    
    # Colunas de faixa etária são as que não são metadados
    age_columns = [col for col in df.columns if col not in available_meta]
    
    # Usar colunas disponíveis
    id_cols = [col for col in available_meta if col in ['Regiao', 'Ano']]
    
    df_long = pd.melt(
        df,
        id_vars=id_cols,
        value_vars=age_columns,
        var_name='FaixaEtaria',
        value_name=value_name
    )
    
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce').fillna(0)
    df_long['Ano'] = pd.to_numeric(df_long['Ano'], errors='coerce').astype(int)
    
    return df_long

def harmonize_age_groups(faixa_etaria):
    """Padroniza as faixas etárias"""
    if pd.isna(faixa_etaria):
        return None
        
    faixa = str(faixa_etaria).strip().lower()
    
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
        '80 anos e mais': '80+',
        '80 anos e mais': '80+',
        'total': 'Total',
        'idade ignorada': 'Ignorada'
    }
    
    return mapping.get(faixa, faixa)

# População padrão OMS
OMS_STD_POP = pd.DataFrame({
    'FaixaEtaria': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', 
                   '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
    'PopulacaoPadrao': [8800, 8700, 8600, 8500, 8000, 7500, 7000, 6500, 
                       6000, 5500, 5000, 4000, 2500, 1500, 800, 200, 100]
})

def map_to_std_age_groups(faixa_original, valor, tipo):
    """Mapeia faixas etárias para o padrão OMS com distribuição proporcional"""
    faixa = harmonize_age_groups(faixa_original)
    
    if faixa == '0-1':
        return [('0-4', valor * 0.2)]
    elif faixa == '1-4':
        return [('0-4', valor * 0.8)]
    elif faixa == '20-29':
        return [('20-24', valor * 0.5), ('25-29', valor * 0.5)]
    elif faixa == '30-39':
        return [('30-34', valor * 0.5), ('35-39', valor * 0.5)]
    elif faixa == '40-49':
        return [('40-44', valor * 0.5), ('45-49', valor * 0.5)]
    elif faixa == '50-59':
        return [('50-54', valor * 0.5), ('55-59', valor * 0.5)]
    elif faixa == '60-69':
        return [('60-64', valor * 0.5), ('65-69', valor * 0.5)]
    elif faixa == '70-79':
        return [('70-74', valor * 0.5), ('75-79', valor * 0.5)]
    elif faixa in OMS_STD_POP['FaixaEtaria'].values:
        return [(faixa, valor)]
    else:
        return []

def calcular_cmb(obitos, populacao):
    """Calcula Coeficiente de Mortalidade Bruto"""
    if populacao > 0:
        return (obitos / populacao) * 100000
    return 0

def processar_dados_para_analise(obitos_long, populacao_long):
    """Processa os dados para cálculo de CMB e CMP"""
    
    # Calcular totais anuais para CMB
    obitos_totais = obitos_long.groupby(['Regiao', 'Ano'])['Obitos'].sum().reset_index()
    populacao_totais = populacao_long.groupby(['Regiao', 'Ano'])['Populacao'].sum().reset_index()
    
    # Calcular CMB
    cmb_df = pd.merge(obitos_totais, populacao_totais, on=['Regiao', 'Ano'])
    cmb_df['CMB'] = cmb_df.apply(lambda x: calcular_cmb(x['Obitos'], x['Populacao']), axis=1)
    
    # Preparar dados para CMP
    cmp_data = []
    
    for regiao in obitos_long['Regiao'].unique():
        for ano in obitos_long['Ano'].unique():
            # Obter dados da região e ano
            obitos_regiao = obitos_long[(obitos_long['Regiao'] == regiao) & (obitos_long['Ano'] == ano)]
            pop_regiao = populacao_long[(populacao_long['Regiao'] == regiao) & (populacao_long['Ano'] == ano)]
            
            # Processar cada faixa etária
            obitos_std = []
            pop_std = []
            
            for _, row in obitos_regiao.iterrows():
                mappings = map_to_std_age_groups(row['FaixaEtaria'], row['Obitos'], 'obitos')
                for faixa_std, valor in mappings:
                    obitos_std.append({'FaixaEtaria': faixa_std, 'Obitos': valor})
            
            for _, row in pop_regiao.iterrows():
                mappings = map_to_std_age_groups(row['FaixaEtaria'], row['Populacao'], 'populacao')
                for faixa_std, valor in mappings:
                    pop_std.append({'FaixaEtaria': faixa_std, 'Populacao': valor})
            
            # Agrupar por faixa etária padrão
            if obitos_std and pop_std:
                obitos_std_df = pd.DataFrame(obitos_std).groupby('FaixaEtaria')['Obitos'].sum().reset_index()
                pop_std_df = pd.DataFrame(pop_std).groupby('FaixaEtaria')['Populacao'].sum().reset_index()
                
                # Juntar com população padrão
                merged = pd.merge(obitos_std_df, pop_std_df, on='FaixaEtaria', how='outer').fillna(0)
                merged = pd.merge(merged, OMS_STD_POP, on='FaixaEtaria', how='inner')
                
                if not merged.empty and merged['PopulacaoPadrao'].sum() > 0:
                    # Calcular CMP
                    merged['TaxaEspecifica'] = merged['Obitos'] / merged['Populacao']
                    merged['TaxaEspecifica'] = merged['TaxaEspecifica'].replace([np.inf, -np.inf], 0).fillna(0)
                    
                    cmp_val = (merged['TaxaEspecifica'] * merged['PopulacaoPadrao']).sum() / merged['PopulacaoPadrao'].sum() * 100000
                    
                    cmp_data.append({
                        'Regiao': regiao,
                        'Ano': ano,
                        'CMP': cmp_val
                    })
    
    cmp_df = pd.DataFrame(cmp_data)
    
    # Combinar resultados
    resultados = pd.merge(cmb_df, cmp_df, on=['Regiao', 'Ano'], how='left')
    
    return resultados, obitos_long, populacao_long

# ------------------------------
# Carregamento Automático dos Arquivos
# ------------------------------

def carregar_arquivos_automaticamente():
    """Carrega automaticamente os arquivos do repositório"""
    arquivos_encontrados = {}
    
    # Procurar por arquivos CSV no diretório
    arquivos_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    st.sidebar.info(f"Arquivos CSV encontrados: {arquivos_csv}")
    
    # Procurar por arquivos específicos
    possiveis_nomes_obitos = [
        'Dados de Obitos.csv', 'Dados de Óbitos.csv', 'obitos.csv', 
        'dados_obitos.csv', 'mortalidade.csv'
    ]
    
    possiveis_nomes_populacao = [
        'Dados de População.csv', 'Dados de Populacao.csv', 'populacao.csv',
        'dados_populacao.csv', 'populacao.csv'
    ]
    
    # Tentar encontrar arquivo de óbitos
    for nome in possiveis_nomes_obitos:
        if os.path.exists(nome):
            arquivos_encontrados['obitos'] = nome
            break
    else:
        # Se não encontrou pelos nomes específicos, usar o primeiro CSV
        if arquivos_csv:
            arquivos_encontrados['obitos'] = arquivos_csv[0]
    
    # Tentar encontrar arquivo de população
    for nome in possiveis_nomes_populacao:
        if os.path.exists(nome):
            arquivos_encontrados['populacao'] = nome
            break
    else:
        # Se não encontrou pelos nomes específicos, usar o segundo CSV
        if len(arquivos_csv) > 1:
            arquivos_encontrados['populacao'] = arquivos_csv[1]
    
    return arquivos_encontrados

# ------------------------------
# Interface Principal
# ------------------------------

st.sidebar.header("📁 Configuração dos Dados")

# Verificar se existem arquivos no repositório
arquivos = carregar_arquivos_automaticamente()

if arquivos.get('obitos') and arquivos.get('populacao'):
    st.sidebar.success("✅ Arquivos encontrados automaticamente!")
    st.sidebar.write(f"**Óbitos:** {arquivos['obitos']}")
    st.sidebar.write(f"**População:** {arquivos['populacao']}")
    
    use_auto = st.sidebar.checkbox("Usar arquivos automáticos", value=True)
else:
    st.sidebar.warning("⚠️ Arquivos não encontrados automaticamente")
    use_auto = False

if use_auto and arquivos.get('obitos') and arquivos.get('populacao'):
    # Usar arquivos automáticos
    df_obitos = read_csv_safely(arquivos['obitos'])
    df_populacao = read_csv_safely(arquivos['populacao'])
else:
    # Upload manual
    st.sidebar.header("📤 Upload Manual")
    uploaded_obitos = st.sidebar.file_uploader("Dados de Óbitos por Leucemia", type=['csv'], key='obitos')
    uploaded_populacao = st.sidebar.file_uploader("Dados de População", type=['csv'], key='populacao')
    
    if uploaded_obitos and uploaded_populacao:
        df_obitos = read_csv_safely(uploaded_obitos)
        df_populacao = read_csv_safely(uploaded_populacao)
    else:
        df_obitos = pd.DataFrame()
        df_populacao = pd.DataFrame()

# ------------------------------
# Filtros na Sidebar
# ------------------------------

if not df_obitos.empty and not df_populacao.empty:
    st.sidebar.header("⚙️ Filtros de Análise")
    
    # Processar dados para obter anos disponíveis
    with st.spinner("Preparando dados para filtros..."):
        df_obitos_corr = detect_column_names(df_obitos)
        df_populacao_corr = detect_column_names(df_populacao)
        
        obitos_long = transform_to_long_format(df_obitos_corr, 'Obitos')
        populacao_long = transform_to_long_format(df_populacao_corr, 'Populacao')
        
        # Obter anos disponíveis
        anos_obitos = sorted(obitos_long['Ano'].unique())
        anos_populacao = sorted(populacao_long['Ano'].unique())
        anos_disponiveis = sorted(list(set(anos_obitos) & set(anos_populacao)))
    
    if anos_disponiveis:
        # Filtro de anos com multiselect
        st.sidebar.subheader("Seleção de Anos")
        anos_selecionados = st.sidebar.multiselect(
            "Selecione os anos para análise:",
            options=anos_disponiveis,
            default=anos_disponiveis  # Todos selecionados por padrão
        )
        
        # Filtro de período com slider
        st.sidebar.subheader("Período de Análise")
        ano_min = min(anos_disponiveis)
        ano_max = max(anos_disponiveis)
        
        periodo = st.sidebar.slider(
            "Selecione o período:",
            min_value=ano_min,
            max_value=ano_max,
            value=(ano_min, ano_max),
            step=1
        )
        
        # Filtro de regiões
        st.sidebar.subheader("Seleção de Regiões")
        regioes_disponiveis = sorted(obitos_long['Regiao'].unique())
        regioes_selecionadas = st.sidebar.multiselect(
            "Selecione as regiões:",
            options=regioes_disponiveis,
            default=regioes_disponiveis
        )
        
        # Aplicar filtros
        aplicar_filtros = st.sidebar.button("Aplicar Filtros")
        
        if aplicar_filtros or not anos_selecionados:
            # Usar período do slider se nenhum ano específico foi selecionado
            if not anos_selecionados:
                anos_selecionados = [ano for ano in anos_disponiveis if periodo[0] <= ano <= periodo[1]]
            
            # Filtrar dados
            obitos_filtrado = obitos_long[
                (obitos_long['Ano'].isin(anos_selecionados)) & 
                (obitos_long['Regiao'].isin(regioes_selecionadas))
            ]
            
            populacao_filtrado = populacao_long[
                (populacao_long['Ano'].isin(anos_selecionados)) & 
                (populacao_long['Regiao'].isin(regioes_selecionadas))
            ]
            
            # Processar análise com dados filtrados
            with st.spinner("Processando análise com filtros aplicados..."):
                resultados, obitos_final, populacao_final = processar_dados_para_analise(obitos_filtrado, populacao_filtrado)
            
            if not resultados.empty:
                st.success(f"✅ Análise filtrada: {len(anos_selecionados)} anos, {len(regioes_selecionadas)} regiões")
                
                # ------------------------------
                # ANÁLISE E VISUALIZAÇÕES
                # ------------------------------
                
                st.header("📈 Análise de Resultados")
                
                # Resumo Executivo
                st.subheader("Resumo Executivo")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_obitos = resultados['Obitos'].sum()
                    st.metric("Total de Óbitos", f"{total_obitos:,.0f}".replace(",", "."))
                
                with col2:
                    avg_cmb = resultados['CMB'].mean()
                    st.metric("CMB Médio", f"{avg_cmb:.2f}")
                
                with col3:
                    avg_cmp = resultados['CMP'].mean()
                    st.metric("CMP Médio", f"{avg_cmp:.2f}")
                
                with col4:
                    anos_analisados = resultados['Ano'].nunique()
                    st.metric("Anos Analisados", anos_analisados)
                
                # Informações do Filtro
                st.info(f"""
                **Filtros Aplicados:**
                - Período: {min(anos_selecionados)} - {max(anos_selecionados)}
                - Regiões: {', '.join(regioes_selecionadas)}
                - Total de anos: {len(anos_selecionados)}
                """)
                
                # Gráficos de Evolução Temporal
                st.subheader("Evolução Temporal dos Coeficientes")
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # CMB
                for regiao in resultados['Regiao'].unique():
                    dados = resultados[resultados['Regiao'] == regiao]
                    ax1.plot(dados['Ano'], dados['CMB'], marker='o', label=regiao, linewidth=2)
                
                ax1.set_title('Coeficiente de Mortalidade Bruto (CMB)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('CMB (óbitos/100.000 hab.)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # CMP
                for regiao in resultados['Regiao'].unique():
                    dados = resultados[resultados['Regiao'] == regiao]
                    ax2.plot(dados['Ano'], dados['CMP'], marker='s', label=regiao, linewidth=2)
                
                ax2.set_title('Coeficiente de Mortalidade Padronizado (CMP)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('CMP (óbitos/100.000 hab.)')
                ax2.set_xlabel('Ano')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tabela de Resultados
                st.subheader("Tabela de Resultados")
                st.dataframe(resultados, use_container_width=True)
                
                # Análise por Faixa Etária
                st.subheader("Análise por Faixa Etária")
                
                # Distribuição de óbitos
                dist_obitos = obitos_final.groupby(['Regiao', 'FaixaEtaria'])['Obitos'].sum().reset_index()
                dist_obitos['Percentual'] = dist_obitos.groupby('Regiao')['Obitos'].transform(
                    lambda x: x / x.sum() * 100
                )
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=dist_obitos, x='FaixaEtaria', y='Percentual', hue='Regiao', ax=ax)
                ax.set_title('Distribuição de Óbitos por Faixa Etária', fontweight='bold')
                ax.set_xlabel('Faixa Etária')
                ax.set_ylabel('Percentual (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # ------------------------------
                # EXPORTAÇÃO E RELATÓRIO
                # ------------------------------
                
                st.header("📤 Exportação de Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download dos dados processados
                    csv = resultados.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Baixar Dados Processados (CSV)",
                        data=csv,
                        file_name="dados_mortalidade_leucemia.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Relatório resumido
                    relatorio = f"""
RELATÓRIO DE ANÁLISE - MORTALIDADE POR LEUCEMIA
Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}

FILTROS APLICADOS:
- Período: {min(anos_selecionados)} - {max(anos_selecionados)}
- Regiões: {', '.join(regioes_selecionadas)}
- Anos analisados: {len(anos_selecionados)}

RESUMO ESTATÍSTICO:
- Total de óbitos analisados: {total_obitos:,}
- CMB médio: {avg_cmb:.2f} óbitos/100.000 hab.
- CMP médio: {avg_cmp:.2f} óbitos/100.000 hab.

METODOLOGIA:
- Coeficiente de Mortalidade Bruto (CMB): (Óbitos / População) × 100.000
- Coeficiente de Mortalidade Padronizado (CMP): Método direto com população padrão OMS
- População padrão: OMS World Standard Population 2000-2025
                    """
                    
                    st.download_button(
                        label="📄 Baixar Relatório (TXT)",
                        data=relatorio,
                        file_name="relatorio_analise.txt",
                        mime="text/plain"
                    )
                
                # ------------------------------
                # CONSIDERAÇÕES ÉTICAS
                # ------------------------------
                
                st.header("🔬 Considerações Éticas e Metodológicas")
                
                with st.expander("Aspectos Éticos"):
                    st.markdown("""
                    **Conforme Resolução CNS nº 510/2016:**
                    - Trata-se de **dados públicos e anonimizados**
                    - Dados em formato agregado, sem possibilidade de identificação individual
                    - Dispensa submissão a Comitê de Ética em Pesquisa
                    
                    **Fontes dos Dados:**
                    - Sistemas de informação em saúde oficiais
                    - Dados censitários do IBGE
                    - Bases públicas do Ministério da Saúde
                    """)
                
                with st.expander("Limitações Metodológicas"):
                    st.markdown("""
                    **Considerações Importantes:**
                    - População disponível apenas para anos censitários
                    - Necessidade de interpolação para anos intercensitários
                    - Subnotificação pode variar entre regiões e períodos
                    - Mudanças na classificação de causas de óbito (CID-9 para CID-10)
                    """)
            
            else:
                st.error("Não foi possível processar os dados com os filtros selecionados.")
    
    else:
        st.error("Não há anos comuns entre os datasets de óbitos e população.")

else:
    st.info("👆 Configure os dados na sidebar para iniciar a análise")
    
    # Instruções
    st.markdown("""
    ### 📝 Instruções para Uso:
    
    1. **Configuração dos Dados:**
       - O sistema tentará encontrar automaticamente os arquivos no repositório
       - Se não encontrar, use o upload manual na sidebar
    
    2. **Aplicar Filtros:**
       - Selecione os anos desejados na sidebar
       - Escolha as regiões para análise
       - Clique em "Aplicar Filtros"
    
    3. **A análise incluirá:**
       - Cálculo do Coeficiente de Mortalidade Bruto (CMB)
       - Padronização por idade (CMP) usando população OMS
       - Gráficos de evolução temporal
       - Análise por faixa etária
       - Estatísticas descritivas
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Trabalho de Conclusão de Curso - Análise de Mortalidade por Leucemia | "
    "Desenvolvido com Streamlit"
    "</div>",
    unsafe_allow_html=True
)
