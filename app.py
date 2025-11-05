
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from unidecode import unidecode

# ========= CONFIG =========
st.set_page_config(page_title="Mortalidade por Leucemia - Nordeste x Sudeste", layout="wide")

DATA_DIR = Path(".")  # os .csv devem estar no mesmo diretório do app
FILE_OBITOS = DATA_DIR / "Obitos_regiões.csv"
FILE_POP_REGIOES = DATA_DIR / "Pop_regiões.csv"
FILE_POP_BR = DATA_DIR / "Pop_BR.csv"

# ========= HELPERS =========
def norm(s: str) -> str:
    return (
        unidecode(s)
        .strip()
        .lower()
        .replace("  ", " ")
        .replace("-", " ")
        .replace(".", "")
        .replace("/", " ")
        .replace("\\", " ")
    )

COLMAP = {
    "regiao": {"regiao", "região", "uf", "macroregiao", "macrorregiao", "region"},
    "ano": {"ano", "periodo", "período", "year"},
    "faixa": {"faixa etaria", "faixa_etaria", "grupo etario", "grupo_etario", "idade", "grupo etario ibge", "grupo_etario_ibge"},
    "obitos": {"obitos", "óbitos", "obitos leucemia", "obitos_leucemia", "deaths", "obitos total", "obitos_total"},
    "pop": {"populacao", "população", "population", "pop_total", "pop", "populacao total"},
    "pop_padrao": {"populacao padrao", "populacao_padrao", "pop padrao", "pop_padrao", "padrao", "standard population", "populacao brasil", "pop_br"},
}

def auto_map_columns(df: pd.DataFrame, required: List[str]) -> Dict[str, str]:
    """Tenta mapear nomes de colunas do df para chaves lógicas em 'required'."""
    candidates = {k: {norm(x) for x in COLMAP.get(k, set())} for k in COLMAP}
    normalized_cols = {norm(c): c for c in df.columns}
    mapping = {}
    for need in required:
        found = None
        for col_norm, original in normalized_cols.items():
            if need in candidates and ((col_norm in candidates[need]) or (need == col_norm)):
                found = original
                break
        # fallback por similaridade simples
        if not found:
            for col_norm, original in normalized_cols.items():
                if any(token in col_norm for token in candidates.get(need, [])):
                    found = original
                    break
        if not found:
            # tentativa por início/contém
            for col_norm, original in normalized_cols.items():
                if need in col_norm or col_norm in candidates.get(need, set()):
                    found = original
                    break
        if not found:
            raise ValueError(f"Não foi possível identificar a coluna para '{need}'. Colunas disponíveis: {list(df.columns)}")
        mapping[need] = found
    return mapping

def read_csv_smart(path: Path) -> pd.DataFrame:
    # tenta encoding comum
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # fallback ao pandas default
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data():
    ob = read_csv_smart(FILE_OBITOS)
    pr = read_csv_smart(FILE_POP_REGIOES)
    pb = read_csv_smart(FILE_POP_BR)

    # mapear colunas
    map_ob = auto_map_columns(ob, ["regiao", "ano", "faixa", "obitos"])
    map_pr = auto_map_columns(pr, ["regiao", "ano", "faixa", "pop"])
    # População padrão pode ter apenas faixa e população (sem ano)
    try:
        map_pb = auto_map_columns(pb, ["faixa", "pop"])
        pb = pb.rename(columns={map_pb["faixa"]: "faixa", map_pb["pop"]: "pop_padrao"})
    except Exception:
        # pode vir com nome já explícito
        map_pb = auto_map_columns(pb, ["faixa", "pop_padrao"])
        pb = pb.rename(columns={map_pb["faixa"]: "faixa", map_pb["pop_padrao"]: "pop_padrao"})

    # renomeia obitos e pop_regioes
    ob = ob.rename(columns={map_ob["regiao"]: "regiao", map_ob["ano"]: "ano", map_ob["faixa"]: "faixa", map_ob["obitos"]: "obitos"})
    pr = pr.rename(columns={map_pr["regiao"]: "regiao", map_pr["ano"]: "ano", map_pr["faixa"]: "faixa", map_pr["pop"]: "pop"})

    # normalizações
    for df in [ob, pr, pb]:
        if "regiao" in df.columns:
            df["regiao"] = df["regiao"].astype(str).str.strip()
        if "faixa" in df.columns:
            df["faixa"] = df["faixa"].astype(str).str.strip()
        if "ano" in df.columns:
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")

    # padroniza categorias de faixa etária (remove espaços duplicados)
    for df in [ob, pr, pb]:
        if "faixa" in df.columns:
            df["faixa"] = df["faixa"].str.replace(r"\s+", " ", regex=True)  # será corrigido logo abaixo

    # Correção: usar regex de whitespace corretamente
    for df in [ob, pr, pb]:
        if "faixa" in df.columns:
            df["faixa"] = df["faixa"].str.replace(r"\s+", " ", regex=True).str.replace(r"\s+", " ", regex=True)

    # Se Pop_BR tiver anos, agrega por faixa
    if "ano" in pb.columns:
        pb = pb.groupby("faixa", as_index=False)["pop_padrao"].sum()

    return ob, pr, pb

def ensure_midpoint_population(pr: pd.DataFrame, regiao: str, anos: List[int]) -> Tuple[int, pd.DataFrame]:
    """Retorna o ano do ponto médio e a tabela de população daquele ano por faixa para a região, com possível interpolação."""
    anos_validos = sorted([int(a) for a in anos if pd.notnull(a)])
    if not anos_validos:
        raise ValueError("Não há anos válidos no filtro.")
    mid = int(round(np.mean(anos_validos)))
    # populações disponíveis para a região
    pr_r = pr.loc[pr["regiao"] == regiao].copy()
    anos_reg = sorted(pr_r["ano"].dropna().astype(int).unique())
    if mid in anos_reg:
        base = pr_r.loc[pr_r["ano"] == mid, ["faixa", "pop"]].copy()
        base["ano"] = mid
        return mid, base
    # interpolação simples por faixa etária
    def interp_for_group(g):
        g = g.sort_values("ano")
        return np.interp(mid, g["ano"], g["pop"])
    est = pr_r.groupby("faixa").apply(interp_for_group).rename("pop").reset_index()
    est["ano"] = mid
    return mid, est[["faixa", "pop", "ano"]]

def compute_tmi(ob: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    """TMI_i por regiao, ano e faixa = obitos / pop (no mesmo ano)."""
    m = pd.merge(ob, pr, on=["regiao", "ano", "faixa"], how="inner")
    m["tmi"] = m["obitos"] / m["pop"]
    return m

def compute_cmb(ob: pd.DataFrame, pr: pd.DataFrame, anos_sel: List[int]) -> pd.DataFrame:
    """CMB por regiao e ano_ponto_medio do período selecionado."""
    out_rows = []
    for reg in sorted(ob["regiao"].unique()):
        anos_reg = sorted(ob.loc[ob["regiao"] == reg, "ano"].dropna().astype(int).unique())
        anos_periodo = [a for a in anos_sel if a in anos_reg]
        if not anos_periodo:
            continue
        mid, pop_mid = ensure_midpoint_population(pr, reg, anos_periodo)
        # total de óbitos no período
        total_obitos = ob.loc[(ob["regiao"] == reg) & (ob["ano"].isin(anos_periodo)), "obitos"].sum()
        # população total no ponto médio (somar sobre faixas)
        pop_total_mid = pop_mid["pop"].sum()
        cmb = (total_obitos / pop_total_mid) * 100000.0 if pop_total_mid > 0 else np.nan
        out_rows.append({"regiao": reg, "periodo_ini": min(anos_periodo), "periodo_fim": max(anos_periodo), "ano_ponto_medio": mid, "obitos_periodo": int(total_obitos), "pop_ponto_medio": float(pop_total_mid), "CMB_100k": float(cmb)})
    return pd.DataFrame(out_rows)

def compute_padronizado_direto(ob: pd.DataFrame, pr: pd.DataFrame, pb: pd.DataFrame) -> pd.DataFrame:
    """Taxa padronizada direta por regiao e ano usando Pop_BR como padrão.
    Resultado expresso por 100.000 habitantes.
    """
    # pesos da população padrão
    pb_use = pb.copy()
    total_padrao = pb_use["pop_padrao"].sum()
    pb_use["peso"] = pb_use["pop_padrao"] / total_padrao

    # unir óbitos, população regional e pesos por faixa
    m = pd.merge(ob, pr, on=["regiao", "ano", "faixa"], how="inner")
    m = pd.merge(m, pb_use[["faixa", "peso"]], on="faixa", how="inner")
    # taxa específica por faixa
    m["tmi"] = m["obitos"] / m["pop"]
    # taxa padronizada (soma das taxas específicas ponderadas pelos pesos)
    pad = m.groupby(["regiao", "ano"], as_index=False).apply(lambda g: pd.Series({
        "taxa_padronizada_100k": float((g["tmi"] * g["peso"]).sum() * 100000.0),
        "taxa_bruta_100k": float((g["obitos"].sum() / g["pop"].sum()) * 100000.0 if g["pop"].sum() > 0 else np.nan),
        "obitos": int(g["obitos"].sum()),
        "pop": float(g["pop"].sum())
    })).reset_index().drop(columns=["level_2"], errors="ignore")
    return pad

def plot_lines(df: pd.DataFrame, y: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for reg, g in df.sort_values("ano").groupby("regiao"):
        ax.plot(g["ano"], g[y], marker="o", label=reg)
    ax.set_xlabel("Ano")
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def try_import_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except Exception as e:
        return None

def joinpoint_analysis(series_df: pd.DataFrame, y: str, max_breaks: int = 2):
    """Estimativa de pontos de mudança com 'ruptures' e ajuste linear por trechos."""
    rpt = try_import_ruptures()
    if rpt is None:
        st.info("Para a análise de Joinpoint, instale o pacote 'ruptures'. Mostrando apenas a série temporal.")
        plot_lines(series_df, y, f"Tendência de {y}")
        return

    s = series_df.sort_values("ano").copy()
    x = s["ano"].to_numpy().reshape(-1, 1).astype(float)
    yv = s[y].to_numpy().astype(float)
    algo = rpt.KernelCPD(kernel="linear").fit(np.column_stack([x.flatten(), yv]))
    n_bkps = min(max_breaks, max(1, len(s)//4))  # limite razoável
    # tenta estimativas de 0..max_breaks e escolhe por penalidade BIC simples
    best_bkps, best_pen = None, np.inf
    for nb in range(0, n_bkps + 1):
        try:
            bkps = algo.predict(n_bkps=nb)
        except Exception:
            continue
        # bkps são índices de fim de segmento; computar SSE básico
        prev = 0
        sse = 0.0
        for b in bkps:
            xi = x[prev:b].flatten()
            yi = yv[prev:b]
            if len(xi) >= 2:
                A = np.vstack([xi, np.ones_like(xi)]).T
                m, c = np.linalg.lstsq(A, yi, rcond=None)[0]
                yhat = m*xi + c
                sse += float(((yi - yhat) ** 2).sum())
            prev = b
        k_params = 2*(len(bkps))  # inclinação e intercepto por segmento aproximado
        n = len(yv)
        bic = n*np.log(sse/n if n>0 and sse>0 else 1.0) + k_params*np.log(n if n>0 else 1.0)
        if bic < best_pen:
            best_pen, best_bkps = bic, bkps

    # plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s["ano"], yv, marker="o")
    if best_bkps is not None:
        prev = 0
        for b in best_bkps:
            xi = x[prev:b].flatten()
            yi = yv[prev:b]
            if len(xi) >= 2:
                A = np.vstack([xi, np.ones_like(xi)]).T
                m, c = np.linalg.lstsq(A, yi, rcond=None)[0]
                ax.plot(xi, m*xi + c, linewidth=2)
            if b < len(s):
                ax.axvline(s.iloc[b-1]["ano"], linestyle="--", alpha=0.5)
            prev = b
    ax.set_title(f"Tendência com Joinpoints estimados para {series_df['regiao'].iloc[0]}")
    ax.set_xlabel("Ano")
    ax.set_ylabel(y.replace("_", " "))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ========= UI =========
st.title("Painel: Mortalidade por Leucemia — Nordeste x Sudeste")

with st.expander("Sobre os dados e método"):
    st.markdown("""
    Este painel lê três arquivos .csv presentes no mesmo diretório do app:
    - **Obitos_regiões.csv**: óbitos por região, ano e faixa etária.
    - **Pop_regiões.csv**: população por região, ano e faixa etária.
    - **Pop_BR.csv**: população padrão do Brasil por faixa etária.

    Ele calcula:
    - **CMB** no período selecionado: total de óbitos no período dividido pela população do ponto médio do período, vezes 100.000.
    - **TMIᵢ** por faixa: óbitos na faixa/ano/região divididos pela população da faixa/ano/região.
    - **Taxa padronizada direta** por 100.000 usando a população padrão do Brasil (Pop_BR) como pesos.
    Também gera tabelas descritivas e gráficos de tendência.
    """)

# Carregar dados
try:
    ob, pr, pb = load_data()
except Exception as e:
    st.error(f"Erro ao carregar os CSVs. Verifique se os arquivos existem e se os nomes de colunas são reconhecíveis. Detalhes: {e}")
    st.stop()

# Filtros
regioes = sorted(ob["regiao"].dropna().unique().tolist())
anos_disponiveis = sorted(ob["ano"].dropna().astype(int).unique().tolist())
col1, col2, col3 = st.columns(3)
with col1:
    regs_sel = st.multiselect("Regiões", regioes, default=[r for r in regioes if r.lower() in {"nordeste", "sudeste"}] or regioes[:2])
with col2:
    ano_ini, ano_fim = st.select_slider("Período", options=anos_disponiveis, value=(anos_disponiveis[0], anos_disponiveis[-1]))
with col3:
    max_breaks = st.slider("Nº máximo de joinpoints", 0, 3, 2, help="Usado na análise de tendência com ruptures.")

anos_sel = [a for a in anos_disponiveis if ano_ini <= a <= ano_fim]
ob_f = ob[ob["regiao"].isin(regs_sel) & ob["ano"].isin(anos_sel)].copy()
pr_f = pr[pr["regiao"].isin(regs_sel) & pr["ano"].isin(anos_sel)].copy()

# ====== Descritivo de óbitos ======
st.subheader("Frequência de óbitos — tabela descritiva")
desc = ob_f.groupby(["regiao", "ano"], as_index=False)["obitos"].sum().pivot(index="ano", columns="regiao", values="obitos").fillna(0).astype(int)
st.dataframe(desc, use_container_width=True)

# ====== TMI por faixa ======
st.subheader("Taxas específicas por faixa etária (TMIᵢ)")
try:
    tmi = compute_tmi(ob_f, pr_f)
    # Mostrar tabela resumida
    tmi_tbl = tmi.groupby(["regiao", "ano", "faixa"], as_index=False).agg(tmi=("tmi", lambda x: (x.mean()*100000.0))).rename(columns={"tmi": "TMI_i_100k"})
    st.dataframe(tmi_tbl, use_container_width=True)
except Exception as e:
    st.warning(f"Não foi possível calcular TMIᵢ com os filtros atuais. {e}")
    tmi_tbl = None

# ====== CMB para o período ======
st.subheader("Coeficiente de Mortalidade Bruto (CMB) por 100.000 habitantes — período selecionado")
try:
    cmb = compute_cmb(ob_f, pr, anos_sel)
    st.dataframe(cmb, use_container_width=True)
except Exception as e:
    st.warning(f"Não foi possível calcular CMB. {e}")
    cmb = None

# ====== Padronização direta ======
st.subheader("Taxa padronizada por idade (método direto) — por 100.000 habitantes")
try:
    pad = compute_padronizado_direto(ob_f, pr_f, pb)
    st.dataframe(pad, use_container_width=True)
except Exception as e:
    st.error(f"Erro ao calcular taxas padronizadas. {e}")
    st.stop()

# ====== Gráficos de tendência ======
st.subheader("Tendência — taxas brutas vs padronizadas")
with st.container():
    colA, colB = st.columns(2)
    with colA:
        try:
            plot_lines(pad, "taxa_bruta_100k", "Taxa bruta por 100.000 ao longo do tempo")
        except Exception as e:
            st.warning(f"Falha ao plotar taxa bruta. {e}")
    with colB:
        try:
            plot_lines(pad, "taxa_padronizada_100k", "Taxa padronizada por 100.000 ao longo do tempo")
        except Exception as e:
            st.warning(f"Falha ao plotar taxa padronizada. {e}")

# ====== Joinpoint (estimativa de pontos de mudança) ======
st.subheader("Análise de tendência com Joinpoints (estimados)")
for reg in regs_sel:
    sr = pad[pad["regiao"] == reg].copy()
    if len(sr) < 4:
        st.info(f"{reg}: poucos pontos no tempo para estimar joinpoints.")
        continue
    joinpoint_analysis(sr, "taxa_padronizada_100k", max_breaks=max_breaks)

st.caption("Observação: as estimativas dependem da qualidade e consistência dos dados por faixa etária. Se algum ano ou faixa estiver ausente, os cálculos serão feitos com os cruzamentos disponíveis.")
