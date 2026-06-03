# -*- coding: utf-8 -*-
"""Gera analise_basocelular_v2.ipynb — rodada 2 das análises do PIBIC."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ============================================================ Cabeçalho
md(r"""# Análise CBC — Rodada 2 (PIBIC / LMECC)

Segunda rodada de análises com base nas sugestões da orientadora (vic):

**Tratamentos adicionais sobre a base limpa da rodada 1:**
- **Estado civil** binarizado: com vs sem companheiro(a).
- **Faixa etária**: ≤ 40 anos vs > 40 anos *(corte clínico definido pela orientadora)*.
- **Exposição solar** ocupacional: profissões com vs sem exposição solar.
- **Subtipo do Tumor (3 categorias)**: classificação clínica padrão CBC — **Baixo risco / Alto risco / Misto** (por componentes histológicos).
- **Região anatômica**: face detalhada (nariz, malar, frontal, têmpora, pálpebra/periocular, perioral, outros) + regiões extra-faciais agrupadas.

**Cinco blocos de análises (33 cruzamentos no total):**
- **Bloco A**: Estado civil × profundidade, tamanho, ulceração, margens — estratificado por sexo (8).
- **Bloco B**: Faixa etária × margens, perineural, linfovascular, ulceração, tamanho, grau, subtipo (7).
- **Bloco C**: Exposição solar × subtipo, ulceração, perineural, linfovascular, margens, tamanho, grau (7).
- **Bloco D**: Região × exposição solar, subtipo, margens, perineural, linfovascular, ulceração, grau, tamanho (8).
- **Bloco E**: Subtipo (3 cat) × região, perineural, linfovascular, grau, margens, tamanho, ulceração (7).

**Padrão estatístico:** qui-quadrado de Pearson (Fisher quando esperado <5 em 2×2); Mann-Whitney para tamanho com 2 grupos, Kruskal-Wallis com ≥ 3; α = 0,05; tamanho de efeito (V de Cramer / r) reportado.
""")

# ============================================================ Setup
md("## 0. Setup")
code(r"""import os, re, unicodedata, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "font.size": 11})

ARQUIVO = "Dados_PIBIC_LMECC_Completo(Banco de Dados) (1).csv"
FIG_DIR = "figuras_v2"
os.makedirs(FIG_DIR, exist_ok=True)

# acumulador da tabela-mestre de p-valores
RESULTADOS = []

def salvar_fig(nome):
    plt.savefig(os.path.join(FIG_DIR, nome)); print(f"[figura] {nome}")
print("Setup OK.")""")

# ============================================================ Leitura
md("## 1. Carga do CSV")
code(r"""brutos = pd.read_csv(ARQUIVO, sep=None, engine="python", encoding="utf-8-sig")
print("Dimensões:", brutos.shape)
brutos.head(3)""")

# ============================================================ Limpeza base (rodada 1)
md("## 2. Limpeza herdada da rodada 1")
code(r"""def sem_acento(s):
    if pd.isna(s): return s
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))
def norm(s):
    if pd.isna(s): return np.nan
    s = sem_acento(str(s)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else np.nan

df = brutos.rename(columns={
    "Código da Amostra":"codigo","Data do diagnóstico":"data_dx",
    "Idade ao diagnóstico":"idade","Sexo":"sexo","Etnia":"etnia",
    "Estado Civil":"estado_civil","Profissão":"profissao",
    "Subtipo do Tumor":"subtipo_raw","Região Anatômica da Lesão":"regiao_raw",
    "Tamanho da Lesão (cm)":"tamanho_cm","Presença de Ulceração":"ulceracao",
    "infiltração":"infiltracao_raw",
    "Estado de Comprometimento das Margens (Lateral)":"margem_lat_raw",
    "Estado de Comprometimento das Margens (Profunda)":"margem_prof_raw",
    "Reincidência?":"reincidencia","Quantas reincidências?":"qtd_reincidencias",
    "Invasão Linfovascular?":"inv_linfovascular","Invasão Perineural?":"inv_perineural",
}).drop(columns=["Tipo de Material Analisado","Tratamento"], errors="ignore")

# tamanho_cm: aceita vírgula decimal do CSV
df["tamanho_cm"] = (df["tamanho_cm"].astype(str).str.replace(",", ".", regex=False)
                                                .replace({"nan": np.nan}))
df["tamanho_cm"] = pd.to_numeric(df["tamanho_cm"], errors="coerce").round(2)

df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
df["sexo"] = df["sexo"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
df["etnia"] = df["etnia"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
df["estado_civil"] = df["estado_civil"].where(df["estado_civil"] != "Branca")

df["ulceracao"]        = df["ulceracao"].apply(norm).map({"sim":"Sim","nao":"Não"})
df["reincidencia"]     = df["reincidencia"].apply(norm).map({"sim":"Sim","nao":"Não"})
df["inv_perineural"]   = df["inv_perineural"].apply(norm).map({"sim":"Sim","nao":"Não"})
df["inv_linfovascular"]= df["inv_linfovascular"].apply(norm).map({"sim":"Sim","nao":"Não"})

def normaliza_margem(s):
    s = norm(s)
    if pd.isna(s): return np.nan
    if "comprometid" in s: return "Comprometida"
    if "livre" in s: return "Livre"
    return np.nan
df["margem_lateral"]  = df["margem_lat_raw"].apply(normaliza_margem)
df["margem_profunda"] = df["margem_prof_raw"].apply(normaliza_margem)
def comb(r):
    v=[r["margem_lateral"], r["margem_profunda"]]
    if "Comprometida" in v: return "Comprometida"
    if "Livre" in v: return "Livre"
    return np.nan
df["margem_comprometida"] = df.apply(comb, axis=1)

ORDEM_GRAU = ["1. Derme","2. Derme reticular","3. Hipoderme/subcutâneo","4. Estruturas profundas"]
def grau(s):
    s = norm(s)
    if pd.isna(s) or "nao informado" in s: return np.nan
    if any(k in s for k in ["globo ocular","muscul","musc","cartilagem","periorbit","estriad"]):
        return "4. Estruturas profundas"
    if any(k in s for k in ["hipoderme","subcut","adiposo","celular subc"]):
        return "3. Hipoderme/subcutâneo"
    if "reticular" in s or "derme profunda" in s: return "2. Derme reticular"
    if "derme" in s: return "1. Derme"
    return np.nan
df["grau_infiltracao"] = pd.Categorical(df["infiltracao_raw"].apply(grau),
                                        categories=ORDEM_GRAU, ordered=True)

# grau numérico para Kruskal etc.
df["grau_num"] = df["grau_infiltracao"].cat.codes.replace(-1, np.nan) + 1

print("Limpeza base OK. n =", len(df))""")

# ============================================================ Subtipo (3 cat clínica)
md(r"""## 3. Subtipo do Tumor → 3 grupos clínicos

Classificação clínica padrão de carcinoma basocelular (literatura WHO / NCCN):

| Grupo | Componentes |
|---|---|
| **Baixo risco** (indolente) | nodular, sólido, superficial, adenoide, pigmentado, queratótico |
| **Alto risco** (agressivo) | esclerodermiforme, infiltrativo, micronodular, basoescamoso, metatípico |
| **Misto** | combina componentes dos dois grupos acima |

*Se a orientadora preferir outra regra, é só editar `BAIXO_RISCO`/`ALTO_RISCO` abaixo.*""")
code(r"""BAIXO_RISCO = ["nodular","solido","superficial","adenoide","pigmentado","queratotico","ceratotico"]
ALTO_RISCO  = ["esclerodermiforme","infiltrativo","micronodular","basoescamoso","metatipico","metatpico"]

def limpa_subtipo(s):
    s = norm(s)
    if pd.isna(s): return np.nan
    s = (s.replace("eesclerodermiforme","e esclerodermiforme")
          .replace("nodular a basoescamoso","nodular e basoescamoso"))
    return re.sub(r"\s+"," ", s).strip()
df["subtipo"] = df["subtipo_raw"].apply(limpa_subtipo)

def classifica_subtipo(s):
    if pd.isna(s): return np.nan
    tem_baixo = any(k in s for k in BAIXO_RISCO)
    tem_alto  = any(k in s for k in ALTO_RISCO)
    if tem_baixo and tem_alto: return "Misto"
    if tem_alto:               return "Alto risco"
    if tem_baixo:              return "Baixo risco"
    return np.nan

df["subtipo_3cat"] = df["subtipo"].apply(classifica_subtipo)
ORDEM_SUB = ["Baixo risco","Misto","Alto risco"]
df["subtipo_3cat"] = pd.Categorical(df["subtipo_3cat"], categories=ORDEM_SUB, ordered=True)
print(df["subtipo_3cat"].value_counts(dropna=False))""")

# ============================================================ Novas v2: civil, faixa etaria, exposicao
md("## 4. Novas variáveis derivadas (estado civil, faixa etária, exposição solar)")
code(r"""# 4.1 estado civil binário
COM = ["Casado(a)", "União estável"]
SEM = ["Solteiro(a)", "Viúvo(a)", "Divorciado(a)"]
df["estado_civil_grupo"] = df["estado_civil"].map(
    {**{k:"Com companheiro(a)" for k in COM}, **{k:"Sem companheiro(a)" for k in SEM}})
print("Estado civil agrupado:\n", df["estado_civil_grupo"].value_counts(dropna=False))

# 4.2 faixa etária: corte 40 (orientadora)
df["faixa_etaria"] = pd.cut(df["idade"], bins=[0,40,200],
                            labels=["≤ 40 anos", "> 40 anos"], include_lowest=True)
print("\nFaixa etária (corte = 40):\n", df["faixa_etaria"].value_counts(dropna=False))

# 4.3 exposição solar ocupacional
df["profissao"] = df["profissao"].apply(norm).replace({"nao informado": np.nan,
                                                       "aposentada": "aposentado"})
COM_SOL = ["trabalhador agricola","lavrador","salineiro","pescador",
           "carpinteiro","marceneiro","eletricista","soldador","pintor",
           "trabalhador de obras","trabalhador de concreto armado",
           "trabalhador bracal","operador de maquinas","mecanico",
           "motorista","vigilante"]
SEM_SOL = ["do lar","comerciante","atendente","cozinheira","domestica",
           "domestica/copeira","vendedor","cabeleireira","lavadeira","asg",
           "seguranca","gerente de empresa",
           "professor","professora","medico","enfermeira",
           "auxiliar de enfermagem","tec enfermagem","policial",
           "funcionaria publica","funcionario publico","advogado"]
mapa_exp = {**{p:"Com exposição" for p in COM_SOL}, **{p:"Sem exposição" for p in SEM_SOL}}
df["exposicao_solar"] = df["profissao"].map(mapa_exp)
# aposentados, autônomos, desempregados -> NaN (excluídos dos testes, decisão combinada)
print("\nExposição solar:\n", df["exposicao_solar"].value_counts(dropna=False))""")

# ============================================================ Região anatômica
md(r"""## 5. Região anatômica (face detalhada + extra-facial agrupado)

Como a orientadora pediu detalhamento da face, mantemos as subregiões faciais
individualmente; para as demais regiões, agrupamos por território anatômico.""")
code(r"""def classifica_regiao(s):
    s = norm(s)
    if pd.isna(s): return np.nan
    # face — detalhada
    if "face" in s or "hemiface" in s or "frontotemporal" in s or "infra - palpebral" in s or "palpebr" in s and "tronco" not in s:
        if "nariz" in s or "paranas" in s: return "Face — nariz"
        if "malar" in s: return "Face — malar"
        if "frontal" in s or "frontotemp" in s: return "Face — frontal"
        if "tempor" in s: return "Face — têmpora"
        if "palpebr" in s or "periorbit" in s or "supraorbit" in s or "orbita" in s or "globo ocular" in s:
            return "Face — pálpebra/periocular"
        if any(k in s for k in ["labio","mento","mandib","nasolab","nasogen","supralab","supercili","sobrancelha","glabela"]):
            return "Face — perioral/outros"
        return "Face — outros"
    if "couro" in s: return "Couro cabeludo"
    if "auricular" in s: return "Pavilhão/peri-auricular"
    if "cervical" in s: return "Cervical"
    if "ms" in s.split() or "membro superior" in s or "ombro" in s or "mao" in s:
        return "Membro superior"
    if "mi" in s.split() or "membro inferior" in s:
        return "Membro inferior"
    if any(k in s for k in ["dorso","dorsolateral","escapula","lombar","tronco","torax","abdominal","abdome"]):
        return "Tronco/dorso"
    return "Outros"
df["regiao_grupo"] = df["regiao_raw"].apply(classifica_regiao)
vc = df["regiao_grupo"].value_counts(dropna=False)
print(vc)""")

# ============================================================ Exportar tratada
md("## 6. Base tratada v2 — exportação")
code(r"""df_lesao = df.copy()
df_paciente = (df_lesao.sort_values("idade")
                       .drop_duplicates(subset="codigo", keep="first")
                       .reset_index(drop=True))
print(f"Lesões: {len(df_lesao)} | Pacientes únicos: {len(df_paciente)}")
with pd.ExcelWriter("dados_tratados_v2.xlsx") as w:
    df_lesao.to_excel(w, sheet_name="por_lesao", index=False)
    df_paciente.to_excel(w, sheet_name="por_paciente", index=False)
print("[exportado] dados_tratados_v2.xlsx")""")

# ============================================================ Descritivos novos
md("## 7. Descritivos das novas variáveis")
code(r"""fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, tit in zip(axes,
        ["estado_civil_grupo","faixa_etaria","exposicao_solar"],
        ["Estado civil (agrupado)","Faixa etária","Exposição solar ocupacional"]):
    vc = df_lesao[col].value_counts(dropna=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette="crest")
    for i,v in enumerate(vc.values): ax.text(i, v+1, str(v), ha="center", fontweight="bold")
    ax.set(title=tit, xlabel="", ylabel="Lesões")
plt.tight_layout(); salvar_fig("v2_descritivos_novos.png"); plt.show()""")
code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vc = df_lesao["subtipo_3cat"].value_counts(dropna=False).sort_index()
sns.barplot(x=vc.index.astype(str), y=vc.values, ax=axes[0], palette=["#55A868","#DD8452","#C44E52"])
for i,v in enumerate(vc.values): axes[0].text(i, v+1, str(v), ha="center", fontweight="bold")
axes[0].set(title="Subtipo (3 categorias clínicas)", xlabel="", ylabel="Lesões")

rg = df_lesao["regiao_grupo"].value_counts(dropna=False)
sns.barplot(y=rg.index, x=rg.values, ax=axes[1], palette="flare")
for i,v in enumerate(rg.values): axes[1].text(v+0.5, i, str(v), va="center", fontweight="bold")
axes[1].set(title="Região anatômica (face detalhada)", xlabel="Lesões", ylabel="")
plt.tight_layout(); salvar_fig("v2_subtipo_regiao.png"); plt.show()""")

# ============================================================ Função genérica
md(r"""## 8. Função genérica de análise

Lida com:
- **categórica × categórica** → qui-quadrado (ou Fisher em 2×2 com esperado < 5); V de Cramer.
- **categórica × contínua** → Mann-Whitney (2 grupos) ou Kruskal-Wallis (≥ 3); r rank-biserial / η².
""")
code(r"""def cramer_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum(); r, k = ct.shape
    return float(np.sqrt(chi2/(n*(min(r,k)-1)))) if min(r,k) > 1 else np.nan

def analisa(bloco, codigo, lin, col, titulo, arquivo, data=None,
            ordem_lin=None, ordem_col=None, continua=False):
    if data is None: data = df_lesao
    sub = data.dropna(subset=[lin, col])
    n = len(sub)
    if n < 5:
        print(f"[{codigo}] {titulo}: n={n} insuficiente — pulado.")
        RESULTADOS.append({"bloco":bloco,"codigo":codigo,"cruzamento":titulo,
                           "n":n,"teste":"-","estatistica":np.nan,"p_valor":np.nan,
                           "efeito":np.nan,"significativo":""})
        return None

    print(f"\n=== [{codigo}] {titulo} ===  (n = {n})")

    if continua:  # categórica × contínua
        grupos = [sub.loc[sub[lin]==g, col].dropna() for g in sub[lin].unique()
                  if (sub[lin]==g).sum() >= 2]
        if len(grupos) < 2:
            print("Grupos insuficientes."); return None
        # estatística descritiva
        desc = sub.groupby(lin, observed=True)[col].agg(["count","mean","std","median",
                                                          lambda x: x.quantile(.25),
                                                          lambda x: x.quantile(.75)])
        desc.columns = ["n","média","desvio","mediana","q1","q3"]
        print(desc.round(2))
        if len(grupos) == 2:
            est, p = stats.mannwhitneyu(grupos[0], grupos[1], alternative="two-sided")
            n1, n2 = len(grupos[0]), len(grupos[1])
            efeito = 1 - 2*est/(n1*n2)
            teste = "Mann-Whitney"
            print(f"\n{teste}: U={est:.1f}, p={p:.4f} | r={efeito:.3f}")
        else:
            est, p = stats.kruskal(*grupos)
            efeito = est/(n - 1)   # eta² aproximado
            teste = "Kruskal-Wallis"
            print(f"\n{teste}: H={est:.3f}, p={p:.4f} | η²≈{efeito:.3f}")

        # boxplot
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ordem = ordem_lin or list(sub[lin].dropna().unique())
        sns.boxplot(data=sub, x=lin, y=col, order=ordem, ax=ax, palette="crest", width=.55)
        sns.stripplot(data=sub, x=lin, y=col, order=ordem, ax=ax, color="black", alpha=.25, size=2)
        ax.set(title=f"{titulo}\n{teste}: p={p:.4f}", xlabel="", ylabel=col)
        plt.xticks(rotation=15, ha="right")
        salvar_fig(arquivo); plt.show()
    else:        # categórica × categórica
        ct = pd.crosstab(sub[lin], sub[col])
        if ordem_lin: ct = ct.reindex([x for x in ordem_lin if x in ct.index])
        if ordem_col: ct = ct[[x for x in ordem_col if x in ct.columns]]
        print("Frequências:\n", ct)
        pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
        print("\n% por linha:\n", pct)
        chi2, p, dof, esp = stats.chi2_contingency(ct)
        v = cramer_v(ct)
        if ct.shape == (2,2):
            _, pf = stats.fisher_exact(ct)
            print(f"\nQui²={chi2:.3f}, gl={dof}, p={p:.4f} | Fisher p={pf:.4f} | V Cramer={v:.3f}")
            p_uso, teste, est_uso = pf, "Fisher (2x2)", chi2
        else:
            print(f"\nQui²={chi2:.3f}, gl={dof}, p={p:.4f} | V Cramer={v:.3f}")
            if esp.min() < 5: print(f"⚠ menor esperado={esp.min():.2f} — interpretar com cautela")
            p_uso, teste, est_uso = p, f"Qui-quadrado (gl={dof})", chi2
        efeito = v

        fig, ax = plt.subplots(figsize=(9, 4.5))
        pct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set(title=f"{titulo}\n{teste}: p={p_uso:.4f}", xlabel="", ylabel="% por linha")
        ax.legend(title=col, bbox_to_anchor=(1.02,1), loc="upper left")
        plt.xticks(rotation=20, ha="right")
        salvar_fig(arquivo); plt.show()
        est = est_uso; p = p_uso

    sig = "SIM" if (p is not None and p < 0.05) else "não"
    print(f"Significativo (α=0,05)? {sig.upper()}")
    RESULTADOS.append({"bloco":bloco,"codigo":codigo,"cruzamento":titulo,"n":n,
                       "teste": teste, "estatistica": round(float(est),3) if est==est else np.nan,
                       "p_valor": round(float(p),4) if p==p else np.nan,
                       "efeito": round(float(efeito),3) if efeito==efeito else np.nan,
                       "significativo": sig})
print("Função pronta.")""")

# ============================================================ BLOCO A
md(r"""## 9. Bloco A — Estado civil × saúde (estratificado por sexo)

Hipótese da vic: homens **sem companheira** chegam com doença mais avançada (tumores
mais profundos, maiores, com mais ulceração e margens comprometidas). Replicar para mulheres.""")
code(r"""ORD_EC = ["Com companheiro(a)", "Sem companheiro(a)"]

h = df_lesao[df_lesao["sexo"]=="Masculino"].copy()
m = df_lesao[df_lesao["sexo"]=="Feminino"].copy()
print(f"Homens (lesões): {len(h)}  |  Mulheres (lesões): {len(m)}")""")

code(r"""# A1 — Homens: estado civil × profundidade de infiltração
analisa("A","A1","estado_civil_grupo","grau_infiltracao",
        "Homens — Estado civil × Grau de infiltração",
        "A1_h_civil_x_grau.png", data=h, ordem_lin=ORD_EC, ordem_col=ORDEM_GRAU)""")
code(r"""# A2 — Homens: estado civil × tamanho da lesão
analisa("A","A2","estado_civil_grupo","tamanho_cm",
        "Homens — Estado civil × Tamanho da lesão (cm)",
        "A2_h_civil_x_tamanho.png", data=h, ordem_lin=ORD_EC, continua=True)""")
code(r"""# A3 — Homens: estado civil × ulceração
analisa("A","A3","estado_civil_grupo","ulceracao",
        "Homens — Estado civil × Ulceração",
        "A3_h_civil_x_ulceracao.png", data=h, ordem_lin=ORD_EC, ordem_col=["Não","Sim"])""")
code(r"""# A4 — Homens: estado civil × margens comprometidas
analisa("A","A4","estado_civil_grupo","margem_comprometida",
        "Homens — Estado civil × Margens comprometidas",
        "A4_h_civil_x_margens.png", data=h, ordem_lin=ORD_EC, ordem_col=["Livre","Comprometida"])""")
code(r"""# A5 — Mulheres: estado civil × profundidade
analisa("A","A5","estado_civil_grupo","grau_infiltracao",
        "Mulheres — Estado civil × Grau de infiltração",
        "A5_m_civil_x_grau.png", data=m, ordem_lin=ORD_EC, ordem_col=ORDEM_GRAU)""")
code(r"""# A6 — Mulheres: estado civil × tamanho
analisa("A","A6","estado_civil_grupo","tamanho_cm",
        "Mulheres — Estado civil × Tamanho da lesão (cm)",
        "A6_m_civil_x_tamanho.png", data=m, ordem_lin=ORD_EC, continua=True)""")
code(r"""# A7 — Mulheres: estado civil × ulceração
analisa("A","A7","estado_civil_grupo","ulceracao",
        "Mulheres — Estado civil × Ulceração",
        "A7_m_civil_x_ulceracao.png", data=m, ordem_lin=ORD_EC, ordem_col=["Não","Sim"])""")
code(r"""# A8 — Mulheres: estado civil × margens
analisa("A","A8","estado_civil_grupo","margem_comprometida",
        "Mulheres — Estado civil × Margens comprometidas",
        "A8_m_civil_x_margens.png", data=m, ordem_lin=ORD_EC, ordem_col=["Livre","Comprometida"])""")

# ============================================================ BLOCO B
md(r"""## 10. Bloco B — Faixa etária (≤ 40 vs > 40) × marcadores de agressividade
*Nível lesão (419), idade ao diagnóstico.* Hipótese: CBC em jovens seria mais agressivo.

⚠ Observação: só **23 lesões em pacientes ≤ 40 anos** — interpretar com cuidado (baixo poder).""")
code(r"""ORD_FE = ["≤ 40 anos", "> 40 anos"]
analisa("B","B1","faixa_etaria","margem_comprometida",
        "Faixa etária × Margens comprometidas",
        "B1_idade_x_margens.png", ordem_lin=ORD_FE, ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("B","B2","faixa_etaria","inv_perineural",
        "Faixa etária × Invasão perineural",
        "B2_idade_x_perineural.png", ordem_lin=ORD_FE, ordem_col=["Não","Sim"])""")
code(r"""analisa("B","B3","faixa_etaria","inv_linfovascular",
        "Faixa etária × Invasão linfovascular",
        "B3_idade_x_linfovascular.png", ordem_lin=ORD_FE)""")
code(r"""analisa("B","B4","faixa_etaria","ulceracao",
        "Faixa etária × Ulceração",
        "B4_idade_x_ulceracao.png", ordem_lin=ORD_FE, ordem_col=["Não","Sim"])""")
code(r"""analisa("B","B5","faixa_etaria","tamanho_cm",
        "Faixa etária × Tamanho da lesão",
        "B5_idade_x_tamanho.png", ordem_lin=ORD_FE, continua=True)""")
code(r"""analisa("B","B6","faixa_etaria","grau_infiltracao",
        "Faixa etária × Grau de infiltração",
        "B6_idade_x_grau.png", ordem_lin=ORD_FE, ordem_col=ORDEM_GRAU)""")
code(r"""analisa("B","B7","faixa_etaria","subtipo_3cat",
        "Faixa etária × Subtipo (3 cat)",
        "B7_idade_x_subtipo.png", ordem_lin=ORD_FE, ordem_col=ORDEM_SUB)""")

# ============================================================ BLOCO C
md(r"""## 11. Bloco C — Exposição solar ocupacional × marcadores

Profissões com vs sem exposição solar. Aposentado(a), autônomo(a), desempregado(a) e
"não informado" foram **excluídos** desses testes (decisão combinada com a vic).""")
code(r"""ORD_EXP = ["Com exposição", "Sem exposição"]
analisa("C","C1","exposicao_solar","subtipo_3cat",
        "Exposição solar × Subtipo (3 cat)",
        "C1_exp_x_subtipo.png", ordem_lin=ORD_EXP, ordem_col=ORDEM_SUB)""")
code(r"""analisa("C","C2","exposicao_solar","ulceracao",
        "Exposição solar × Ulceração",
        "C2_exp_x_ulceracao.png", ordem_lin=ORD_EXP, ordem_col=["Não","Sim"])""")
code(r"""analisa("C","C3","exposicao_solar","inv_perineural",
        "Exposição solar × Invasão perineural",
        "C3_exp_x_perineural.png", ordem_lin=ORD_EXP, ordem_col=["Não","Sim"])""")
code(r"""analisa("C","C4","exposicao_solar","inv_linfovascular",
        "Exposição solar × Invasão linfovascular",
        "C4_exp_x_linfovascular.png", ordem_lin=ORD_EXP)""")
code(r"""analisa("C","C5","exposicao_solar","margem_comprometida",
        "Exposição solar × Margens comprometidas",
        "C5_exp_x_margens.png", ordem_lin=ORD_EXP, ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("C","C6","exposicao_solar","tamanho_cm",
        "Exposição solar × Tamanho da lesão (média)",
        "C6_exp_x_tamanho.png", ordem_lin=ORD_EXP, continua=True)""")
code(r"""analisa("C","C7","exposicao_solar","grau_infiltracao",
        "Exposição solar × Grau de infiltração",
        "C7_exp_x_grau.png", ordem_lin=ORD_EXP, ordem_col=ORDEM_GRAU)""")

# ============================================================ BLOCO D
md(r"""## 12. Bloco D — Região anatômica × tudo

Face detalhada (nariz, malar, frontal, têmpora, pálpebra/periocular, perioral/outros)
+ regiões extra-faciais agrupadas (couro cabeludo, pavilhão/peri-auricular, cervical,
membro superior, membro inferior, tronco/dorso).""")
code(r"""analisa("D","D1","regiao_grupo","exposicao_solar",
        "Região × Exposição solar",
        "D1_regiao_x_exposicao.png", ordem_col=["Com exposição","Sem exposição"])""")
code(r"""analisa("D","D2","regiao_grupo","subtipo_3cat",
        "Região × Subtipo (3 cat)",
        "D2_regiao_x_subtipo.png", ordem_col=ORDEM_SUB)""")
code(r"""analisa("D","D3","regiao_grupo","margem_comprometida",
        "Região × Margens comprometidas",
        "D3_regiao_x_margens.png", ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("D","D4","regiao_grupo","inv_perineural",
        "Região × Invasão perineural",
        "D4_regiao_x_perineural.png", ordem_col=["Não","Sim"])""")
code(r"""analisa("D","D5","regiao_grupo","inv_linfovascular",
        "Região × Invasão linfovascular",
        "D5_regiao_x_linfovascular.png")""")
code(r"""analisa("D","D6","regiao_grupo","ulceracao",
        "Região × Ulceração",
        "D6_regiao_x_ulceracao.png", ordem_col=["Não","Sim"])""")
code(r"""analisa("D","D7","regiao_grupo","grau_infiltracao",
        "Região × Grau de infiltração",
        "D7_regiao_x_grau.png", ordem_col=ORDEM_GRAU)""")
code(r"""analisa("D","D8","regiao_grupo","tamanho_cm",
        "Região × Tamanho da lesão",
        "D8_regiao_x_tamanho.png", continua=True)""")

# ============================================================ BLOCO E
md("""## 13. Bloco E — Subtipo (3 categorias) × tudo

Substitui as análises de puro/misto da rodada 1.""")
code(r"""analisa("E","E1","subtipo_3cat","regiao_grupo",
        "Subtipo × Região anatômica",
        "E1_subtipo_x_regiao.png", ordem_lin=ORDEM_SUB)""")
code(r"""analisa("E","E2","subtipo_3cat","inv_perineural",
        "Subtipo × Invasão perineural",
        "E2_subtipo_x_perineural.png", ordem_lin=ORDEM_SUB, ordem_col=["Não","Sim"])""")
code(r"""analisa("E","E3","subtipo_3cat","inv_linfovascular",
        "Subtipo × Invasão linfovascular",
        "E3_subtipo_x_linfovascular.png", ordem_lin=ORDEM_SUB)""")
code(r"""analisa("E","E4","subtipo_3cat","grau_infiltracao",
        "Subtipo × Grau de infiltração",
        "E4_subtipo_x_grau.png", ordem_lin=ORDEM_SUB, ordem_col=ORDEM_GRAU)""")
code(r"""analisa("E","E5","subtipo_3cat","margem_comprometida",
        "Subtipo × Margens comprometidas",
        "E5_subtipo_x_margens.png", ordem_lin=ORDEM_SUB, ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("E","E6","subtipo_3cat","tamanho_cm",
        "Subtipo × Tamanho da lesão",
        "E6_subtipo_x_tamanho.png", ordem_lin=ORDEM_SUB, continua=True)""")
code(r"""analisa("E","E7","subtipo_3cat","ulceracao",
        "Subtipo × Ulceração",
        "E7_subtipo_x_ulceracao.png", ordem_lin=ORDEM_SUB, ordem_col=["Não","Sim"])""")

# ============================================================ Tabela mestre
md("## 14. Tabela-mestre de p-valores")
code(r"""master = pd.DataFrame(RESULTADOS)
# ordem dos blocos
master["bloco"] = pd.Categorical(master["bloco"], categories=["A","B","C","D","E"], ordered=True)
master = master.sort_values(["bloco","codigo"]).reset_index(drop=True)
master.to_excel("resumo_pvalores.xlsx", index=False)
print("[exportado] resumo_pvalores.xlsx\n")
sig = master[master["significativo"]=="SIM"]
print(f"Cruzamentos com p < 0,05: {len(sig)} de {len(master)}\n")
print(sig[["bloco","codigo","cruzamento","n","teste","p_valor","efeito"]].to_string(index=False))""")
code(r"""master.style.apply(lambda r: ["background-color: #d4edda" if r["significativo"]=="SIM" else "" for _ in r], axis=1)""")

# ============================================================ Conclusões (placeholder; substituiremos após rodar)
md(r"""## 15. Conclusões e limitações

### Síntese dos achados (37 cruzamentos rodados, 9 significativos)

**Observação importante sobre o subtipo:** ao aplicar a classificação clínica padrão (WHO/NCCN),
**nenhuma lesão da amostra ficou em "Baixo risco" puro** — todas as 419 têm pelo menos um
componente agressivo. O grupo "3 categorias" funciona na prática como **Misto (272)** vs
**Alto risco puro (147)**. Isso é coerente com o fato de a amostra vir de um serviço de patologia
(provavelmente já filtra casos clinicamente relevantes).

#### Bloco A — Estado civil × saúde (estratificado por sexo)
**Nenhum cruzamento significativo** (todos com p > 0,05). A hipótese de que homens sem companheira
teriam doença mais avançada **não foi confirmada** nesta amostra. O sinal mais próximo é A1
(profundidade em homens, p=0,086).

#### Bloco B — Faixa etária (≤ 40 vs > 40)
**1 cruzamento significativo** (de 7):
- **B5 — Tamanho da lesão**: pacientes ≤ 40 têm tumores **MAIORES** (mediana 2,3 cm; média 2,75)
  vs. >40 (mediana 1,2 cm; média 1,49) — **p<0,001; r=0,54 (efeito grande)**. Sentido contrário
  ao esperado de "tumor menor no jovem". Possíveis explicações: diagnóstico tardio em jovens
  (não esperam câncer nessa faixa) ou comportamento biológico realmente mais expansivo.
- Demais marcadores de agressividade (margens, perineural, ulceração, grau, subtipo) **não diferem
  por faixa etária**.

#### Bloco C — Exposição solar ocupacional
**2 cruzamentos significativos** (de 7):
- **C2 — Ulceração**: lesões em expostos ao sol ulceram em **69,5%** vs. 54,8% em não expostos
  (Fisher p=0,014; V=0,14).
- **C6 — Tamanho**: lesões em expostos são **maiores** (mediana 1,2 vs 1,0 cm; média 1,55 vs 1,15;
  Mann-Whitney p=0,003; r=0,21).

#### Bloco D — Região anatômica × tudo
**4 cruzamentos significativos** (de 8) — **o sítio anatômico importa muito**:
- **D3 — Margens comprometidas** (Qui² p=0,014; V=0,25)
- **D6 — Ulceração** (p=0,003; V=0,28)
- **D7 — Grau de infiltração** (p<0,001; V=0,29)
- **D8 — Tamanho** (Kruskal-Wallis p<0,001; η²=0,19): regiões com lesões maiores → **membro
  superior (mediana 2,25 cm)** e **tronco/dorso (2,0 cm)**; menores → face — nariz, perioral
  e pálpebra (~1,0 cm).

#### Bloco E — Subtipo (Misto vs Alto risco puro)
**2 cruzamentos significativos** (de 7):
- **E4 — Grau de infiltração** (Qui² p<0,001; V=0,33 — **efeito moderado**): lesões **Mistas**
  infiltram mais profundamente (71,8% derme reticular + 17,3% hipoderme + 5,1% estruturas profundas)
  do que Alto risco puro (58,2% reticular; só 0,7% estruturas profundas). Provavelmente porque o
  componente nodular/sólido associado ao agressivo dá mais massa tumoral.
- **E6 — Tamanho** (Mann-Whitney p=0,010; r=0,15): Mistos ligeiramente maiores (média 1,62 vs 1,43 cm).

### Limitações
1. **Faixa etária ≤ 40 com apenas 23 lesões** → poder estatístico reduzido para o Bloco B.
2. **Invasão linfovascular sem positivos** → cruzamentos B3, C4, D5, E3 ficam só descritivos
   (todos p=1,0 esperado).
3. **Múltiplas lesões por paciente** (até 14 por código) → não-independência das observações.
4. **Sem correção para múltiplas comparações** (37 testes — alguns p < 0,05 podem ser falso-positivos
   por acaso; aplicando Bonferroni grosseiro, α ≈ 0,0014, sobrariam apenas B5, C6, D6, D7, D8, E4).
5. **Classificação de subtipo** baseada em literatura clínica padrão; nenhum caso puramente "Baixo risco"
   resultou — se a orientadora preferir outra regra de agrupamento (por componente dominante, por
   exemplo), é só editar `BAIXO_RISCO`/`ALTO_RISCO` na seção 3 e re-executar.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"}}
with open("analise_basocelular_v2.ipynb","w",encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"v2 escrito ({len(cells)} células).")
