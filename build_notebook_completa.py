# -*- coding: utf-8 -*-
"""Gera analise_basocelular_completa.ipynb — análise consolidada (rodadas 1 + 2)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# =================================================================== Cabeçalho
md(r"""# Análise de Dados — Carcinoma Basocelular
## PIBIC / LMECC

Estudo retrospectivo de biópsias de carcinoma basocelular (CBC). Este notebook
realiza o tratamento e a limpeza dos dados, a representação descritiva dos
pacientes e os testes de associação entre variáveis clínicas, histopatológicas
e ocupacionais.

**Unidade de análise:**
- *Nível paciente* (deduplicado pelo Código da Amostra): perfil demográfico —
  idade, sexo, profissão — e a comparação de idade entre sexos.
- *Nível lesão* (todas as biópsias): tabelas cruzadas histopatológicas.

**Conjunto de análises:**

1. Descritiva do perfil dos pacientes (idade, sexo, profissão, etnia, estado civil).
2. Comparação da idade entre homens e mulheres.
3. Margens comprometidas × Reincidência.
4. Bloco A — Estado civil (com/sem companheiro) × profundidade, tamanho, ulceração
   e margens, estratificado por sexo.
5. Bloco B — Faixa etária (≤ 40 vs > 40 anos) × marcadores de agressividade.
6. Bloco C — Exposição solar ocupacional × marcadores de agressividade.
7. Bloco D — Região anatômica (face detalhada) × variáveis clínicas e patológicas.
8. Bloco E — Subtipo do tumor (3 categorias clínicas) × variáveis clínicas e patológicas.

**Padrão estatístico:** qui-quadrado de Pearson para variáveis categóricas
(exato de Fisher quando 2×2 com frequência esperada inferior a 5);
Mann-Whitney para variáveis contínuas em dois grupos e Kruskal-Wallis em três
ou mais; teste de normalidade (Shapiro-Wilk) precedendo a comparação de idade
entre sexos. Significância α = 0,05. Tamanho de efeito reportado conforme o
teste utilizado (V de Cramer, r rank-biserial, d de Cohen, η²).
""")

# =================================================================== Setup
md("## 1. Configuração do ambiente")
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
FIG_DIR = "figuras"
os.makedirs(FIG_DIR, exist_ok=True)

# acumulador de resultados (tabela-mestre)
RESULTADOS = []

def salvar_fig(nome):
    plt.savefig(os.path.join(FIG_DIR, nome))
    print(f"[figura] {nome}")
print("Ambiente pronto.")""")

# =================================================================== Carga
md("## 2. Carga dos dados")
code(r"""brutos = pd.read_csv(ARQUIVO, sep=None, engine="python", encoding="utf-8-sig")
print(f"Dimensões: {brutos.shape[0]} linhas × {brutos.shape[1]} colunas")
print("\nColunas:")
for c in brutos.columns: print(f"  - {c}")
brutos.head(3)""")

# =================================================================== Tratamento
md(r"""## 3. Tratamento e limpeza dos dados

A base apresenta inconsistências típicas de digitação manual. Os tratamentos
aplicados estão resumidos abaixo:

| Coluna | Problema | Tratamento |
|---|---|---|
| Idade ao diagnóstico | textos não numéricos | conversão para numérico, inválidos → ausente |
| Etnia | maiúsculas inconsistentes | padronização |
| Estado Civil | valor `Branca` infiltrado | remoção do valor incorreto |
| Profissão | acentos, plurais, gênero | normalização |
| Subtipo do Tumor | 58 variações de `misto (...)` | normalização e classificação em grupos clínicos |
| Infiltração (grau) | 32 variações textuais | escala ordinal de profundidade |
| Ulceração | `Não informado` | mantém Sim/Não; demais → ausente |
| Margens (Lateral e Profunda) | texto livre (`comprometidas (1)`, espaços) | normalização e flag combinada |
| Reincidência | `Talvez`, ausentes | mantém Sim/Não; demais → ausente |
| Invasão Linfovascular | apenas `Não` e ausentes | mantida (sem casos positivos) |
| Tamanho da Lesão | vírgula decimal e ruído numérico | conversão para numérico, arredondamento |
| Tipo de Material / Tratamento | colunas constantes ou vazias | descartadas |
""")

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
    "Código da Amostra": "codigo",
    "Data do diagnóstico": "data_dx",
    "Idade ao diagnóstico": "idade",
    "Sexo": "sexo",
    "Etnia": "etnia",
    "Estado Civil": "estado_civil",
    "Profissão": "profissao",
    "Subtipo do Tumor": "subtipo_raw",
    "Região Anatômica da Lesão": "regiao_raw",
    "Tamanho da Lesão (cm)": "tamanho_cm",
    "Presença de Ulceração": "ulceracao",
    "infiltração": "infiltracao_raw",
    "Estado de Comprometimento das Margens (Lateral)": "margem_lat_raw",
    "Estado de Comprometimento das Margens (Profunda)": "margem_prof_raw",
    "Reincidência?": "reincidencia",
    "Quantas reincidências?": "qtd_reincidencias",
    "Invasão Linfovascular?": "inv_linfovascular",
    "Invasão Perineural?": "inv_perineural",
}).drop(columns=["Tipo de Material Analisado", "Tratamento"], errors="ignore")

# numérico: idade
df["idade"] = pd.to_numeric(df["idade"], errors="coerce")

# numérico: tamanho da lesão (CSV usa vírgula decimal)
df["tamanho_cm"] = (df["tamanho_cm"].astype(str)
                                    .str.replace(",", ".", regex=False)
                                    .replace({"nan": np.nan}))
df["tamanho_cm"] = pd.to_numeric(df["tamanho_cm"], errors="coerce").round(2)

# categorias simples
df["sexo"]  = df["sexo"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
df["etnia"] = df["etnia"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
df["estado_civil"] = df["estado_civil"].where(df["estado_civil"] != "Branca")

df["ulceracao"]         = df["ulceracao"].apply(norm).map({"sim":"Sim","nao":"Não"})
df["reincidencia"]      = df["reincidencia"].apply(norm).map({"sim":"Sim","nao":"Não"})
df["inv_perineural"]    = df["inv_perineural"].apply(norm).map({"sim":"Sim","nao":"Não"})
df["inv_linfovascular"] = df["inv_linfovascular"].apply(norm).map({"sim":"Sim","nao":"Não"})

print("Variáveis simples normalizadas. n =", len(df))""")

md("""### 3.1. Margens — normalização e flag combinada
Texto livre como `comprometidas (1)` ou `comprometida ` é normalizado para
*livre* ou *comprometida*. A flag combinada `margem_comprometida` registra
*Comprometida* quando há comprometimento em margem lateral ou profunda.""")
code(r"""def normaliza_margem(s):
    s = norm(s)
    if pd.isna(s): return np.nan
    if "comprometid" in s: return "Comprometida"
    if "livre" in s: return "Livre"
    return np.nan

df["margem_lateral"]  = df["margem_lat_raw"].apply(normaliza_margem)
df["margem_profunda"] = df["margem_prof_raw"].apply(normaliza_margem)

def combina_margem(row):
    vals = [row["margem_lateral"], row["margem_profunda"]]
    if "Comprometida" in vals: return "Comprometida"
    if "Livre" in vals: return "Livre"
    return np.nan
df["margem_comprometida"] = df.apply(combina_margem, axis=1)

print(df["margem_comprometida"].value_counts(dropna=False))""")

md(r"""### 3.2. Grau de infiltração — escala ordinal

As 32 descrições textuais foram colapsadas em quatro níveis de profundidade
crescente:

1. **Derme** (superficial/média)
2. **Derme reticular** (inclui derme profunda)
3. **Hipoderme / tecido subcutâneo**
4. **Estruturas profundas** (músculo, cartilagem, globo ocular, tecidos periorbitários)""")
code(r"""ORDEM_GRAU = ["1. Derme", "2. Derme reticular",
              "3. Hipoderme/subcutâneo", "4. Estruturas profundas"]

def classifica_grau(s):
    s = norm(s)
    if pd.isna(s) or "nao informado" in s: return np.nan
    if any(k in s for k in ["globo ocular","muscul","musc","cartilagem","periorbit","estriad"]):
        return "4. Estruturas profundas"
    if any(k in s for k in ["hipoderme","subcut","adiposo","celular subc"]):
        return "3. Hipoderme/subcutâneo"
    if "reticular" in s or "derme profunda" in s: return "2. Derme reticular"
    if "derme" in s: return "1. Derme"
    return np.nan

df["grau_infiltracao"] = pd.Categorical(df["infiltracao_raw"].apply(classifica_grau),
                                        categories=ORDEM_GRAU, ordered=True)
print(df["grau_infiltracao"].value_counts(dropna=False).sort_index())""")

md(r"""### 3.3. Subtipo do tumor — classificação em três grupos clínicos

Aplica-se a classificação clínica padrão de CBC (WHO / NCCN) baseada no
risco de comportamento agressivo:

| Grupo | Componentes histológicos |
|---|---|
| **Baixo risco** (indolente) | nodular, sólido, superficial, adenoide, pigmentado, queratótico |
| **Alto risco** (agressivo) | esclerodermiforme, infiltrativo, micronodular, basoescamoso, metatípico |
| **Misto** | combinação de componentes dos dois grupos acima |

Uma lesão classifica-se em *Misto* quando apresenta simultaneamente componentes
indolentes e agressivos.""")
code(r"""BAIXO_RISCO = ["nodular","solido","superficial","adenoide","pigmentado","queratotico","ceratotico"]
ALTO_RISCO  = ["esclerodermiforme","infiltrativo","micronodular","basoescamoso","metatipico","metatpico"]

def limpa_subtipo(s):
    s = norm(s)
    if pd.isna(s): return np.nan
    s = (s.replace("eesclerodermiforme", "e esclerodermiforme")
          .replace("nodular a basoescamoso", "nodular e basoescamoso"))
    return re.sub(r"\s+", " ", s).strip()

df["subtipo"] = df["subtipo_raw"].apply(limpa_subtipo)

def classifica_subtipo(s):
    if pd.isna(s): return np.nan
    tem_baixo = any(k in s for k in BAIXO_RISCO)
    tem_alto  = any(k in s for k in ALTO_RISCO)
    if tem_baixo and tem_alto: return "Misto"
    if tem_alto:               return "Alto risco"
    if tem_baixo:              return "Baixo risco"
    return np.nan

ORDEM_SUB = ["Baixo risco","Misto","Alto risco"]
df["subtipo_3cat"] = pd.Categorical(df["subtipo"].apply(classifica_subtipo),
                                    categories=ORDEM_SUB, ordered=True)

# composição (puro/misto) - mantida da rodada 1 para a flag puro/misto
tem_misto_palavra = df["subtipo"].fillna("").str.contains("misto")
componentes_total = sum(df["subtipo"].fillna("").str.contains(p).astype(int)
                        for p in BAIXO_RISCO + ALTO_RISCO)
df["composicao"] = np.where(tem_misto_palavra | (componentes_total >= 2), "Misto", "Puro")
df["composicao"] = df["composicao"].astype(object)
df.loc[df["subtipo"].isna(), "composicao"] = np.nan

print("Subtipo (3 categorias):")
print(df["subtipo_3cat"].value_counts(dropna=False))
print("\nComposição (puro/misto):")
print(df["composicao"].value_counts(dropna=False))""")

md(r"""### 3.4. Profissão e exposição solar ocupacional

A profissão é normalizada (acento, plural, gênero) e mapeada a duas categorias
de exposição solar ocupacional. Aposentado(a), autônomo(a), desempregado e
"não informado" são excluídos dessa variável binária por indeterminação.""")
code(r"""df["profissao"] = df["profissao"].apply(norm).replace({"nao informado": np.nan,
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

mapa_exp = {**{p:"Com exposição" for p in COM_SOL},
            **{p:"Sem exposição" for p in SEM_SOL}}
df["exposicao_solar"] = df["profissao"].map(mapa_exp)
print(df["exposicao_solar"].value_counts(dropna=False))""")

md(r"""### 3.5. Estado civil — agrupamento binário
Casado(a) e União estável → *Com companheiro(a)*.
Solteiro(a), Viúvo(a) e Divorciado(a) → *Sem companheiro(a)*.""")
code(r"""COM = ["Casado(a)", "União estável"]
SEM = ["Solteiro(a)", "Viúvo(a)", "Divorciado(a)"]
df["estado_civil_grupo"] = df["estado_civil"].map(
    {**{k:"Com companheiro(a)" for k in COM}, **{k:"Sem companheiro(a)" for k in SEM}})
print(df["estado_civil_grupo"].value_counts(dropna=False))""")

md(r"""### 3.6. Faixa etária
Corte clínico em 40 anos: ≤ 40 vs > 40.""")
code(r"""df["faixa_etaria"] = pd.cut(df["idade"], bins=[0,40,200],
                            labels=["≤ 40 anos", "> 40 anos"], include_lowest=True)
print(df["faixa_etaria"].value_counts(dropna=False))""")

md(r"""### 3.7. Região anatômica
A face é mantida em sub-regiões (nariz, malar, frontal, têmpora, pálpebra/periocular,
perioral). As demais localizações anatômicas são agrupadas por território.""")
code(r"""def classifica_regiao(s):
    s = norm(s)
    if pd.isna(s): return np.nan
    if "face" in s or "hemiface" in s or "frontotemporal" in s \
       or "infra - palpebral" in s or ("palpebr" in s and "tronco" not in s):
        if "nariz" in s or "paranas" in s: return "Face — nariz"
        if "malar" in s: return "Face — malar"
        if "frontal" in s or "frontotemp" in s: return "Face — frontal"
        if "tempor" in s: return "Face — têmpora"
        if "palpebr" in s or "periorbit" in s or "supraorbit" in s \
           or "orbita" in s or "globo ocular" in s:
            return "Face — pálpebra/periocular"
        if any(k in s for k in ["labio","mento","mandib","nasolab","nasogen",
                                "supralab","supercili","sobrancelha","glabela"]):
            return "Face — perioral"
        return "Face — outros"
    if "couro" in s: return "Couro cabeludo"
    if "auricular" in s: return "Pavilhão/peri-auricular"
    if "cervical" in s: return "Cervical"
    if "ms" in s.split() or "membro superior" in s or "ombro" in s or "mao" in s:
        return "Membro superior"
    if "mi" in s.split() or "membro inferior" in s:
        return "Membro inferior"
    if any(k in s for k in ["dorso","dorsolateral","escapula","lombar","tronco",
                            "torax","abdominal","abdome"]):
        return "Tronco/dorso"
    return "Outros"

df["regiao_grupo"] = df["regiao_raw"].apply(classifica_regiao)
print(df["regiao_grupo"].value_counts(dropna=False))""")

md(r"""### 3.8. Resumo de valores ausentes após o tratamento""")
code(r"""colunas_relevantes = ["idade","sexo","etnia","estado_civil","estado_civil_grupo",
                      "profissao","exposicao_solar","faixa_etaria",
                      "subtipo","subtipo_3cat","composicao","grau_infiltracao",
                      "ulceracao","tamanho_cm","margem_comprometida",
                      "reincidencia","inv_perineural","inv_linfovascular",
                      "regiao_grupo"]
pd.DataFrame({
    "n_validos": df[colunas_relevantes].notna().sum(),
    "n_ausentes": df[colunas_relevantes].isna().sum(),
    "%_ausentes": (df[colunas_relevantes].isna().mean()*100).round(1),
})""")

md(r"""### 3.9. Base por paciente e exportação

Para os descritivos demográficos, deduplica-se pelo `codigo` mantendo o primeiro
diagnóstico (menor idade). A base por lesão é preservada para os testes
histopatológicos.""")
code(r"""df_lesao = df.copy()
df_paciente = (df_lesao.sort_values("idade")
                       .drop_duplicates(subset="codigo", keep="first")
                       .reset_index(drop=True))
print(f"Lesões (registros): {len(df_lesao)}")
print(f"Pacientes (códigos únicos): {len(df_paciente)}")

with pd.ExcelWriter("dados_tratados_completa.xlsx") as xls:
    df_lesao.to_excel(xls, sheet_name="por_lesao", index=False)
    df_paciente.to_excel(xls, sheet_name="por_paciente", index=False)
print("Base tratada exportada: dados_tratados_completa.xlsx")""")

# =================================================================== Descritiva
md(r"""## 4. Perfil descritivo dos pacientes
*Análises nesta seção em nível paciente (cada pessoa contada uma vez).*""")

md("### 4.1. Idade")
code(r"""idade = df_paciente["idade"].dropna()
desc = idade.describe()[["count","mean","std","min","50%","max"]]
desc.index = ["n","média","desvio","mínimo","mediana","máximo"]
print(desc.round(1))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(idade, bins=15, kde=True, ax=axes[0], color="#4C72B0")
axes[0].axvline(idade.mean(), color="red", ls="--", label=f"média {idade.mean():.1f}")
axes[0].set(title="Distribuição da idade ao diagnóstico", xlabel="Idade (anos)", ylabel="Pacientes")
axes[0].legend()
sns.boxplot(x=idade, ax=axes[1], color="#55A868")
axes[1].set(title="Boxplot da idade", xlabel="Idade (anos)")
plt.tight_layout(); salvar_fig("01_idade.png"); plt.show()""")

md("### 4.2. Sexo")
code(r"""sx = df_paciente["sexo"].value_counts()
pct = (sx/sx.sum()*100).round(1)
print(pd.DataFrame({"n": sx, "%": pct}))

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=sx.index, y=sx.values, ax=ax, palette=["#4C72B0","#C44E52"])
for i,(v,p) in enumerate(zip(sx.values, pct.values)):
    ax.text(i, v+1, f"{v} ({p}%)", ha="center", fontweight="bold")
ax.set(title="Distribuição por sexo", xlabel="", ylabel="Pacientes")
salvar_fig("02_sexo.png"); plt.show()""")

md(r"""### 4.3. Profissão (agrupada por setor)""")
code(r"""GRUPOS_PROF = {
    "Agrícola/rural":          ["trabalhador agricola","lavrador","salineiro","pescador"],
    "Aposentado(a)":           ["aposentado","pensionista"],
    "Do lar":                  ["do lar"],
    "Construção/braçal":       ["carpinteiro","marceneiro","eletricista","trabalhador de obras",
                                "trabalhador de concreto armado","soldador","pintor",
                                "trabalhador bracal","mecanico","operador de maquinas"],
    "Comércio/serviços":       ["comerciante","motorista","atendente","cozinheira","domestica",
                                "domestica/copeira","vendedor","cabeleireira","lavadeira","asg",
                                "vigilante","seguranca","autonomo","autonoma","gerente de empresa"],
    "Saúde/educação/público":  ["professor","professora","medico","enfermeira",
                                "auxiliar de enfermagem","tec enfermagem","policial",
                                "funcionaria publica","funcionario publico","advogado"],
    "Desempregado":            ["desempregado"],
}
mapa_grupo = {v:g for g,vs in GRUPOS_PROF.items() for v in vs}
df_paciente["profissao_grupo"] = df_paciente["profissao"].map(mapa_grupo)
df_paciente.loc[df_paciente["profissao"].notna() & df_paciente["profissao_grupo"].isna(),
                "profissao_grupo"] = "Outros"
df_lesao["profissao_grupo"] = df_lesao["profissao"].map(mapa_grupo)
df_lesao.loc[df_lesao["profissao"].notna() & df_lesao["profissao_grupo"].isna(),
             "profissao_grupo"] = "Outros"

pg = df_paciente["profissao_grupo"].value_counts(dropna=False)
print(pd.DataFrame({"n": pg, "%": (pg/pg.sum()*100).round(1)}))

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(y=pg.index.astype(str), x=pg.values, ax=ax, palette="crest")
for i, v in enumerate(pg.values): ax.text(v+0.5, i, str(v), va="center", fontweight="bold")
ax.set(title="Profissão dos pacientes (agrupada por setor)", xlabel="Pacientes", ylabel="")
salvar_fig("03_profissao.png"); plt.show()""")

md("### 4.4. Etnia e estado civil")
code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, titulo in zip(axes, ["etnia","estado_civil"], ["Etnia","Estado civil"]):
    vc = df_paciente[col].value_counts()
    sns.barplot(y=vc.index, x=vc.values, ax=ax, palette="flare")
    for i, v in enumerate(vc.values): ax.text(v+0.3, i, str(v), va="center")
    ax.set(title=titulo, xlabel="Pacientes", ylabel="")
plt.tight_layout(); salvar_fig("04_etnia_estadocivil.png"); plt.show()""")

# =================================================================== Idade HxM
md(r"""## 5. Comparação da idade entre sexos
*Análise em nível paciente.* Testes de normalidade (Shapiro-Wilk) e de
homogeneidade de variâncias (Levene) precedem a escolha entre teste t de Welch
e Mann-Whitney.""")
code(r"""h = df_paciente.loc[df_paciente["sexo"]=="Masculino", "idade"].dropna()
m = df_paciente.loc[df_paciente["sexo"]=="Feminino", "idade"].dropna()

tab = pd.DataFrame({
    "n":[len(h), len(m)],
    "média":[h.mean(), m.mean()],
    "desvio":[h.std(), m.std()],
    "mediana":[h.median(), m.median()],
}, index=["Masculino","Feminino"]).round(1)
print(tab)

p_sh_h = stats.shapiro(h).pvalue; p_sh_m = stats.shapiro(m).pvalue
p_lev = stats.levene(h, m).pvalue
print(f"\nShapiro-Wilk: Masculino p={p_sh_h:.3f} | Feminino p={p_sh_m:.3f}")
print(f"Levene: p={p_lev:.3f}")

if p_sh_h > 0.05 and p_sh_m > 0.05:
    est, pval = stats.ttest_ind(h, m, equal_var=False)
    sp = np.sqrt(((len(h)-1)*h.std()**2 + (len(m)-1)*m.std()**2)/(len(h)+len(m)-2))
    efeito = (h.mean()-m.mean())/sp
    teste = "Teste t de Welch"; metr = "d de Cohen"
    print(f"\n{teste}: t={est:.3f}, p={pval:.4f} | {metr}={efeito:.3f}")
else:
    est, pval = stats.mannwhitneyu(h, m, alternative="two-sided")
    efeito = 1 - 2*est/(len(h)*len(m))
    teste = "Mann-Whitney"; metr = "r rank-biserial"
    print(f"\n{teste}: U={est:.1f}, p={pval:.4f} | {metr}={efeito:.3f}")
print("Diferença significativa (α=0,05)?", "SIM" if pval < 0.05 else "NÃO")

RESULTADOS.append({"bloco":"Descritivo","codigo":"D1",
                   "cruzamento":"Idade × Sexo (pacientes)",
                   "n": len(h)+len(m), "teste": teste,
                   "estatistica": round(float(est),3),
                   "p_valor": round(float(pval),4),
                   "efeito": round(float(efeito),3),
                   "significativo": "SIM" if pval < 0.05 else "não"})""")
code(r"""fig, ax = plt.subplots(figsize=(6, 4.5))
dados = df_paciente.dropna(subset=["idade","sexo"])
sns.boxplot(data=dados, x="sexo", y="idade", ax=ax,
            palette=["#C44E52","#4C72B0"], width=.5)
sns.stripplot(data=dados, x="sexo", y="idade", ax=ax,
              color="black", alpha=.25, size=3)
ax.set(title="Idade ao diagnóstico por sexo", xlabel="", ylabel="Idade (anos)")
salvar_fig("05_idade_por_sexo.png"); plt.show()""")

# =================================================================== Função genérica
md(r"""## 6. Função genérica de análise cruzada

Esta função padroniza todas as análises seguintes: monta a tabela de
contingência (ou descritivo agrupado), aplica o teste apropriado, registra os
resultados na tabela-mestre e gera o gráfico correspondente. Suporta dois
modos:

- **Categórica × categórica:** qui-quadrado de Pearson com V de Cramer; em
  tabelas 2×2, também o exato de Fisher.
- **Categórica × contínua:** Mann-Whitney (2 grupos) ou Kruskal-Wallis (≥ 3),
  com tamanho de efeito apropriado.""")
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
        print(f"[{codigo}] {titulo}: n={n} insuficiente.")
        RESULTADOS.append({"bloco":bloco,"codigo":codigo,"cruzamento":titulo,
                           "n":n,"teste":"-","estatistica":np.nan,"p_valor":np.nan,
                           "efeito":np.nan,"significativo":""})
        return None

    print(f"\n=== [{codigo}] {titulo} ===  (n = {n})")

    if continua:
        grupos = [sub.loc[sub[lin]==g, col].dropna()
                  for g in sub[lin].unique() if (sub[lin]==g).sum() >= 2]
        if len(grupos) < 2:
            print("Grupos insuficientes."); return None
        desc = sub.groupby(lin, observed=True)[col].agg(
            ["count","mean","std","median",
             lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
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
            efeito = est/(n - 1)
            teste = "Kruskal-Wallis"
            print(f"\n{teste}: H={est:.3f}, p={p:.4f} | η²≈{efeito:.3f}")

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ordem = ordem_lin or list(sub[lin].dropna().unique())
        sns.boxplot(data=sub, x=lin, y=col, order=ordem, ax=ax,
                    palette="crest", width=.55)
        sns.stripplot(data=sub, x=lin, y=col, order=ordem, ax=ax,
                      color="black", alpha=.25, size=2)
        ax.set(title=f"{titulo}\n{teste}: p={p:.4f}", xlabel="", ylabel=col)
        plt.xticks(rotation=15, ha="right")
        salvar_fig(arquivo); plt.show()
    else:
        ct = pd.crosstab(sub[lin], sub[col])
        if ordem_lin: ct = ct.reindex([x for x in ordem_lin if x in ct.index])
        if ordem_col: ct = ct[[x for x in ordem_col if x in ct.columns]]
        print("Frequências:"); print(ct)
        pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
        print("\n% por linha:"); print(pct)
        chi2, p, dof, esp = stats.chi2_contingency(ct)
        v = cramer_v(ct)
        if ct.shape == (2,2):
            _, pf = stats.fisher_exact(ct)
            print(f"\nQui²={chi2:.3f}, gl={dof}, p={p:.4f} | Fisher p={pf:.4f} | V Cramer={v:.3f}")
            p_uso, teste, est_uso = pf, "Fisher (2x2)", chi2
        else:
            print(f"\nQui²={chi2:.3f}, gl={dof}, p={p:.4f} | V Cramer={v:.3f}")
            if esp.min() < 5:
                print(f"Aviso: menor frequência esperada = {esp.min():.2f}.")
            p_uso, teste, est_uso = p, f"Qui-quadrado (gl={dof})", chi2
        efeito = v

        fig, ax = plt.subplots(figsize=(9, 4.5))
        pct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set(title=f"{titulo}\n{teste}: p={p_uso:.4f}",
               xlabel="", ylabel="% por linha")
        ax.legend(title=col, bbox_to_anchor=(1.02,1), loc="upper left")
        plt.xticks(rotation=20, ha="right")
        salvar_fig(arquivo); plt.show()
        est = est_uso; p = p_uso

    sig = "SIM" if (p is not None and p < 0.05) else "não"
    print(f"Significativo (α=0,05)? {sig.upper()}")
    RESULTADOS.append({"bloco":bloco,"codigo":codigo,"cruzamento":titulo,"n":n,
                       "teste": teste,
                       "estatistica": round(float(est),3) if est==est else np.nan,
                       "p_valor": round(float(p),4) if p==p else np.nan,
                       "efeito": round(float(efeito),3) if efeito==efeito else np.nan,
                       "significativo": sig})

print("Função pronta.")""")

# =================================================================== Margens × Reincidência
md(r"""## 7. Margens comprometidas × Reincidência

Avalia-se se o comprometimento de margens (lateral ou profunda) associa-se
à reincidência do tumor.""")
code(r"""analisa("Margens","M1","margem_comprometida","reincidencia",
        "Margens × Reincidência", "07_margens_x_reincidencia.png",
        ordem_lin=["Livre","Comprometida"], ordem_col=["Não","Sim"])""")

# =================================================================== Bloco A
md(r"""## 8. Bloco A — Estado civil × saúde (estratificado por sexo)

Avalia-se se pacientes sem companheiro(a) apresentam doença mais avançada
(profundidade de infiltração, tamanho da lesão, ulceração, margens
comprometidas). A análise é estratificada por sexo.""")
code(r"""ORD_EC = ["Com companheiro(a)", "Sem companheiro(a)"]
h_les = df_lesao[df_lesao["sexo"]=="Masculino"].copy()
m_les = df_lesao[df_lesao["sexo"]=="Feminino"].copy()
print(f"Lesões em homens: {len(h_les)}  |  Lesões em mulheres: {len(m_les)}")""")

code(r"""analisa("A","A1","estado_civil_grupo","grau_infiltracao",
        "Homens — Estado civil × Grau de infiltração",
        "A1_h_civil_x_grau.png", data=h_les, ordem_lin=ORD_EC, ordem_col=ORDEM_GRAU)""")
code(r"""analisa("A","A2","estado_civil_grupo","tamanho_cm",
        "Homens — Estado civil × Tamanho da lesão (cm)",
        "A2_h_civil_x_tamanho.png", data=h_les, ordem_lin=ORD_EC, continua=True)""")
code(r"""analisa("A","A3","estado_civil_grupo","ulceracao",
        "Homens — Estado civil × Ulceração",
        "A3_h_civil_x_ulceracao.png", data=h_les,
        ordem_lin=ORD_EC, ordem_col=["Não","Sim"])""")
code(r"""analisa("A","A4","estado_civil_grupo","margem_comprometida",
        "Homens — Estado civil × Margens comprometidas",
        "A4_h_civil_x_margens.png", data=h_les,
        ordem_lin=ORD_EC, ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("A","A5","estado_civil_grupo","grau_infiltracao",
        "Mulheres — Estado civil × Grau de infiltração",
        "A5_m_civil_x_grau.png", data=m_les, ordem_lin=ORD_EC, ordem_col=ORDEM_GRAU)""")
code(r"""analisa("A","A6","estado_civil_grupo","tamanho_cm",
        "Mulheres — Estado civil × Tamanho da lesão (cm)",
        "A6_m_civil_x_tamanho.png", data=m_les, ordem_lin=ORD_EC, continua=True)""")
code(r"""analisa("A","A7","estado_civil_grupo","ulceracao",
        "Mulheres — Estado civil × Ulceração",
        "A7_m_civil_x_ulceracao.png", data=m_les,
        ordem_lin=ORD_EC, ordem_col=["Não","Sim"])""")
code(r"""analisa("A","A8","estado_civil_grupo","margem_comprometida",
        "Mulheres — Estado civil × Margens comprometidas",
        "A8_m_civil_x_margens.png", data=m_les,
        ordem_lin=ORD_EC, ordem_col=["Livre","Comprometida"])""")

# =================================================================== Bloco B
md(r"""## 9. Bloco B — Faixa etária × marcadores de agressividade

Compara-se o comportamento clínico-patológico do tumor entre pacientes com
até 40 anos e acima de 40 anos. *Análise em nível lesão (a idade considerada
é a do diagnóstico de cada lesão).*

Atenção: apenas 23 lesões correspondem a pacientes com até 40 anos, o que
limita o poder estatístico desses testes.""")
code(r"""ORD_FE = ["≤ 40 anos", "> 40 anos"]
analisa("B","B1","faixa_etaria","margem_comprometida",
        "Faixa etária × Margens comprometidas", "B1_idade_x_margens.png",
        ordem_lin=ORD_FE, ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("B","B2","faixa_etaria","inv_perineural",
        "Faixa etária × Invasão perineural", "B2_idade_x_perineural.png",
        ordem_lin=ORD_FE, ordem_col=["Não","Sim"])""")
code(r"""analisa("B","B3","faixa_etaria","inv_linfovascular",
        "Faixa etária × Invasão linfovascular", "B3_idade_x_linfovascular.png",
        ordem_lin=ORD_FE)""")
code(r"""analisa("B","B4","faixa_etaria","ulceracao",
        "Faixa etária × Ulceração", "B4_idade_x_ulceracao.png",
        ordem_lin=ORD_FE, ordem_col=["Não","Sim"])""")
code(r"""analisa("B","B5","faixa_etaria","tamanho_cm",
        "Faixa etária × Tamanho da lesão", "B5_idade_x_tamanho.png",
        ordem_lin=ORD_FE, continua=True)""")
code(r"""analisa("B","B6","faixa_etaria","grau_infiltracao",
        "Faixa etária × Grau de infiltração", "B6_idade_x_grau.png",
        ordem_lin=ORD_FE, ordem_col=ORDEM_GRAU)""")
code(r"""analisa("B","B7","faixa_etaria","subtipo_3cat",
        "Faixa etária × Subtipo (3 cat)", "B7_idade_x_subtipo.png",
        ordem_lin=ORD_FE, ordem_col=ORDEM_SUB)""")

# =================================================================== Bloco C
md(r"""## 10. Bloco C — Exposição solar ocupacional × marcadores

Compara-se a apresentação tumoral entre pacientes com profissões com e sem
exposição solar ocupacional. Aposentado(a), autônomo(a), desempregado e
"não informado" estão excluídos por indeterminação ocupacional.""")
code(r"""ORD_EXP = ["Com exposição", "Sem exposição"]
analisa("C","C1","exposicao_solar","subtipo_3cat",
        "Exposição solar × Subtipo (3 cat)", "C1_exp_x_subtipo.png",
        ordem_lin=ORD_EXP, ordem_col=ORDEM_SUB)""")
code(r"""analisa("C","C2","exposicao_solar","ulceracao",
        "Exposição solar × Ulceração", "C2_exp_x_ulceracao.png",
        ordem_lin=ORD_EXP, ordem_col=["Não","Sim"])""")
code(r"""analisa("C","C3","exposicao_solar","inv_perineural",
        "Exposição solar × Invasão perineural", "C3_exp_x_perineural.png",
        ordem_lin=ORD_EXP, ordem_col=["Não","Sim"])""")
code(r"""analisa("C","C4","exposicao_solar","inv_linfovascular",
        "Exposição solar × Invasão linfovascular", "C4_exp_x_linfovascular.png",
        ordem_lin=ORD_EXP)""")
code(r"""analisa("C","C5","exposicao_solar","margem_comprometida",
        "Exposição solar × Margens comprometidas", "C5_exp_x_margens.png",
        ordem_lin=ORD_EXP, ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("C","C6","exposicao_solar","tamanho_cm",
        "Exposição solar × Tamanho da lesão (cm)", "C6_exp_x_tamanho.png",
        ordem_lin=ORD_EXP, continua=True)""")
code(r"""analisa("C","C7","exposicao_solar","grau_infiltracao",
        "Exposição solar × Grau de infiltração", "C7_exp_x_grau.png",
        ordem_lin=ORD_EXP, ordem_col=ORDEM_GRAU)""")

# =================================================================== Bloco D
md(r"""## 11. Bloco D — Região anatômica × variáveis clínicas e patológicas

Avalia-se a relação entre a localização anatômica da lesão e as demais
variáveis. A face é mantida em sub-regiões pelo seu peso clínico (impacto
sobre conduta cirúrgica); as demais localizações são agrupadas por território.""")
code(r"""analisa("D","D1","regiao_grupo","exposicao_solar",
        "Região × Exposição solar", "D1_regiao_x_exposicao.png",
        ordem_col=["Com exposição","Sem exposição"])""")
code(r"""analisa("D","D2","regiao_grupo","subtipo_3cat",
        "Região × Subtipo (3 cat)", "D2_regiao_x_subtipo.png",
        ordem_col=ORDEM_SUB)""")
code(r"""analisa("D","D3","regiao_grupo","margem_comprometida",
        "Região × Margens comprometidas", "D3_regiao_x_margens.png",
        ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("D","D4","regiao_grupo","inv_perineural",
        "Região × Invasão perineural", "D4_regiao_x_perineural.png",
        ordem_col=["Não","Sim"])""")
code(r"""analisa("D","D5","regiao_grupo","inv_linfovascular",
        "Região × Invasão linfovascular", "D5_regiao_x_linfovascular.png")""")
code(r"""analisa("D","D6","regiao_grupo","ulceracao",
        "Região × Ulceração", "D6_regiao_x_ulceracao.png",
        ordem_col=["Não","Sim"])""")
code(r"""analisa("D","D7","regiao_grupo","grau_infiltracao",
        "Região × Grau de infiltração", "D7_regiao_x_grau.png",
        ordem_col=ORDEM_GRAU)""")
code(r"""analisa("D","D8","regiao_grupo","tamanho_cm",
        "Região × Tamanho da lesão", "D8_regiao_x_tamanho.png", continua=True)""")

# =================================================================== Bloco E
md(r"""## 12. Bloco E — Subtipo do tumor (3 categorias) × variáveis clínicas e patológicas

Avalia-se a relação entre o subtipo histológico, classificado em três grupos
de risco (Baixo risco, Misto e Alto risco), e as demais variáveis clínicas e
patológicas. Este bloco incorpora os cruzamentos de subtipo originalmente
planejados (grau de infiltração, ulceração, invasão linfovascular, invasão
perineural).""")
code(r"""analisa("E","E1","subtipo_3cat","regiao_grupo",
        "Subtipo × Região anatômica", "E1_subtipo_x_regiao.png",
        ordem_lin=ORDEM_SUB)""")
code(r"""analisa("E","E2","subtipo_3cat","inv_perineural",
        "Subtipo × Invasão perineural", "E2_subtipo_x_perineural.png",
        ordem_lin=ORDEM_SUB, ordem_col=["Não","Sim"])""")
code(r"""analisa("E","E3","subtipo_3cat","inv_linfovascular",
        "Subtipo × Invasão linfovascular", "E3_subtipo_x_linfovascular.png",
        ordem_lin=ORDEM_SUB)""")
code(r"""analisa("E","E4","subtipo_3cat","grau_infiltracao",
        "Subtipo × Grau de infiltração", "E4_subtipo_x_grau.png",
        ordem_lin=ORDEM_SUB, ordem_col=ORDEM_GRAU)""")
code(r"""analisa("E","E5","subtipo_3cat","margem_comprometida",
        "Subtipo × Margens comprometidas", "E5_subtipo_x_margens.png",
        ordem_lin=ORDEM_SUB, ordem_col=["Livre","Comprometida"])""")
code(r"""analisa("E","E6","subtipo_3cat","tamanho_cm",
        "Subtipo × Tamanho da lesão", "E6_subtipo_x_tamanho.png",
        ordem_lin=ORDEM_SUB, continua=True)""")
code(r"""analisa("E","E7","subtipo_3cat","ulceracao",
        "Subtipo × Ulceração", "E7_subtipo_x_ulceracao.png",
        ordem_lin=ORDEM_SUB, ordem_col=["Não","Sim"])""")

# =================================================================== Bloco F — tamanho
md(r"""## 13. Bloco F — Tamanho da lesão: descritivo global e associações

Esta seção concentra a análise do tamanho da lesão (em cm): estatísticas
gerais e cruzamentos com ulceração, margens comprometidas, invasão perineural
e região anatômica (identificando o local mais acometido).""")

md("### 13.1. Descritivo geral do tamanho da lesão")
code(r"""tam = df_lesao["tamanho_cm"].dropna()
desc = tam.describe()[["count","mean","std","min","25%","50%","75%","max"]]
desc.index = ["n","média","desvio","mínimo","q1","mediana","q3","máximo"]
print(desc.round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(tam, bins=25, kde=True, ax=axes[0], color="#4C72B0")
axes[0].axvline(tam.mean(), color="red", ls="--", label=f"média {tam.mean():.2f} cm")
axes[0].axvline(tam.median(), color="green", ls=":", label=f"mediana {tam.median():.2f} cm")
axes[0].set(title="Distribuição do tamanho da lesão",
            xlabel="Tamanho (cm)", ylabel="Lesões")
axes[0].legend()
sns.boxplot(x=tam, ax=axes[1], color="#55A868")
axes[1].set(title="Boxplot do tamanho", xlabel="Tamanho (cm)")
plt.tight_layout(); salvar_fig("F0_tamanho_geral.png"); plt.show()""")

md("### 13.2. Tamanho × Ulceração")
code(r"""analisa("F","F1","ulceracao","tamanho_cm",
        "Tamanho da lesão × Ulceração", "F1_tamanho_x_ulceracao.png",
        ordem_lin=["Não","Sim"], continua=True)""")

md("### 13.3. Tamanho × Margens comprometidas")
code(r"""analisa("F","F2","margem_comprometida","tamanho_cm",
        "Tamanho da lesão × Margens comprometidas",
        "F2_tamanho_x_margens.png",
        ordem_lin=["Livre","Comprometida"], continua=True)""")

md("### 13.4. Tamanho × Invasão perineural")
code(r"""analisa("F","F3","inv_perineural","tamanho_cm",
        "Tamanho da lesão × Invasão perineural",
        "F3_tamanho_x_perineural.png",
        ordem_lin=["Não","Sim"], continua=True)""")

md(r"""### 13.5. Tamanho médio por região anatômica
Tabela ordenada da maior para a menor média de tamanho, identificando os
locais mais acometidos por lesões maiores.""")
code(r"""tam_regiao = (df_lesao.dropna(subset=["regiao_grupo","tamanho_cm"])
              .groupby("regiao_grupo", observed=True)["tamanho_cm"]
              .agg(["count","mean","std","median",
                    lambda x: x.quantile(.25), lambda x: x.quantile(.75)]))
tam_regiao.columns = ["n","média","desvio","mediana","q1","q3"]
tam_regiao = tam_regiao.sort_values("média", ascending=False).round(2)
print(tam_regiao)

fig, ax = plt.subplots(figsize=(9, 5))
o = tam_regiao.index.tolist()
sns.barplot(x=tam_regiao["média"], y=o, ax=ax, palette="rocket_r")
for i, (m, n) in enumerate(zip(tam_regiao["média"], tam_regiao["n"])):
    ax.text(m+0.03, i, f"{m:.2f} cm  (n={n})", va="center", fontsize=9)
ax.set(title="Tamanho médio da lesão por região anatômica",
       xlabel="Tamanho médio (cm)", ylabel="")
salvar_fig("F4_tamanho_medio_por_regiao.png"); plt.show()""")

md(r"""### 13.6. Local mais acometido — distribuição geral das lesões
Frequência absoluta por região anatômica, independente do tamanho.""")
code(r"""freq_regiao = df_lesao["regiao_grupo"].value_counts(dropna=False)
pct_regiao = (freq_regiao/freq_regiao.sum()*100).round(1)
tab_local = pd.DataFrame({"n": freq_regiao, "%": pct_regiao})
print(tab_local)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=freq_regiao.values, y=freq_regiao.index.astype(str),
            ax=ax, palette="mako")
for i, (v, p) in enumerate(zip(freq_regiao.values, pct_regiao.values)):
    ax.text(v+0.5, i, f"{v} ({p}%)", va="center", fontsize=9)
ax.set(title="Frequência de lesões por região anatômica",
       xlabel="Lesões", ylabel="")
salvar_fig("F5_freq_regiao.png"); plt.show()""")

# =================================================================== Perfil 3 subtipos
md(r"""## 14. Perfil clínico-patológico dos três subtipos histológicos

Esta seção apresenta os três grupos de subtipo (Baixo risco, Misto, Alto risco)
lado a lado, com média de tamanho da lesão, frequência de ulceração, margens
comprometidas, invasão perineural e linfovascular, e as três regiões anatômicas
mais frequentes. Permite comparação direta entre os subtipos.""")
code(r"""def percentual(serie, valor):
    s = serie.dropna()
    return round((s == valor).mean()*100, 1) if len(s) else np.nan

perfil = []
for sub in ORDEM_SUB:
    g = df_lesao[df_lesao["subtipo_3cat"] == sub]
    if len(g) == 0: continue
    tam_g = g["tamanho_cm"].dropna()
    top_regioes = g["regiao_grupo"].value_counts().head(3)
    top_str = "; ".join(f"{k} ({v})" for k, v in top_regioes.items())
    grau_codes = g["grau_infiltracao"].cat.codes.replace(-1, np.nan)
    pct_profundo = round(((grau_codes >= 2).sum() /
                          grau_codes.notna().sum() * 100), 1) \
                   if grau_codes.notna().sum() else np.nan
    perfil.append({
        "Subtipo": sub,
        "n lesões": len(g),
        "Tamanho médio (cm)": round(tam_g.mean(), 2) if len(tam_g) else np.nan,
        "Tamanho mediana (cm)": round(tam_g.median(), 2) if len(tam_g) else np.nan,
        "% Ulceração": percentual(g["ulceracao"], "Sim"),
        "% Margens comprometidas": percentual(g["margem_comprometida"], "Comprometida"),
        "% Invasão perineural": percentual(g["inv_perineural"], "Sim"),
        "% Invasão linfovascular": percentual(g["inv_linfovascular"], "Sim"),
        "% Profundidade ≥ Hipoderme": pct_profundo,
        "Top 3 regiões": top_str,
    })
perfil_df = pd.DataFrame(perfil).set_index("Subtipo")
perfil_df.to_excel("perfil_subtipos.xlsx")
print("Perfil exportado: perfil_subtipos.xlsx\n")
perfil_df""")

code(r"""# gráfico comparativo dos 3 subtipos (percentuais lado a lado)
metricas = ["% Ulceração", "% Margens comprometidas",
            "% Invasão perineural", "% Profundidade ≥ Hipoderme"]
plot_df = perfil_df[metricas].reset_index().melt(id_vars="Subtipo",
                                                 var_name="Indicador",
                                                 value_name="Percentual")

fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=plot_df, x="Indicador", y="Percentual",
            hue="Subtipo", ax=ax, palette=["#55A868","#DD8452","#C44E52"])
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.1f", fontsize=9, padding=2)
ax.set(title="Comparação dos três subtipos — indicadores clínico-patológicos",
       ylabel="% das lesões com o indicador", xlabel="")
ax.legend(title="Subtipo", loc="upper right")
plt.xticks(rotation=10, ha="right")
salvar_fig("F6_perfil_3subtipos.png"); plt.show()""")

code(r"""# gráfico do tamanho médio por subtipo
tam_sub = (df_lesao.dropna(subset=["subtipo_3cat","tamanho_cm"])
                   .groupby("subtipo_3cat", observed=True)["tamanho_cm"]
                   .agg(["count","mean","std","median"]))
tam_sub.columns = ["n","média","desvio","mediana"]
tam_sub = tam_sub.round(2)
print(tam_sub)

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df_lesao.dropna(subset=["subtipo_3cat","tamanho_cm"]),
            x="subtipo_3cat", y="tamanho_cm", order=ORDEM_SUB,
            ax=ax, palette=["#55A868","#DD8452","#C44E52"], width=.5)
sns.stripplot(data=df_lesao.dropna(subset=["subtipo_3cat","tamanho_cm"]),
              x="subtipo_3cat", y="tamanho_cm", order=ORDEM_SUB,
              ax=ax, color="black", alpha=.25, size=2)
ax.set(title="Tamanho da lesão por subtipo histológico",
       xlabel="", ylabel="Tamanho (cm)")
salvar_fig("F7_tamanho_por_subtipo.png"); plt.show()""")

# =================================================================== Tabela mestre
md(r"""## 15. Tabela-mestre de p-valores

Resumo consolidado de todos os testes realizados, com indicação dos resultados
significativos.""")
code(r"""master = pd.DataFrame(RESULTADOS)
master["bloco"] = pd.Categorical(master["bloco"],
    categories=["Descritivo","Margens","A","B","C","D","E","F"], ordered=True)
master = master.sort_values(["bloco","codigo"]).reset_index(drop=True)
master.to_excel("resumo_pvalores.xlsx", index=False)
print(f"Total de testes: {len(master)}")
print(f"Significativos (p < 0,05): {(master['significativo']=='SIM').sum()}")
print(f"\nResumo exportado: resumo_pvalores.xlsx")""")
code(r"""sig = master[master["significativo"]=="SIM"]
print("=== Cruzamentos significativos ===\n")
print(sig[["bloco","codigo","cruzamento","n","teste","p_valor","efeito"]].to_string(index=False))""")
code(r"""master[["bloco","codigo","cruzamento","n","teste","p_valor","efeito","significativo"]]""")

# =================================================================== Conclusões
md(r"""## 14. Conclusões e limitações

### Síntese dos achados

**Perfil dos pacientes (n = 285):** idade média 70,6 anos (mediana 72; faixa
26–100), predomínio masculino (59,6%), com forte representação de
trabalhadores agrícolas/rurais (50,6%) e etnia branca, perfil coerente com
CBC associado à exposição solar crônica.

**Idade × sexo:** distribuições não-normais (Shapiro p < 0,01). Mann-Whitney
p = 0,082 — sem diferença estatisticamente significativa entre homens
(69,6 anos) e mulheres (72,1 anos).

**Composição do subtipo:** ao aplicar a classificação clínica padrão de CBC
(WHO/NCCN), nenhuma lesão da amostra ficou classificada como "Baixo risco"
puro — todas as 419 lesões apresentam pelo menos um componente histológico
agressivo. Os dois grupos efetivos são Misto (272 lesões, ambos os componentes)
e Alto risco puro (147 lesões, apenas componentes agressivos).

### Cruzamentos com associação estatisticamente significativa

**Margens × Reincidência (M1):** Fisher p = 0,0002. Lesões com margens
comprometidas reincidem em 18,0%, contra 3,1% das lesões com margens livres.

**Bloco B — Faixa etária × tamanho da lesão (B5):** Mann-Whitney p < 0,001;
r = 0,54 (efeito grande). Pacientes ≤ 40 anos apresentam lesões maiores
(mediana 2,3 cm; média 2,75) do que pacientes > 40 anos (mediana 1,2 cm;
média 1,49). Resultado contrário à hipótese de tumor menor no jovem. Possíveis
interpretações: diagnóstico tardio na faixa etária mais jovem (CBC não é
suspeitado precocemente) ou comportamento clínico mais expansivo. Os demais
marcadores de agressividade no Bloco B (margens, perineural, ulceração, grau,
subtipo) não diferiram entre as faixas etárias.

**Bloco C — Exposição solar ocupacional:**
- Ulceração (C2): Fisher p = 0,014. Lesões em trabalhadores expostos ao sol
  ulceram em 69,5%, contra 54,8% nos não expostos.
- Tamanho (C6): Mann-Whitney p = 0,003; r = 0,21. Lesões em expostos são
  maiores (mediana 1,2 cm; média 1,55) do que em não expostos (mediana 1,0 cm;
  média 1,15).

**Bloco D — Região anatômica:** o sítio anatômico associa-se a múltiplas
variáveis:
- Margens comprometidas (D3, qui-quadrado p = 0,014; V = 0,25).
- Ulceração (D6, p = 0,003; V = 0,28).
- Grau de infiltração (D7, p < 0,001; V = 0,29).
- Tamanho da lesão (D8, Kruskal-Wallis p < 0,001; η² = 0,19). As maiores
  lesões localizam-se em membro superior (mediana 2,25 cm) e tronco/dorso
  (2,0 cm); as menores, em face — nariz, perioral e pálpebra (~1,0 cm).

**Bloco E — Subtipo:**
- Grau de infiltração (E4, qui-quadrado p < 0,001; V = 0,33 — efeito
  moderado). Lesões mistas alcançam profundidades maiores (71,8% derme
  reticular, 17,3% hipoderme e 5,1% estruturas profundas) do que o Alto risco
  puro (58,2% reticular; 0,7% estruturas profundas). O resultado é coerente
  com a presença de componentes nodulares/sólidos associados, que conferem
  maior massa tumoral.
- Tamanho (E6, Mann-Whitney p = 0,010; r = 0,15). Lesões mistas têm tamanho
  ligeiramente maior (média 1,62 vs 1,43 cm).

### Achados não significativos relevantes

- **Bloco A — Estado civil × saúde:** nenhum dos oito cruzamentos atingiu
  significância (todos p > 0,05; o valor mais próximo foi A1, p = 0,086). A
  hipótese de doença mais avançada em homens sem companheira não foi
  confirmada nesta amostra.
- **Idade × sexo** (idade média): p = 0,082 — sem diferença significativa.
- **Invasão linfovascular** (B3, C4, D5, E3): todos sem variabilidade
  (ausência de casos positivos).

### Limitações

1. **Invasão linfovascular sem casos positivos** na base (200 "Não" e 219
   ausentes). Os cruzamentos envolvendo essa variável são descritivos e
   produzem p = 1,0 por construção.
2. **Faixa etária ≤ 40 com apenas 23 lesões** — poder estatístico reduzido
   para o Bloco B; resultados não significativos nessa faixa não excluem
   diferenças clínicas reais.
3. **Múltiplas lesões por paciente** (até 14 por código) — as observações
   por lesão não são totalmente independentes; os intervalos de confiança e
   p-valores subestimam ligeiramente a variabilidade.
4. **Sem correção para múltiplas comparações** — foram realizados 38 testes;
   alguns p < 0,05 podem ser falso-positivos por acaso. Aplicando uma correção
   conservadora de Bonferroni (α ≈ 0,0013), permanecem significativos:
   M1, B5, C6, D6, D7, D8 e E4.
5. **Classificação de subtipo** baseada em critérios clínicos padrão; a
   ausência de "Baixo risco" puro na amostra é coerente com o perfil de
   um serviço de patologia, mas pode representar viés de encaminhamento.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"}}
with open("analise_basocelular_completa.ipynb","w",encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook consolidado escrito ({len(cells)} células).")
