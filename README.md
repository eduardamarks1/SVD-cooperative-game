# Least-core sob escassez de posto — experimento gaussiano

Experimento numérico para o artigo *Rank-Budget Cooperative Games for PCA: Least-Core,
Spectral Scarcity, and Stability Diagnostics*. Tudo está no notebook
[`experimento_least_core.ipynb`](experimento_least_core.ipynb).

Este README é escrito para quem **não conhece teoria dos jogos cooperativos** — a ideia
é construir a intuição do zero, com uma analogia única (uma banda de música) que volta
em todo lugar.

---

## 0. TL;DR

Geramos uma matriz de dados `A` (m×p) com entradas gaussianas `N(μ, σ²)`, formamos
`C = AᵀA`, e medimos uma quantidade chamada **least-core** `η*_k` (e sua versão
relativa `γ_k`). Variamos a **média `μ`** e observamos a instabilidade `γ_k` ir de
"máxima e simétrica" (quando `μ=0`) para "quase nula" (quando `μ` é grande). O parâmetro
`μ/σ` funciona como um **botão** que transforma o problema de um regime no outro.

---

## 1. Teoria dos jogos cooperativos em 5 minutos (com a analogia da banda 🎸)

### 1.1 O que é um "jogo cooperativo"

Esqueça "jogo" no sentido de competição. Em **teoria dos jogos cooperativos**, um jogo é
só uma receita que diz, para *cada grupo possível de participantes*, **quanto valor aquele
grupo consegue produzir junto**. O problema central é: **como dividir o valor total de
forma justa/estável entre os participantes?**

Exemplos do dia a dia:
- dividir a conta do aluguel entre colegas de quarto;
- repartir o prêmio de um projeto entre os membros da equipe;
- dividir o lucro de uma cooperativa.

> **🎸 A analogia da banda.** Temos `p` **músicos**. Mas o estúdio só tem `k`
> **microfones** (esse `k` é o *orçamento de posto*, o número de componentes da PCA).
> Cada som ("direção") tem um **volume/talento** (= a variância, ou autovalor).
> - Um **subgrupo `S`** de músicos, se gravasse sozinho com seus `k` microfones,
>   conseguiria uma gravação de qualidade total **`v_k(S)`** (= soma dos `k` maiores
>   volumes daquele subgrupo).
> - A **banda inteira** gera fama/dinheiro **`v_k(N)`**.
> - O problema: **como dividir essa fama** entre os `p` músicos (uma alocação `x`, onde
>   `xᵢ` é a parte do músico `i`)?

### 1.2 O "core" — uma divisão sem revolta

Uma divisão `x` é **estável** (está no *core*) se **nenhum subgrupo tem motivo para se
rebelar**. Formalmente: para todo subgrupo `S`, a soma do que seus membros recebem
(`x(S) = Σ_{i∈S} xᵢ`) deve ser **pelo menos** o que esse grupo conseguiria sozinho
(`v_k(S)`):

```
x(S) ≥ v_k(S)   para todo grupo S        (ninguém ganha saindo)
x(N) = v_k(N)                            (divide-se exatamente o total)
```

> 🎸 **Core = uma divisão da fama em que nenhum subgrupo de músicos pensa "a gente
> ganharia mais se largasse a banda e gravasse só entre nós".** Se existe uma divisão
> assim, a banda é estável.

### 1.3 O resultado-chave do artigo: o core é (quase sempre) **vazio**

Aqui está a sacada. Cada músico sozinho (grupo de 1) reivindica `v_k({i}) = Cᵢᵢ` (seu
volume individual). Somando todo mundo, qualquer divisão estável precisaria distribuir
pelo menos `tr(C) = Σᵢ Cᵢᵢ`. **Mas** a banda inteira só fatura `v_k(N) = λ₁+…+λ_k`
(soma dos `k` maiores volumes). Se há mais "vozes independentes" do que microfones
(`rank(C) > k`), então

```
v_k(N) = λ₁ + … + λ_k   <   λ₁ + … + λ_p = tr(C)
```

ou seja, **a banda fatura menos do que a soma das reivindicações individuais**. Não dá
para contentar todo mundo. **O core é vazio** (Teorema 4 do artigo).

> 🎸 **Por que vazio?** Porque há **menos microfones que sons distintos**. Sempre vai
> existir um subgrupo que, com os microfones só para si, grava algo melhor do que a
> fatia que a banda lhe oferece. É escassez: o orçamento de posto `k` não comporta todas
> as reivindicações ao mesmo tempo.

Isso não é um defeito — é **a mensagem do artigo**: truncar a PCA (usar `k < p`
componentes) cria instabilidade estrutural, não por acaso.

### 1.4 O "least-core" `η` — medindo *quão* instável

Se o core está vazio (ninguém fica feliz), a pergunta deixa de ser "existe divisão
estável? (sim/não)" e vira **"quão perto chegamos?"**. Damos a *todos* os grupos um
**bônus uniforme `η`** (um "agrado") e perguntamos: qual é o **menor** `η` que faz
todos pararem de reclamar?

```
η*_k = menor η ≥ 0  tal que  x(S) + η ≥ v_k(S)  para todo grupo S
```

> 🎸 **`η` = o menor suborno/bônus igual para todos os subgrupos que impede a banda de se
> dissolver.** `η` **pequeno** → quase estável, banda harmoniosa. `η` **grande** → muito
> conflito, todo mundo quase saindo.

Como `η` depende da escala (uma banda "mais alta" tem números maiores sem ser mais
instável), usamos a versão **relativa**, que é o que realmente comparamos:

```
γ_k = η*_k / v_k(N)        (instabilidade relativa, entre 0 e ~1)
```

`γ_k = 0` significa banda perfeitamente estável; `γ_k` alto significa muito conflito.

A curva `k ↦ γ_k` é o **"scree plot cooperativo"** proposto no artigo: um diagnóstico de
quantos componentes (`k`) usar, que enxerga conflitos entre grupos de variáveis — coisa
que o scree plot clássico (que só olha autovalores) não vê.

---

## 2. O experimento

### 2.1 De onde vêm os dados

Geramos `A` (m linhas = amostras, p colunas = variáveis/músicos) com **toda entrada
sorteada de uma gaussiana `N(μ, σ²)`** — mesma média `μ`, mesma variância `σ²`. Depois
`C = AᵀA` (a matriz de covariância não-normalizada do artigo).

**Não centralizamos** as colunas de propósito: é a média `μ` que queremos estudar.

### 2.2 A conta que governa tudo

A média de `C` é (ver dedução no notebook e no `estudo.md`):

```
E[C] = m·σ²·I_p   +   m·μ²·(11ᵀ)
        └─ platô ─┘     └─ agulha na direção "todos iguais" (1,1,...,1) ─┘
```

Os autovalores disso são:
- **um valor grande** `m(σ² + p·μ²)` na direção `1` (a **spike**);
- **`p−1` valores iguais** `m·σ²` (o **platô / "bulk"**).

### 2.3 Os termos, na analogia 🎸

| Termo | O que é | Na banda |
|---|---|---|
| **flat** (plano) | todos os autovalores iguais (`C=λI`) | todos os músicos igualmente bons, cada um tocando algo independente → **briga máxima** pelos microfones |
| **spike** (agulha) | um autovalor domina os outros | existe um **uníssono**: uma melodia que todos tocam junto → **um líder claro** soa muito mais alto |
| **`μ` (média)** | quanto as entradas "puxam" na mesma direção | **o quanto a banda toca em uníssono.** `μ=0`: cada um na sua (flat). `μ` grande: todos na mesma melodia (spike) |
| **knob / botão** | parâmetro de controle | a razão **`μ/σ`** — gira o sistema de flat (`μ=0`) para spike (`μ` grande) |

**Por que `μ` cria a spike:** se toda entrada tem média `μ>0`, cada linha de `A` é
aproximadamente `μ·(1,1,…,1) + ruído` — todas as amostras apontam um pouco na mesma
direção comum. Essa "melodia compartilhada" vira o eixo dominante de variância.

### 2.4 O que esperamos ver

| Regime | Espectro | Previsão para `γ_k` |
|---|---|---|
| `μ = 0` | plano (Wishart ≈ `σ²I`) | `γ_k ≈ 1 − k/p` (**Teorema 9**) — instabilidade máxima e simétrica |
| `μ` médio | spike emergindo | `γ_k` fica **não-monótono** (corcova em `k` pequeno) |
| `μ ≫ σ` | spike forte | `γ_1` **despenca** (um microfone já grava o líder) |

Resumo: **`μ/σ` interpola continuamente entre os cenários "flat" e "spike" da Tabela 1
do artigo**, com um único botão.

---

## 3. O que o notebook faz (célula a célula)

1. **Setup + funções base** — `vk(...)` (valor de uma coalizão) e `least_core_curve(...)`
   (resolve o LP do least-core por enumeração exata das 254 coalizões, para todo `k`).
2. **Teste de sanidade (Teorema 9)** — valida o solver no caso plano: `γ_k` bate com
   `1 − k/p` a ~`1e-16`. ✅
3. **Gerador de dados** — `gen_C(...)` e o Monte Carlo `mc_gamma(...)`.
4. **Experimento 1** — `γ_k` vs `k` para vários `μ` (a família de curvas flat→spike).
5. **Experimento 2** — colapso de `γ_1` e `η*_1` ao girar `μ/σ` (relativo vs absoluto).
6. **Experimento 3** — mapa de calor `γ_k` sobre `(k, μ/σ)`.
7. **Experimento 4** — convergência ao Teorema 9 conforme `m` cresce (`μ=0`).
8. **Experimento 5** — `γ_k` vs cota de singleton (o **excesso coalizional**, a parte da
   instabilidade invisível ao espectro global).
9. **Controle** — invariância de escala de `γ` (muda `σ`, mantém `μ/σ` → `γ` não muda).
10. **Experimento A — *quem* briga (coalizões ativas no dual).** Ver §3.1 abaixo.
11. **Experimento B — a banda saudável (jogo log-determinante).** Ver §3.2 abaixo.
12. **Conclusões.**

### 3.1. Experimento A — quais grupos certificam o conflito

`γ_k` diz *quanta* instabilidade existe; o **dual** do least-core (Teorema 7) diz **quem** a
causa. Ao resolver o LP, cada coalizão `S` ganha um peso `y_S ≥ 0`; as de peso positivo são as
**ativas** — os grupos que certificam o conflito.

> 🎸 Em vez de *"a banda está 30% instável"*, apontamos o dedo para os subgrupos que estão se
> estranhando. Como o modelo é simétrico sob permutar variáveis, olhamos o **tamanho** dos grupos
> ativos (a quantidade interpretável e estável no Monte Carlo), não variáveis individuais.

O notebook plota um mapa de calor (tamanho da coalizão × `μ/σ`). **Resultado:** em `μ=0` o peso
gruda no tamanho `|S| = k` — confirma a prova do Teorema 9 (são as coalizões de tamanho `k` que
travam a banda); conforme a *spike* cresce, o peso **migra** para outros tamanhos. O least-core não
só mede, ele **localiza** o conflito.

### 3.2. Experimento B — o jogo log-determinante como linha de base estável

O artigo usa um segundo jogo de **controle**, `g_τ(S) = log det(I + τ⁻¹ C_SS)`, que usa *todos* os
modos espectrais (não só os `k` maiores) e é **submodular** — a propriedade "boa" que `v_k` não tem.

> 🎸 É a **banda saudável de referência**: todos os microfones, sem escassez artificial. O botão
> `τ` controla: `τ→∞` deixa `g_τ` quase modular (banda perfeitamente estável, `γ≈0`); `τ→0` o
> empurra para o regime singular.

O notebook faz duas coisas:
- **B1 (submodularidade na prática):** conta violações da desigualdade `v(S)+v(T) ≥ v(S∪T)+v(S∩T)`.
  Resultado validado: `v_1` viola **1729** de 5000 vezes (não é submodular — Prop. 3); `g_τ` viola
  **0** (é submodular — Prop. 10).
- **B2 (estabilidade comparada):** `γ` de `v_k` (singular, instável) vs `γ` de `g_τ` (regularizado,
  calmo) ao variar `μ`. A curva de `v_k` fica bem acima; `g_τ` é a banda calma, mais ainda com `τ`
  grande. É a tese central do artigo (regime singular vs. regularizado) visível nos dados.

### Resultados já validados (notebook executado)

- Teorema 9: erro máximo `3.3e-16` (solver correto).
- Convergência `μ=0`: erro médio cai `0.298 → 0.087` conforme `m` vai de 16 a 512.
- Invariância de escala: `max|Δγ_k| = 0.0000` ao variar `σ` com `μ/σ` fixo.
- Submodularidade (Exp. B1): `v_1` → 1729 violações, `g_τ` → 0 violações.
- 7 figuras geradas, 0 erros de execução.

---

## 4. Como rodar

Requisitos: `numpy`, `scipy`, `matplotlib`, e (para abrir o notebook) `jupyter`.

```bash
# abrir interativamente
jupyter notebook experimento_least_core.ipynb

# ou re-executar tudo do zero pela linha de comando
jupyter nbconvert --to notebook --execute --inplace experimento_least_core.ipynb
```

**Parâmetros que você pode mexer** (no topo das células):
- `N_SEEDS` — nº de sorteios no Monte Carlo (↑ = curvas mais lisas, mais lento; padrão 60);
- `P` — nº de variáveis (mantenha ≤ ~12: a enumeração é `2ᵖ−2` coalizões);
- `M_MAIN` — nº de amostras (linhas de `A`);
- `MUS`, `MU_GRID` — grades de `μ` varridas.

> O script [`_build_nb.py`](_build_nb.py) gera o `.ipynb` a partir de código Python puro
> (útil para versionar e regerar o notebook). Não é necessário para rodar o experimento.

---

## 5. Conexão com o EigenGame (a foto grande)

O artigo lê a PCA de **dois** jeitos de teoria dos jogos, **duais** entre si:

- **EigenGame** (Gemp et al., *PCA as a Nash equilibrium*): jogo **não-cooperativo**, os
  jogadores são as `k` **direções**; serve para *computar* a PCA. O equilíbrio **existe e
  é único** sob "gap espectral" (= existir uma spike).
- **Jogo de rank-budget** (este artigo): jogo **cooperativo**, os jogadores são as `p`
  **variáveis**; mede quão **instável** é repartir a variância. O core é **vazio** sob
  truncagem.

O experimento conecta os dois: **no regime em que o EigenGame é mais estável (spike forte
= gap grande), o jogo cooperativo é menos instável (`γ` menor)**. Giramos `μ/σ` e vemos
as duas leituras concordarem. (E o artigo mostra, na Seção 9.3, que o valor de Nash do
EigenGame é literalmente o "oráculo de valor" do least-core — eles se encaixam.)

Veja [`../estudo.md`](../estudo.md) para o estudo teórico completo dos dois artigos.
