# Estudo: Jogo Cooperativo de Rank-Budget para PCA e sua conexão com EigenGame

> Log de leitura e entendimento dos dois artigos, com foco na sua contribuição
> (`SVD_cooperative_game.pdf`) e na ligação com a leitura de PCA como equilíbrio de
> Nash (EigenGame). Fecha com o desenho de um experimento "caso simples" com
> entradas gaussianas.
>
> Data: 2026-06-04

---

## 1. Visão geral: dois jogos, dois eixos do mesmo objeto espectral

Os dois trabalhos olham para o **mesmo objeto** — a decomposição espectral / PCA de
uma matriz de covariância `C = AᵀA ⪰ 0` — mas o tratam como **jogos de naturezas
opostas**:

| | EigenGame (Gemp et al.) | Sua contribuição (rank-budget) |
|---|---|---|
| Tipo de jogo | **Não-cooperativo** | **Cooperativo (TU)** |
| Jogadores | as `k` **direções** `v_i ∈ ℝᵖ` | as `p` **variáveis** observadas |
| Pergunta | *como* computar a PCA truncada | quão **instável** é a distribuição de variância |
| Solução | Equilíbrio de Nash | Core / Least-core |
| Existência | Nash **existe e é único** (sob gap espectral) | Core **vazio** sempre que `rank(C) > k` |

A frase-chave do seu próprio artigo (Seção 2, Related work) resume a dualidade:

> "EigenGame describes how the truncated PCA can be **computed** as competition among
> directions, and the rank-budget game describes how **unstable** the resulting
> variance distribution is when read as cooperation among variables."

E, mais forte ainda, a Seção 9.3 mostra que eles são **algoritmicamente componíveis**:
*"the equilibrium of one is the value oracle of the other"*.

---

## 2. O artigo original: PCA como equilíbrio de Nash (EigenGame)

(Os PDFs disponíveis são a linhagem EigenGame — o `910_..._generalized_eigengame`
é a **extensão a GEPs** [GHA-GEP / δ-EigenGame, ICLR 2023]; o artigo "original"
citado por você como [4] é *EigenGame: PCA as a Nash equilibrium*, Gemp et al. 2020.)

### 2.1 A ideia central

PCA é normalmente um problema de otimização **com restrições** (ortonormalidade).
EigenGame reescreve a busca dos top-`k` autovetores como um **jogo de `k` jogadores**,
onde cada jogador `i` controla uma direção `v_i ∈ ℝᵖ` na esfera unitária e maximiza
uma utilidade própria:

- **Recompensa**: alinhamento com a covariância, `v_iᵀ C v_i` (capturar variância).
- **Penalidade**: alinhamento com as direções `v_j`, `j < i`, já "reivindicadas"
  pelos jogadores anteriores (hierarquia / deflação).

Esquematicamente, a utilidade do jogador `i` tem a forma

```
u_i(v_i) =  v_iᵀ C v_i  −  Σ_{j<i} (penalização por alinhamento com v_j) .
```

O **equilíbrio de Nash** desse jogo é exatamente a base dos top-`k` autovetores de
`C` (a SVD truncada). A versão generalizada (PDF 910) troca a restrição de esfera por
um multiplicador de Lagrange (penalidade de variância `Γ_ii`), obtendo updates do tipo
Hebbiano:

```
Δ_i = A v_i  −  Σ_{j≤i} B v_j (v_jᵀ A v_i)        (reward − variance penalty − orthogonality penalty)
```

Para `B = I` (o caso PCA), isso colapsa exatamente no Generalized Hebbian Algorithm.

### 2.2 Por que isso importa para a sua contribuição

Três pontos da maquinaria do EigenGame que o seu artigo reaproveita:

1. **Princípio variacional de Ky Fan**. A soma dos `k` maiores autovalores é
   `‖M‖_(k) = max_{QᵀQ=I_k} tr(QᵀMQ)`. EigenGame *resolve* esse `max`; o seu `v_k(S)`
   *usa* esse mesmo `max` (restrito ao subespaço de coordenadas de `S`) como o **valor
   da coalizão**. Ou seja, o valor de cada coalizão no seu jogo cooperativo **é** o
   valor de Nash de um EigenGame jogado na submatriz `C_SS`.

2. **Existência vs. instabilidade**. Sob gap espectral, o Nash do EigenGame existe e é
   essencialmente único. O seu Teorema 4 mostra o oposto no lado cooperativo: o core é
   **vazio** sempre que `rank(C) > k`. Não é contradição — são perguntas duais. O
   EigenGame garante que a *computação* converge; o seu jogo mede que a *alocação de
   crédito de variância entre variáveis* é estruturalmente insatisfatível.

3. **Oráculo diferenciável (Seção 9.3)**. O passo difícil do least-core é o oráculo de
   separação `max_S v_k(S) − x̄(S)`. Você propõe relaxar o indicador binário `1_S` por
   uma máscara contínua `m ∈ [0,1]ᵖ`, definir `C(m) = diag(m) C diag(m)`, e observar que

   ```
   ‖C(m)‖_(k) = max_{VᵀV=I_k} tr(Vᵀ C(m) V)
   ```

   é **exatamente** o valor de Nash de um EigenGame jogado em `C(m)`. Assim a busca
   combinatória sobre coalizões vira um problema bi-nível contínuo, cujo solver interno
   são `k` ascensões de gradiente paralelas (EigenGame). É aqui que "o equilíbrio de um
   é o oráculo de valor do outro".

---

## 3. A sua contribuição em detalhe: o jogo de rank-budget

### 3.1 Definição

Dada `C ⪰ 0` (`p × p`) e um orçamento de posto `k ∈ {1,…,p}`, a cada coalizão de
variáveis `S ⊆ N = {1,…,p}` associa-se o valor

```
v_k(S) = ‖C_SS‖_(k) = Σ_{r=1}^{min(k,|S|)} λ_r(C_SS) ,      v_k(∅) = 0.
```

Interpretação direta (Ky Fan): `v_k(S)` é a **variância máxima explicável por uma
projeção de posto `k`** quando só se observam as variáveis em `S` (i.e., PCA de posto
`k` aplicado a `A_{:,S}`). O `k` é lido como um **orçamento de dimensão** compartilhado.

### 3.2 Propriedades básicas (Prop. 2)

- Normalizado (`v_k(∅)=0`) e **monótono** (via interlacing de Cauchy).
- `0 ≤ v_k(S) ≤ tr(C_SS) = Σ_{i∈S} C_ii`.
- `v_p(S) = tr(C_SS)` (com `k=p` somam-se todos os autovalores → função modular).
- **Não é submodular nem supermodular em geral** (Prop. 3): contraexemplos com 2 e 3
  variáveis. Logo PCA truncado **não** é um problema submodular padrão — não há
  polítopo-base óbvio que estabilize as reivindicações.

### 3.3 O resultado central: dicotomia do core (Teorema 4)

```
Core(v_k) ≠ ∅   ⟺   rank(C) ≤ k.
```

E quando não-vazio, o core é o **singleton** `x_i = C_ii`.

**A intuição (escassez de posto).** Cada variável `i` sozinha reivindica `v_k({i}) =
C_ii`. Somando, qualquer alocação no core precisaria de `x(N) ≥ Σ_i C_ii = tr(C)`. Mas
a eficiência exige `x(N) = v_k(N) = Σ_{r≤k} λ_r(C)`. Se `λ_{k+1}(C) > 0` (i.e.,
`rank(C) > k`), então

```
Σ_{r≤k} λ_r(C)  <  tr(C),
```

logo é **impossível** satisfazer todas as reivindicações simultaneamente. **A
truncagem espectral cria escassez**: as variáveis, juntas, reivindicam mais variância
(`tr C`) do que o orçamento de posto consegue distribuir (`Σ_{r≤k} λ_r`).

→ Para **qualquer** `C` de posto cheio e **qualquer** `k < p`, o core é vazio. A
instabilidade coalizional não é anomalia; é consequência estrutural da truncagem.

(Corolário 5: `v_k` é modular ⟺ `rank(C) ≤ k`. O conteúdo coalizional de `v_k`
desaparece exatamente quando o core fica não-vazio.)

### 3.4 Least-core: transformando o vazio em uma medida (LP 3)

Como o core é quase sempre vazio, usa-se o **least-core**: a menor folga uniforme `η`
que torna possível uma alocação eficiente.

```
η*_k = min_{x, η≥0}  η
       s.a.  x(N) = v_k(N),
             x(S) + η ≥ v_k(S)   ∀ S ∈ F  (coalizões próprias não-vazias).
```

E a versão **relativa** (invariante a escala de `C`):

```
γ_k = η*_k / v_k(N) .
```

### 3.5 Dual e certificados balanceados (Teorema 7)

O dual do LP é

```
η*_k = max_{y,α}  Σ_S y_S [ v_k(S) − (|S|/p) v_k(N) ]
       s.a.  y_S ≥ 0,  Σ_S y_S ≤ 1,  Σ_{S∋i} y_S = α  ∀i.
```

`y` é uma **coleção balanceada** de coalizões (cada jogador aparece com a mesma
intensidade total `α`). O objetivo mede o **excesso médio** entre a variância truncada
reivindicada pelas coalizões e a fatia proporcional do valor da grande coalizão. Um
`y` ótimo é um **certificado de conflito espectral**: identifica *quais* grupos de
variáveis competem pelas mesmas direções.

### 3.6 Cotas inferiores

- **Cota de singleton (Cor. 6)** — puramente espectral:
  ```
  η*_k ≥ (1/p) Σ_{r>k} λ_r(C),    logo   γ_k ≥ (Σ_{r>k} λ_r) / (p Σ_{r≤k} λ_r).
  ```
  Qualquer excesso de `η*_k` acima disso é instabilidade **estritamente coalizional**,
  invisível no espectro global.
- **Cotas de cardinalidade fixa (Cor. 8)**: média de `v_k` sobre coalizões de tamanho
  `q`. O caso `q=k` recupera `η*_k` exatamente no espectro plano.

### 3.7 Caso resolvido: espectro plano `C = λ I_p` (Teorema 9)

Isola a escassez de posto, sem correlação nem anisotropia:

```
v_k(S) = λ · min(k, |S|),
η*_k   = λ k (1 − k/p),
γ_k    = 1 − k/p,         (alocação ótima x_i = λk/p)
```

Observações importantes:
- `γ_k = 1 − k/p` **decresce linearmente** até 0 em `k=p`.
- O **absoluto** `η*_k` pode *crescer* com `k` (até ~`p/2`) porque a grande coalizão
  tem mais valor a distribuir; o **relativo** `γ_k` decresce. → Normalizar importa.
- O máximo da restrição ocorre em coalizões de tamanho `s = k`.

### 3.8 Shapley vs. least-core: duas noções de importância (Seção 8)

- **Shapley** `φ_i^(k)` = importância marginal média (sempre eficiente,
  `Σ_i φ_i = v_k(N)`). Mede *quanto uma variável adiciona* em ordens aleatórias.
- **Least-core** = estabilidade coalizional. Mede *quanta violação é inevitável*.

Eficiência ≠ estabilidade: a alocação de Shapley também viola coalizões quando
`rank(C) > k`. As duas medidas podem divergir (uma variável com alto Shapley pode
pertencer a coalizões muito insatisfeitas, e vice-versa). Empiricamente, o artigo
também mostra que **Shapley ≠ leverage scores**.

### 3.9 Controle submodular: o jogo log-determinante (Seção 10)

```
g_τ(S) = log det(I_{|S|} + τ⁻¹ C_SS) = Σ_r log(1 + μ_r(S)/τ).
```

Preserva **todos** os modos espectrais com peso logarítmico. Pela desigualdade de
Hadamard–Fischer (Koteljanskii) é **submodular** (Prop. 10) → regime
"polimatroidal" estável, com garantias guloso. Contraste:

- `g_τ` = regime **regularizado** (todos os modos, submodular, core bem-comportado);
- `v_k` = regime **singular** (só `k` modos, submodularidade falha, core vazio).

`τ` interpola: `τ→∞` ⇒ `g_τ ∝ tr(C_SS)` (modular, = caso `k=p`); `τ→0⁺` ⇒
`log det(C_SS)` (singular quando `|S| > rank C`).

### 3.10 Diagnóstico proposto: o "scree plot cooperativo"

A curva `k ↦ γ_k` é proposta como **assinatura espectral cooperativa**, complementar
ao scree plot clássico:

- `γ_k = 0 ⟺ k ≥ rank(C)`;
- `γ_k` alto ⇒ o posto `k` gera forte conflito entre coalizões;
- quedas de `γ_k` ⇒ postos onde o orçamento passa a acomodar as reivindicações
  dominantes;
- as coalizões ativas no dual identificam os grupos que **certificam** o conflito.

Achados empíricos (`p=8`, enumeração exata de 254 coalizões, 5 cenários: flat, spike,
blocks, duplicates, lowrank):
1. `γ_k` **não é monótona** em `k` (ex.: spike sobe de 0.109 em `k=1` até 0.188 em
   `k=3`) — fenômeno **estritamente coalizional** (a cauda espectral clássica é sempre
   monótona).
2. `γ_k` **identifica o posto efetivo** (duplicates cai 2 ordens de grandeza entre
   `k=4` e `k=5`, no posto efetivo ≈5).
3. A ordenação dos cenários **não** se alinha com a cauda espectral → `γ_k` carrega
   informação não-redutível aos autovalores globais.

---

## 4. A conexão, em uma frase

> **EigenGame e o jogo de rank-budget são duais.** EigenGame é a face *computacional*
> (não-cooperativa, jogadores = direções, Nash existe): resolve o `max` de Ky Fan que
> *produz* `v_k(S)`. O jogo de rank-budget é a face *distributiva* (cooperativa,
> jogadores = variáveis, core vazio): pergunta se a variância que o EigenGame computou
> pode ser repartida de forma estável entre as variáveis — e mostra que, sob escassez
> de posto, **não pode**. O least-core `η*_k`/`γ_k` quantifica exatamente o tamanho
> dessa impossibilidade. E os dois se encaixam: o valor de Nash do EigenGame em `C(m)`
> é o oráculo de valor (diferenciável) do least-core.

---

## 5. Desenho do experimento: caso simples gaussiano

Objetivo declarado:
> 1. Gerar vetores com entradas gaussianas com parâmetros definidos (var e média)
> 2. Colocá-los em `A`
> 3. Olhar o comportamento do `η` no least-core quando variamos esses parâmetros.

### 5.1 Setup

- `A ∈ ℝ^{m×p}` com entradas i.i.d. `A_{ni} ~ N(μ, σ²)`.
- `C = AᵀA` (`p × p`), `p` pequeno (sugiro **`p = 8`**) para enumerar as `2ᵖ − 2 = 254`
  coalizões e resolver o LP (3) **exato**, igual à Tabela 1 do artigo.
- Para cada `k ∈ {1,…,p}`: resolver o LP → `η*_k` e `γ_k = η*_k/v_k(N)`.
- **Monte Carlo**: repetir com muitas sementes (ex.: 100–500) e reportar média ± IC,
  pois `C` é aleatória (Wishart).

### 5.2 O que a teoria já prevê (hipóteses a testar)

A chave é entender o que `μ` e `σ` fazem com o **espectro** de `C = AᵀA` **sem
centralizar** as colunas:

```
E[C_ij] = m (μ² + σ² δ_ij)   ⇒   E[C] = m σ² I_p  +  m μ² 11ᵀ.
```

Ou seja, `E[C]` é **espectro plano + spike de posto 1** na direção `1`:
- autovalor da spike ≈ `m(σ² + p μ²)` na direção `1/√p`;
- bulk ≈ `m σ²` nas outras `p−1` direções (com flutuações de Marchenko–Pastur de
  largura `~σ²√(p/m)`).

Logo o **único knob que importa para `γ_k`** (que é invariante a escala) é a razão

```
ρ = μ² / σ²      (ou o SNR da spike:  SNR = p μ² / σ²),
```

e secundariamente a razão de aspecto `m/p`. Previsões:

| Regime | Espectro | Previsão para `γ_k` |
|---|---|---|
| **`μ = 0`** (só ruído) | Wishart ≈ plano (`C/m → σ² I` quando `m→∞`) | `γ_k ≈ 1 − k/p` (cenário **flat**); converge ao Teorema 9 quando `m` cresce |
| **`μ` intermediário** | spike emergindo do bulk | aparece a **não-monotonicidade** de `γ_k` (pico em `k` pequeno), como no cenário **spike** |
| **`μ ≫ σ`** | uma direção dominante forte | `γ_1` **despenca** (a spike é capturada já em `k=1`; reivindicações ficam quase satisfazíveis), curva tipo **spike** da Tabela 1 (`γ_1 ≈ 0.11 ≪ 0.875`) |

Em outras palavras: **variar `μ/σ` interpola continuamente entre os cenários "flat" e
"spike" da Tabela 1**, usando um único parâmetro. Esse é o gancho bonito do
experimento.

Pontos finos a checar/explicar:
1. **Centralizar ou não.** Se você centralizar as colunas de `A`, a média some e você
   fica *sempre* no regime Wishart/plano — o efeito de `μ` desaparece. O experimento só
   é interessante **sem centralizar** (deixando a média gerar a spike). Vale rodar os
   dois para evidenciar isso.
2. **Invariância de escala.** `η*_k` escala linearmente com `m` e `σ²`; `γ_k` **não**.
   Confirme numericamente que variar `σ` com `μ/σ` fixo deixa `γ_k` invariante (bom
   *sanity check* do código e da Prop. de invariância).
3. **Cota de singleton (Cor. 6).** Plote `γ_k` vs. a cota `(Σ_{r>k}λ_r)/(p Σ_{r≤k}λ_r)`.
   Previsão: justa em `μ=0`, `k=1` (simetria); folga crescente conforme a spike aparece
   (excesso coalizional).
4. **`η*_k` absoluto vs. `γ_k` relativo.** Mostre que `η*_k` pode ser não-monótono/crescer
   com `m` enquanto `γ_k` se estabiliza — reforça por que normalizar.

### 5.3 Variáveis a varrer (grid sugerido)

- `μ ∈ {0, 0.25, 0.5, 1, 2, 4}` com `σ = 1` fixo (só a razão importa).
- `m ∈ {16, 64, 256}` (razão de aspecto: undersampled → bem amostrado).
- `k ∈ {1,…,8}`.
- (controle) `σ ∈ {0.5, 1, 2}` com `μ/σ` fixo → confirmar invariância de `γ_k`.
- 100+ sementes por célula.

### 5.4 Saídas / figuras

1. **`γ_k` vs `k`**, uma curva por `μ` (família de curvas interpolando flat→spike).
2. **`γ_1` (e `η*_1`) vs `μ/σ`** — a "curva de colapso" da instabilidade do singleton.
3. **Heatmap `γ_k` sobre `(k, μ)`** — visão global da transição.
4. **`γ_k` vs cota de singleton** (excesso coalizional) por nível de `μ`.
5. **Convergência ao Teorema 9**: erro `|γ_k − (1−k/p)|` vs `m` no caso `μ=0`.
6. (opcional) Coalizões ativas no dual em função de `μ` — checar se elas se concentram
   na direção `1` conforme a spike cresce.

### 5.5 Pergunta científica que o experimento responde

> *Como a estrutura espectral induzida pela média (`μ`) — uma única direção dominante —
> reduz a instabilidade coalizional medida pelo least-core, e em que ponto (`μ/σ`,
> `m/p`) a assinatura cooperativa `γ_k` deixa de parecer "plana" e passa a "spike"?*

Isso conecta diretamente o seu diagnóstico `γ_k` a um modelo gerador controlado
(spike + ruído), que é o mesmo modelo onde o EigenGame tem garantia de convergência
(gap espectral = a spike). Ou seja: **no regime em que o EigenGame é mais estável
(spike forte), o seu jogo cooperativo é menos instável** — uma ponte quantitativa
entre as duas leituras.

---

## 6. Próximos passos sugeridos

- [ ] Implementar o solver do least-core por enumeração (LP com `scipy.optimize.linprog`
      ou `cvxpy`) para `p=8`; validar contra a Tabela 1 (cenário flat ⇒ `1−k/p`).
- [ ] Rodar o grid `(μ, m)` da Seção 5.3 e gerar as figuras 1–5.
- [ ] Verificar as 3 identidades do artigo como teste de sanidade (eficiência de
      Shapley, normalização de leverage, cota de Cor. 6).
- [ ] (Avançado) Trocar a enumeração pelo oráculo diferenciável via EigenGame
      restrito (Seção 9.3) e validar que reproduz `η*_k` para `p` maior.
