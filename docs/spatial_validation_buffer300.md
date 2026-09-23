# Etapa 4 — sensibilidade metodológica com buffer de 300 m

## Papel desta análise

O resultado oficial continua sendo a validação espacial com blocos de 1.500 m,
5 folds, seed 42 e nenhum buffer, em `outputs/models/spatial_validation/`.
Esta análise adicional usa um buffer **pré-definido de 300 m**, sem otimização
de raio, seed, blocos, limiar ou hiperparâmetros. Não substitui o resultado
principal, não altera o dataset e não produz mapa de suscetibilidade.

```powershell
& C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe -m src.models.evaluate_spatial_buffer
& C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe -m pytest -q
```

As saídas ficam exclusivamente em `outputs/models/spatial_validation_buffer300/`.
O executável não oferece opções de seed ou raio. O código rejeita diretórios
de saída iguais, ancestrais ou descendentes do diretório principal protegido.

## Separação e preservação do protocolo

A atribuição oficial é carregada de `fold_assignments.csv`, verificada contra
o dataset e a configuração original, e copiada byte a byte para o diretório
de sensibilidade. Não há redefinição dos folds de validação.

Em cada fold, calcula-se a distância euclidiana, em EPSG:31982 (metros), entre
cada centro candidato de treino e todos os centros de validação. Remove-se do
treino a amostra cuja menor distância seja **estritamente inferior a 300 m**.
Distâncias exatamente iguais a 300 m são permitidas. A validação não sofre
remoções, portanto as mesmas 540 amostras continuam sendo avaliadas exatamente
uma vez. As classes de todos os conjuntos são verificadas antes de qualquer
ajuste; um conjunto vazio ou com classe ausente provoca falha explícita.

Os oito preditores, pipelines e hiperparâmetros são os mesmos da análise
principal. Coordenadas, IDs, row e col continuam excluídos dos preditores.
Cada fold ajusta novos scaler/encoder/modelo somente nas amostras de treino
restantes. Os testes interceptam o ajuste e verificam as linhas recebidas,
as categorias do encoder, as médias do scaler e os parâmetros dos modelos.
O limiar continua em 0,5. PR-AUC trapezoidal e average precision são distintas.

## Tamanho e composição após o buffer

Os conjuntos são idênticos para Logistic Regression e Random Forest.
L = landslide; B = background. A contagem de remoções é por fold: uma célula
pode ser excluída de mais de um treino, portanto sua soma não representa
necessariamente células distintas.

| Fold | Treino original | Treino restante | Removidas | Treino L/B | Removidas L/B | Validação L/B | Distância mínima original → buffer (m) |
|---|---:|---:|---:|---|---|---|---|
| 1 | 424 | 404 | 20 | 192 / 212 | 16 / 4 | 62 / 54 | 134,164 → 300,000 |
| 2 | 443 | 422 | 21 | 208 / 214 | 17 / 4 | 45 / 52 | 60,000 → 300,000 |
| 3 | 482 | 475 | 7 | 248 / 227 | 2 / 5 | 20 / 38 | 201,246 → 300,000 |
| 4 | 406 | 378 | 28 | 180 / 198 | 21 / 7 | 69 / 65 | 60,000 → 300,000 |
| 5 | 405 | 381 | 24 | 188 / 193 | 8 / 16 | 74 / 61 | 67,082 → 301,496 |

As remoções correspondem a 4,72%, 4,74%, 1,45%, 6,90% e 5,93% dos treinos
originais. Há ambas as classes em todos os conjuntos restantes e de validação.
A distância mínima global observada aumenta de 60 m para **300 m**.

## Métricas com buffer por fold

BA = balanced accuracy; AP = average precision. Valores arredondados;
`fold_metrics.csv` mantém a precisão completa. `fold_comparison.csv` contém,
para todas as métricas de cada modelo/fold, o valor principal, o valor com
buffer e a diferença buffer menos principal.

| Modelo | Fold | Accuracy | BA | Precision | Recall | F1 | ROC-AUC | PR-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1 | 0,8017 | 0,7990 | 0,8000 | 0,8387 | 0,8189 | 0,8548 | 0,8737 | 0,8750 |
| Logistic Regression | 2 | 0,7835 | 0,7936 | 0,7000 | 0,9333 | 0,8000 | 0,8915 | 0,8119 | 0,8241 |
| Logistic Regression | 3 | 0,7759 | 0,7579 | 0,6667 | 0,7000 | 0,6829 | 0,8697 | 0,7132 | 0,7392 |
| Logistic Regression | 4 | 0,7388 | 0,7379 | 0,7361 | 0,7681 | 0,7518 | 0,8308 | 0,8210 | 0,8230 |
| Logistic Regression | 5 | 0,7556 | 0,7626 | 0,8361 | 0,6892 | 0,7556 | 0,8392 | 0,8665 | 0,8676 |
| Random Forest | 1 | 0,7759 | 0,7772 | 0,8103 | 0,7581 | 0,7833 | 0,8617 | 0,8918 | 0,8927 |
| Random Forest | 2 | 0,7320 | 0,7410 | 0,6610 | 0,8667 | 0,7500 | 0,8201 | 0,7308 | 0,7363 |
| Random Forest | 3 | 0,7586 | 0,7092 | 0,6875 | 0,5500 | 0,6111 | 0,8645 | 0,8131 | 0,8174 |
| Random Forest | 4 | 0,7015 | 0,7017 | 0,7164 | 0,6957 | 0,7059 | 0,8301 | 0,8201 | 0,8229 |
| Random Forest | 5 | 0,7704 | 0,7718 | 0,8116 | 0,7568 | 0,7832 | 0,8531 | 0,8856 | 0,8864 |

## Comparação das médias e dispersão

Média não ponderada dos cinco folds ± desvio padrão amostral (`ddof=1`).
Δ = buffer menos principal, em unidades da métrica. DP não é intervalo de
confiança; esta análise não contém teste de significância ou margem de
equivalência pré-especificada. Mínimos e máximos também estão no JSON.

| Modelo | Métrica | Principal: média ± DP | Buffer: média ± DP | Δ |
|---|---|---:|---:|---:|
| LR | Accuracy | 0,7758 ± 0,0252 | 0,7711 ± 0,0245 | −0,0047 |
| LR | Balanced accuracy | 0,7746 ± 0,0249 | 0,7702 ± 0,0256 | −0,0044 |
| LR | Precision | 0,7479 ± 0,0654 | 0,7478 ± 0,0699 | −0,0001 |
| LR | Recall | 0,7980 ± 0,1083 | 0,7859 ± 0,1019 | −0,0121 |
| LR | F1 | 0,7676 ± 0,0569 | 0,7618 ± 0,0526 | −0,0058 |
| LR | ROC-AUC | 0,8600 ± 0,0253 | 0,8572 ± 0,0243 | −0,0028 |
| LR | PR-AUC | 0,8191 ± 0,0636 | 0,8173 ± 0,0642 | −0,0018 |
| LR | Average precision | 0,8276 ± 0,0534 | 0,8258 ± 0,0540 | −0,0018 |
| RF | Accuracy | 0,7549 ± 0,0348 | 0,7477 ± 0,0309 | −0,0072 |
| RF | Balanced accuracy | 0,7482 ± 0,0163 | 0,7402 ± 0,0347 | −0,0080 |
| RF | Precision | 0,7544 ± 0,0654 | 0,7374 ± 0,0700 | −0,0171 |
| RF | Recall | 0,7450 ± 0,1153 | 0,7254 ± 0,1158 | −0,0195 |
| RF | F1 | 0,7414 ± 0,0342 | 0,7267 ± 0,0720 | −0,0147 |
| RF | ROC-AUC | 0,8476 ± 0,0144 | 0,8459 ± 0,0198 | −0,0017 |
| RF | PR-AUC | 0,8265 ± 0,0591 | 0,8283 ± 0,0654 | +0,0018 |
| RF | Average precision | 0,8295 ± 0,0571 | 0,8311 ± 0,0634 | +0,0016 |

## Conclusão qualitativa

**As conclusões centrais permanecem estáveis sob o buffer de 300 m, com
ressalvas por métrica e fold.** Logistic Regression mantém médias superiores
de accuracy, balanced accuracy, recall, F1 e ROC-AUC. Random Forest mantém
PR-AUC e average precision médias ligeiramente superiores. A pequena vantagem
de precision de RF na análise principal inverte-se: LR passa de 0,7479 para
0,7478 e RF de 0,7544 para 0,7374. Portanto, não há estabilidade absoluta de
todas as ordenações nem fundamento para declarar um vencedor universal.

ROC-AUC varia pouco em média (LR: −0,0028; RF: −0,0017), mas recall cai 0,0121
e 0,0195, respectivamente. Os resultados continuam abaixo dos baselines
aleatórios nas médias das oito métricas. Isso é compatível com a leitura
anterior de avaliação espacial mais exigente; não constitui demonstração
causal sobre autocorrelação.

As médias escondem sensibilidade local: no fold 3, RF mantém ROC-AUC elevado
(0,8632 → 0,8645), mas recall cai de 0,6000 para 0,5500 e F1 de 0,6857 para
0,6111. Nesse fold, há somente 20 positivos; a queda do recall corresponde
a um positivo adicional não detectado. No fold 4, recall cai de 0,8261 para
0,7681 em LR e de 0,7826 para 0,6957 em RF. No fold 5 há melhora em ambos
os modelos. O DP do F1 de RF cresce de 0,0342 para 0,0720, reforçando a
necessidade de registrar a heterogeneidade entre folds.

O buffer garante a distância mínima entre os centros observados, mas não
independência espacial completa ou separação por evento. A exclusão altera
tamanho e composição do treino; seus efeitos não podem ser separados dos
efeitos de maior distância por esta única comparação. O estudo é uma
sensibilidade pré-definida, não tuning nem prova estatística de equivalência.
A composição balanceada do dataset também limita extrapolações de precision
e métricas PR para a prevalência municipal real.

## Evidências e artefatos

- `fold_assignments.csv`: cópia idêntica à principal, sem mudanças de validação.
- `buffer_membership.csv`: 2.700 registros (540 × 5), com papel de cada célula
  em cada fold e distância à validação: treino, removida ou validação.
- `fold_class_counts.csv`: tamanhos, remoções por classe e distância mínima.
- `fold_metrics.csv`, `confusion_matrices.csv`, `oof_predictions.csv`: resultados
  por fold, matrizes e previsões das mesmas 540 células.
- `principal_vs_buffer300.csv`: comparação de média/DP das oito métricas.
- `fold_comparison.csv`: diferenças pareadas em cada fold, para ambas as classes
  de modelos, sem confundir essas diferenças com médias de validações distintas.
- `summary_metrics.json`: configuração, parâmetros, estatísticas, diagnósticos,
  versões, SHA-256 de cada fonte protegida e verificação de preservação.
- `principal_vs_buffer300.png`: comparação visual com a mesma escala e DP.

Não houve warning de ajuste ou convergência. Categorias de `land_cover`
desconhecidas no treino permanecem nos folds 1 (41) e 5 (24), uma amostra
por fold, tratadas pelo encoder. Os artefatos oficiais e o dataset são
verificados por hash antes e depois da execução.

Testes específicos: cálculo independente das distâncias; exclusão exata de
distâncias menores que 300 m; retenção na fronteira de 300 m; preservação das
540 validações; disjunção espacial; presença das duas classes; falha antes
do ajuste quando inviável; contrato dos pipelines; proteção do diretório
principal; métricas, diferenças e exportação sem alteração de fontes.

Suíte completa executada pelo Python solicitado da `.venv`: **111 passed,
85 warnings in 74.34s**, incluindo 13 testes adicionais nesta análise.
Os 85 warnings são de depreciação do operador `*` em `Affine`, nas rotinas
da Etapa 3, e não de ajuste dos modelos. O gráfico de comparação foi
inspecionado visualmente. Dataset e todos os 11 arquivos oficiais permanecem
idênticos aos hashes coletados antes de implementar a sensibilidade.
