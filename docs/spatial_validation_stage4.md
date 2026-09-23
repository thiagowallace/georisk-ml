# Etapa 4 — validação espacial por blocos

## Objetivo e execução

A avaliação principal passa a ser a validação cruzada espacial, preservando a
Etapa 3 como referência histórica. Entrada: `data/processed/santa_tereza_ml_dataset.csv`,
540 células únicas, 270 landslide e 270 background. O dataset e os artefatos do
baseline não são alterados. Não há tuning, treinamento de modelo final ou mapa
de suscetibilidade nesta etapa.

```powershell
& C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe -m src.models.evaluate_spatial_validation --n-splits 5 --block-size-m 1500 --random-state 42
& C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe -m pytest -q
```

Também são configuráveis `--input`, `--baseline` e `--output-dir`. A execução
oficial exige 540 amostras e igualdade do SHA-256 com a fonte registrada no
baseline. Para configurações exploratórias, use outro diretório de saída e
declare previamente a escala espacial; não escolha blocos ou sementes pelo
melhor resultado. A API de particionamento aceita outros tamanhos de amostra.

## Geometria e atribuição

Coordenadas: EPSG:31982, SIRGAS 2000 / UTM 22S, metros, conforme
`docs/dataset_stage3.md`. Cada pixel mede 30 × 30 m. Blocos quadrados de
1.500 × 1.500 m (2,25 km²; 50 × 50 células) são alinhados às bordas dos pixels.
A escala é uma escolha operacional inicial, anterior à avaliação, não uma
estimativa do alcance da autocorrelação e não um hiperparâmetro otimizado.

A origem sudoeste é `(min(x) - 15, min(y) - 15)`:
`(426579.046476552, 6768267.17810355)` m. Os índices inteiros dos pixels relativos
a essa origem são divididos por 50, com divisão inteira. Os intervalos dos
blocos são semiabertos `[mínimo, máximo)`, sem duplicação de células. A grade
envolvente tem 7 colunas × 10 linhas, com 42 blocos ocupados.

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` atribui blocos
inteiros aos folds, procurando preservar as proporções de classes. Os rótulos
são usados somente para a estratificação, não para escolher a escala ou buscar
sementes. Não há nova tentativa automática se algum fold for inválido. Todos
os folds, tanto no treino como na validação, precisam conter as duas classes;
caso contrário, a execução falha antes do ajuste dos modelos.

Cada amostra é validada exatamente uma vez e participa dos outros quatro
treinos. Nenhum bloco, célula ou coordenada pertence simultaneamente ao treino
e à validação dentro de um fold. Os folds podem reunir blocos desconectados.
Não há buffer: blocos de folds distintos podem ser vizinhos, e 1.500 m **não é**
uma distância mínima entre amostras de treino e validação. As distâncias mínimas
observadas entre centros são registradas abaixo. A independência espacial
completa, inclusive entre células do mesmo evento em blocos distintos, não é
garantida. A metodologia controla a partilha de blocos, mas não equivale a
validação por regiões contíguas ou por eventos.

| Fold | Treino | Validação | Landslide | Background | Blocos de validação | Distância mínima (m) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 424 | 116 | 62 | 54 | 9 | 134,16 |
| 2 | 443 | 97 | 45 | 52 | 8 | 60,00 |
| 3 | 482 | 58 | 20 | 38 | 7 | 201,25 |
| 4 | 406 | 134 | 69 | 65 | 10 | 60,00 |
| 5 | 405 | 135 | 74 | 61 | 8 | 67,08 |
| Total validado | — | 540 | 270 | 270 | 42 | — |

O fold 3 é menor e tem 34,5% de positivos; a estratificação respeita a integridade
dos blocos, sem impor igualdade artificial de tamanhos. Contagens das classes
de treino também constam em `fold_class_counts.csv` e `summary_metrics.json`.

## Preditores, ajuste e métricas

Preditores, nesta ordem: `elevation`, `slope`, `aspect_sin`, `aspect_cos`,
`plan_curvature`, `profile_curvature`, `ndvi_pre_event`, `land_cover`.
`x` e `y` servem exclusivamente à partição, auditoria da separação e sua figura;
coordenadas, `row`, `col`, IDs, alvo e atributos do inventário não entram no modelo.

Cada fold cria pipelines novos por meio das funções da Etapa 3:

- Logistic Regression: `StandardScaler` nas sete contínuas e
  `OneHotEncoder(handle_unknown="ignore")` em `land_cover`.
- Random Forest: contínuas sem scaling e o mesmo encoder categórico.
- Todo `fit`, inclusive dos transformers, recebe exclusivamente as linhas do
  treino. Categorias desconhecidas na validação são codificadas com zeros.

Hiperparâmetros preservados: Logistic Regression usa `penalty=l2`, `C=1`,
`solver=lbfgs`, `max_iter=5000`, `random_state=42`, `class_weight=None`.
Random Forest usa `n_estimators=500`, `max_depth=8`, `min_samples_leaf=5`,
`min_samples_split=10`, `max_features=sqrt`, `bootstrap=True`, `random_state=42`,
`n_jobs=1`, `class_weight=None`. A comparação verifica esses parâmetros contra
`outputs/models/baseline/metrics.json`.

Classe positiva: landslide=1; limiar fixo: 0,5. Precision, recall e F1 são
binários, com `zero_division=0`. ROC-AUC usa probabilidades. PR-AUC é a área
trapezoidal da curva precision-recall; average precision é não interpolada,
calculada separadamente. Matrizes são exportadas como TN, FP, FN e TP.

## Resultados por fold

Valores arredondados a quatro casas; CSVs preservam a precisão numérica.
BA = balanced accuracy, AP = average precision.

| Modelo | Fold | Accuracy | BA | Precision | Recall | F1 | ROC-AUC | PR-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1 | 0,8103 | 0,8070 | 0,8030 | 0,8548 | 0,8281 | 0,8542 | 0,8731 | 0,8744 |
| Logistic Regression | 2 | 0,7835 | 0,7936 | 0,7000 | 0,9333 | 0,8000 | 0,8966 | 0,8160 | 0,8281 |
| Logistic Regression | 3 | 0,7759 | 0,7579 | 0,6667 | 0,7000 | 0,6829 | 0,8737 | 0,7150 | 0,7410 |
| Logistic Regression | 4 | 0,7687 | 0,7669 | 0,7500 | 0,8261 | 0,7862 | 0,8363 | 0,8232 | 0,8253 |
| Logistic Regression | 5 | 0,7407 | 0,7477 | 0,8197 | 0,6757 | 0,7407 | 0,8392 | 0,8681 | 0,8691 |
| Random Forest | 1 | 0,7672 | 0,7691 | 0,8070 | 0,7419 | 0,7731 | 0,8611 | 0,8885 | 0,8894 |
| Random Forest | 2 | 0,7320 | 0,7440 | 0,6508 | 0,9111 | 0,7593 | 0,8325 | 0,7396 | 0,7451 |
| Random Forest | 3 | 0,8103 | 0,7605 | 0,8000 | 0,6000 | 0,6857 | 0,8632 | 0,8152 | 0,8194 |
| Random Forest | 4 | 0,7388 | 0,7375 | 0,7297 | 0,7826 | 0,7552 | 0,8343 | 0,8148 | 0,8183 |
| Random Forest | 5 | 0,7259 | 0,7298 | 0,7846 | 0,6892 | 0,7338 | 0,8467 | 0,8744 | 0,8754 |

## Resumo e comparação com o baseline aleatório

Médias não ponderadas dos cinco folds; desvio padrão amostral (`ddof=1`).
O desvio mostra dispersão entre folds, não intervalo de confiança. Mínimo e
máximo de todas as métricas estão em `summary_metrics.json`. Não confundir
média por fold com métricas calculadas sobre todas as previsões OOF juntas.

| Métrica | Logistic Regression: média ± DP | Baseline LR | Δ espacial − baseline | Random Forest: média ± DP | Baseline RF | Δ espacial − baseline |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0,7758 ± 0,0252 | 0,8074 | −0,0316 | 0,7549 ± 0,0348 | 0,8370 | −0,0822 |
| Balanced accuracy | 0,7746 ± 0,0249 | 0,8079 | −0,0333 | 0,7482 ± 0,0163 | 0,8371 | −0,0889 |
| Precision | 0,7479 ± 0,0654 | 0,7662 | −0,0184 | 0,7544 ± 0,0654 | 0,8261 | −0,0717 |
| Recall | 0,7980 ± 0,1083 | 0,8806 | −0,0826 | 0,7450 ± 0,1153 | 0,8507 | −0,1058 |
| F1 | 0,7676 ± 0,0569 | 0,8194 | −0,0518 | 0,7414 ± 0,0342 | 0,8382 | −0,0968 |
| ROC-AUC | 0,8600 ± 0,0253 | 0,8870 | −0,0270 | 0,8476 ± 0,0144 | 0,8962 | −0,0486 |
| PR-AUC | 0,8191 ± 0,0636 | 0,8822 | −0,0631 | 0,8265 ± 0,0591 | 0,8839 | −0,0574 |
| Average precision | 0,8276 ± 0,0534 | 0,8832 | −0,0556 | 0,8295 ± 0,0571 | 0,8852 | −0,0557 |

Todas as médias ficaram abaixo do teste aleatório salvo. Isso é compatível com
uma avaliação espacial mais realista, e não demonstra erro na implementação.
A comparação é descritiva: o baseline usa um único teste de 135 amostras,
enquanto a validação espacial cobre 540 amostras ao longo de cinco folds, com
diferentes tamanhos de treino. Não se atribui causalmente toda a diferença à
autocorrelação. Logistic Regression supera RF em ROC-AUC, recall e F1 médios
nesta configuração, mas isso não estabelece superioridade geral.

Extremos que merecem atenção:

- LR: fold 2 tem maior ROC-AUC (0,8966) e recall (0,9333); fold 4 tem menor
  ROC-AUC (0,8363). Fold 5 tem menor accuracy (0,7407) e recall (0,6757).
  Fold 3 tem menor F1 (0,6829) e PR-AUC (0,7150).
- RF: fold 3 tem maior ROC-AUC (0,8632) e accuracy (0,8103), mas menor recall
  (0,6000) e F1 (0,6857). A proporção maior de background torna accuracy
  isolada especialmente insuficiente nesse fold. Fold 2 tem menor ROC-AUC
  (0,8325), mas maior recall (0,9111). Fold 5 tem menor accuracy (0,7259).

## Warnings, rastreabilidade e testes

Não ocorreram warnings de ajuste/convergência. Há uma categoria de `land_cover`
não vista no treino no fold 1 (41, uma amostra) e no fold 5 (24, uma amostra),
em ambos os modelos; são tratadas por `handle_unknown="ignore"` e registradas.
Os avisos metodológicos do JSON cobrem ausência de buffer, escala não estimada
por autocorrelação, blocos desconectados, dependência potencial entre células
do mesmo evento e limitações da comparação com holdout. A amostra balanceada
não representa a prevalência municipal; precision, PR-AUC e AP dependem dessa
composição. Probabilidades não são apresentadas como risco municipal calibrado.

Saídas em `outputs/models/spatial_validation/`:

- `fold_assignments.csv`: vínculo fonte/célula/coordenadas/bloco/fold.
- `fold_class_counts.csv`: tamanhos, classes, blocos e distância mínima.
- `fold_metrics.csv`, `confusion_matrices.csv`: evidência por modelo e fold.
- `summary_metrics.json`: média, DP, mínimo, máximo, configuração, parâmetros,
  diagnósticos, versões e hashes das fontes.
- `baseline_vs_spatial.csv`: teste aleatório, média/DP espacial e diferença.
- `oof_predictions.csv`: probabilidades e decisões de validação de cada célula.
- `spatial_folds.png`, `metric_comparison.png`, `roc_auc_by_fold.png`,
  `recall_by_fold.png`: partição e comparação, inspecionadas visualmente.

SHA-256 do dataset:
`334fd5bcd190ffc4ea7ad4d34debecce8d1c4af6910e663411eab6dd5b379162`.

Os testes cobrem 540 atribuições únicas, disjunção de amostras e blocos,
classes em treino/validação, configuração reproduzível, entrada inválida,
ajuste exclusivamente no treino (incluindo interceptação da função de ajuste,
médias do scaler e categorias do encoder), exclusão de preditores proibidos,
probabilidades/métricas válidas, agregação, matrizes e exportação sem alteração
da fonte. Categorias presentes apenas na validação são testadas explicitamente.

Nesta máquina, `.venv/pyvenv.cfg` apontava para um Python removido. Foi preservado
um backup em `.venv/pyvenv.cfg.stage4-backup` e instalado o runtime oficial
Python 3.10.11 embutido em `.venv/runtime310`, com ajuste local do `home`.
O executável solicitado `.venv/Scripts/python.exe` passou a funcionar com as
dependências existentes; não foi necessário modificar `requirements.txt`.

Resultado da suíte completa pelo executável solicitado: **98 passed, 85 warnings
in 37.76s**, incluindo 16 testes novos. Os 85 avisos são
`PendingDeprecationWarning` pelo operador `*` em transformações `Affine`:
29 em `rasterio/transform.py:178` e 56 em `src/features/build_ml_dataset.py:97`,
exercitados pelos testes da Etapa 3. Não são warnings dos modelos espaciais.

Nenhum commit foi criado. `git status --short` ao final:

```text
?? docs/spatial_validation_stage4.md
?? outputs/
?? scripts/
?? src/models/evaluate_spatial_validation.py
?? src/models/spatial_validation.py
?? tests/test_spatial_validation.py
```

`outputs/` e `scripts/` já estavam não rastreados antes desta etapa; dentro de
`outputs/`, esta execução acrescentou `models/spatial_validation/`.
