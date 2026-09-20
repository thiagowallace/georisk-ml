# Etapa 3 — preparação das features e modelos baseline

## Escopo e situação da execução

Esta etapa compara Logistic Regression e Random Forest com parâmetros fixos,
sem tuning, remoção de outliers, agrupamento de categorias raras ou serialização
de modelos. Entrada única: `data/processed/santa_tereza_ml_dataset.csv`.

**A divisão estratificada aleatória não controla autocorrelação espacial e não
constitui a avaliação final do projeto. A Etapa 4 substituirá esta avaliação
por validação espacial.** Os resultados não estabelecem superioridade
definitiva de um modelo nem probabilidade territorial calibrada: background
representa pseudoausência e o balanceamento 1:1 decorre da amostragem.

A implementação e a execução foram concluídas com o Python 3.10.11 da `.venv`
oficial, sem alterar o ambiente. A falha inicial de execução era uma restrição
de acesso ao Python base no contexto de execução; o mesmo executável funcionou
fora dessa restrição. Não foi utilizado outro Python como substituto.

## Contrato de dados

Preditores exclusivos, na ordem de entrada:

1. `elevation`
2. `slope`
3. `aspect_sin`
4. `aspect_cos`
5. `plan_curvature`
6. `profile_curvature`
7. `ndvi_pre_event`
8. `land_cover`

Target: `target`, com 1 = landslide e 0 = background. Não entram em X:
`aspect`, `x`, `y`, `row`, `col`, `sample_id`, `cell_id`, `landslide_count`,
`landslide_ids`, `sample_type` ou `target`.

IDs opcionais são copiados para tabelas de resultados exclusivamente para
rastreabilidade. `source_row` identifica a posição original no CSV, iniciada
em zero. O gerador oficial exige 540 linhas, 270 por classe e células únicas.
Preditores devem ser finitos, sem NoData; componentes circulares e intervalos
físicos são conferidos. Não há imputação nem exclusão de amostras.

## Divisão preliminar e prevenção de leakage

Usa-se exatamente `train_test_split(test_size=0.25, stratify=y, random_state=42)`
sobre a ordem original do CSV. Para 540 linhas, esperam-se 405 em treino e 135
em teste. Como ambos os tamanhos são ímpares, as classes diferem por uma amostra
em cada subconjunto. As contagens realizadas são registradas em `metrics.json`.

Cada modelo possui seu próprio `Pipeline` e `ColumnTransformer`, inicialmente
não ajustados. Apenas `pipeline.fit(X_train, y_train)` aprende transformações
e modelo. O teste passa somente por `transform` e `predict_proba`.

`OneHotEncoder(handle_unknown="ignore", sparse_output=False)` aprende categorias
exclusivamente no treino. Categorias presentes apenas no teste produzem zeros
no bloco categórico, sem refit; nomes e quantidades dessas ocorrências são
registrados. Não há categorias fixadas olhando a tabela inteira, agrupamento
de códigos raros ou interpretação ordinal dos números MapBiomas.

## Parâmetros fixos

| Componente | Logistic Regression | Random Forest |
|---|---|---|
| Contínuas | `StandardScaler`, somente sete contínuas | Sem padronização (`passthrough`) |
| Categórica | One-hot de `land_cover` | One-hot de `land_cover` |
| Regularização | L2, `C=1.0` | Restrição da complexidade das árvores |
| Solver | `lbfgs` | Não aplicável |
| Iterações/árvores | `max_iter=5000` | `n_estimators=500` |
| Profundidade máxima | Não aplicável | `max_depth=8` |
| Mínimo por folha | Não aplicável | `min_samples_leaf=5` |
| Mínimo para divisão | Não aplicável | `min_samples_split=10` |
| Features por divisão | Não aplicável | `max_features="sqrt"` |
| Bootstrap | Não aplicável | `True` |
| Class weights | `None` | `None` |
| Seed | 42 | 42 |
| Paralelismo | Padrão do solver | `n_jobs=1` |

Os demais parâmetros mantêm os defaults da versão registrada do scikit-learn.
As 500 árvores reduzem variabilidade de Monte Carlo, sem garantia prévia de
estabilidade completa. Profundidade e tamanhos mínimos restringem ajuste a
amostras isoladas. Não foram escolhidos com base em desempenho no teste.

5.000 iterações dão margem à convergência, mas não a garantem: warnings de
ajuste, presença de `ConvergenceWarning` e número de iterações realizadas são
registrados. Resultados com esse warning exigem revisão antes de interpretação.

## Métricas e interpretação

Treino e teste recebem accuracy, balanced accuracy, precision, recall, F1,
ROC-AUC e PR-AUC. A classe positiva é 1; o limiar fixo é probabilidade ≥ 0,5.
Precision/recall/F1 usam `zero_division=0`.

**PR-AUC é a integral trapezoidal `auc(recall, precision)` calculada a partir de
`precision_recall_curve`.** Também se exporta `average_precision_score`, que
usa outra definição e não é apresentado como sinônimo de PR-AUC. As curvas
ROC e PR usam probabilidades, não rótulos binários. A referência horizontal de
PR usa a proporção de positivos efetivamente presente no teste.

Matrizes de confusão têm linhas = observado e colunas = predito, na ordem
[background (0), landslide (1)]: `[[TN, FP], [FN, TP]]`.

`train_minus_test` registra cada diferença de desempenho. Lacunas positivas
grandes podem sinalizar sobreajuste, mas um único holdout pequeno não mede sua
estabilidade. Nenhum limiar ou parâmetro é ajustado a partir desses resultados.

Coeficientes da Logistic Regression referem-se à classe landslide e preservam
nomes transformados, incluindo `land_cover__land_cover_3`, por exemplo. São
ordenados por magnitude absoluta. Contínuas estão em unidades de um desvio
padrão do treino; colunas categóricas representam indicadores 0/1. O intercepto
é registrado separadamente. Como o one-hot completo é mantido com regularização,
os coeficientes das categorias não são contrastes contra uma categoria omitida.

Importâncias da Random Forest são reduções normalizadas de impureza do modelo,
ordenadas de forma decrescente, por coluna transformada. Podem favorecer
variáveis com mais possibilidades de corte e não substituem avaliação fora da
amostra. Ambos os resultados são interpretação dos modelos, **não causalidade**.

## Execução com o executável oficial

Na raiz do repositório:

```powershell
& 'C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests
& 'C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe' -B -m src.models.train_baselines
& 'C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe' -B -m src.models.evaluate_baselines
```

O último comando verifica métricas e predições de teste já salvas e regenera
figuras, sem retreinar. Os módulos devem ser executados com `-m`, a partir da
raiz. As dependências já estão listadas em `requirements.txt`.

## Saídas em outputs/models/baseline

- `metrics.json`: métricas de treino/teste, matrizes, diferenças, parâmetros,
  divisão, features, avisos, categorias, SHA-256 da entrada e versões do ambiente.
- `logistic_coefficients.csv`: nomes, coeficientes e magnitudes absolutas.
- `random_forest_importance.csv`: nomes e importâncias por coluna transformada.
- `test_predictions.csv`: rótulo observado, probabilidades e predições dos dois
  modelos, com rastreabilidade de linhas/células.
- `split_membership.csv`: associação de cada amostra ao treino ou teste.
- `confusion_matrix_logistic.png` e `confusion_matrix_random_forest.png`.
- `roc_curves.png` e `precision_recall_curves.png`.

Uma nova execução substitui esses arquivos; não são serializados estimadores.
O dataset é lido sem alterações, e seu conteúdo é conferido antes de exportar.
Reprodutibilidade refere-se a divisão, probabilidades, coeficientes,
importâncias e métricas no mesmo ambiente, não ao timestamp de geração nem a
identidade binária entre diferentes plataformas/bibliotecas.

## Testes

`tests/test_baseline_models.py` cobre contrato de features, exclusão das colunas
proibidas, split determinístico/estratificado/disjunto, transformadores
independentes, ajuste do scaler e encoder somente no treino, categoria inédita
no teste, valores contínuos sem escala na floresta, probabilidades/métricas,
cálculo manual de ROC-AUC/PR-AUC/AP, nomes transformados e reprodução do ajuste.
Um teste de integração sintético verifica resultados e quatro figuras; o
contrato do split real é testado quando o CSV está disponível. Nenhum teste
treina usando coordenadas ou IDs como preditores.

## Resultados da execução inicial

Ambiente: Python 3.10.11, scikit-learn 1.7.2, NumPy 2.2.6, Pandas 2.3.3,
SciPy 1.15.3 e Matplotlib 3.10.9. A suíte completa terminou com **82 testes
aprovados**. Houve 85 avisos `PendingDeprecationWarning` relativos à operação
Affine/Rasterio em testes existentes do dataset, sem falhas. Ambos os ajustes
baseline terminaram sem warnings; a Logistic Regression usou 21 iterações.

Divisão: treino com 405 células (202 background, 203 landslide); teste com
135 (68 background, 67 landslide). O código MapBiomas 41 apareceu somente no
teste, em uma célula, e foi processado como categoria desconhecida pelo encoder,
sem alteração das categorias aprendidas no treino.

| Métrica | Logistic treino | Logistic teste | Forest treino | Forest teste |
|---|---:|---:|---:|---:|
| Accuracy | 0,7802 | 0,8074 | 0,8815 | 0,8370 |
| Balanced accuracy | 0,7802 | 0,8079 | 0,8815 | 0,8371 |
| Precision | 0,7689 | 0,7662 | 0,8744 | 0,8261 |
| Recall | 0,8030 | 0,8806 | 0,8916 | 0,8507 |
| F1 | 0,7855 | 0,8194 | 0,8829 | 0,8382 |
| ROC-AUC | 0,8610 | 0,8870 | 0,9591 | 0,8962 |
| PR-AUC trapezoidal | 0,8366 | 0,8822 | 0,9604 | 0,8839 |
| Average Precision | 0,8395 | 0,8832 | 0,9605 | 0,8852 |

Matrizes de teste: Logistic Regression `[[50,18],[8,59]]`; Random Forest
`[[56,12],[10,57]]`, ambas na convenção `[[TN,FP],[FN,TP]]`.

Os cinco maiores coeficientes absolutos da regressão são slope (+1,1767),
elevation (−0,8968), indicador MapBiomas 15 (+0,7612), indicador MapBiomas 9
(−0,5892) e NDVI (+0,4606). A interpretação é condicional às demais variáveis,
à regularização e ao pré-processamento, não à associação marginal da EDA.

As cinco maiores importâncias de impureza da floresta são slope (0,3224),
elevation (0,1795), aspect_cos (0,1191), aspect_sin (0,1086) e NDVI (0,0977).

Na floresta, treino menos teste corresponde a +0,0444 em accuracy, +0,0630 em
ROC-AUC e +0,0765 em PR-AUC: um sinal de sobreajuste a acompanhar. Na regressão,
o teste teve métricas geralmente superiores ao treino neste split, sem uma
lacuna geral de treino favorável. Isso não demonstra ausência de sobreajuste
nem garante generalização espacial. As diferenças entre os modelos neste
holdout não estabelecem superioridade definitiva.

As métricas de teste exportadas foram conferidas pelo módulo de avaliação e
as quatro figuras foram inspecionadas. O hash do CSV permaneceu
`334fd5bcd190ffc4ea7ad4d34debecce8d1c4af6910e663411eab6dd5b379162`.
