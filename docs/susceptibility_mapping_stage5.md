# Etapa 5 — produto espacial de suscetibilidade

## Produto e interpretação

Treinamento final de Logistic Regression (LR) e Random Forest (RF) com todas
as **540 amostras supervisionadas**, 270 positive e 270 background. Inferência
nas **80.931 células ambientalmente válidas** de Santa Tereza/RS. Não houve
tuning, nova comparação de desempenho, alteração do dataset ou classificação
dos scores em baixa/média/alta.

Os valores são **landslide susceptibility scores**. A amostragem balanceada
1:1 não representa a frequência territorial de ocorrência. Background é
pseudoausência. O `ensemble_mean_score` é apenas a média aritmética de LR e RF,
um produto exploratório adicional sem validação própria.

## Contrato de treinamento

As funções `select_features`, `make_pipelines`, `fit_baselines` e a seleção da
classe positiva de `src/models/train_baselines.py` são reutilizadas. As Etapas
3 e 4 permanecem intactas. O treinamento confere a lista ordenada de features,
os parâmetros e o SHA-256 do CSV contra `baseline/metrics.json`,
`spatial_validation/summary_metrics.json` e
`spatial_validation_buffer300/summary_metrics.json` antes de ajustar.

Preditores, nesta ordem:

1. `elevation`
2. `slope`
3. `aspect_sin`
4. `aspect_cos`
5. `plan_curvature`
6. `profile_curvature`
7. `ndvi_pre_event`
8. `land_cover`

LR usa `StandardScaler` nas sete contínuas; RF usa `passthrough`.
Ambos usam `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
em `land_cover`, preservando os códigos originais. Scaler e encoder são
ajustados nas 540 amostras, dentro de cada pipeline independente.
Coordenadas, índices de célula, IDs, `aspect` original, target e atributos do
inventário são excluídos de X por seleção explícita das oito features.

| Parâmetro | LR | RF |
|---|---|---|
| Regularização/solver | L2, C=1, lbfgs | — |
| Iterações/árvores | max_iter=5000 | n_estimators=500 |
| max_depth | — | 8 |
| min_samples_leaf / split | — | 5 / 10 |
| max_features / bootstrap | — | sqrt / True |
| class_weight | None | None |
| random_state | 42 | 42 |
| n_jobs | default | 1 |

Demais defaults pertencem ao scikit-learn 1.7.2; o metadata registra versões
e hashes dos modelos serializados. LR convergiu em 26 iterações. Os dois
ajustes terminaram sem warnings.

## Grade e máscara de inferência

As sete entradas ambientais e o polígono municipal são lidos dos mesmos
caminhos usados na Etapa 3. Cada raster precisa ter exatamente o mesmo CRS,
transform, largura, altura e uma banda. Não há reprojeção ou reamostragem.

A máscara é a interseção das células cujos centros estão estritamente dentro
do município com todas as sete variáveis finitas, sem máscara e diferentes do
NoData declarado. Das 81.774 células municipais, 80.931 são válidas; 843 são
excluídas pela interseção ambiental. A inferência inclui todos os pixels
elegíveis, sem aplicar exclusão de proximidade ao inventário ou amostragem.

`inference_grid.csv` contém `row`, `col`, `x`, `y` somente para reconstrução
espacial, seguidos das oito features. A ordem é crescente por linha e coluna.
Os centros são calculados pelo transform original. Os componentes circulares
reutilizam a função da Etapa 3: `sin/cos(deg2rad(aspect.astype(float64)))`.

Os hashes das fontes ambientais e a contagem da máscara são conferidos contra
o metadata da Etapa 3. Features e coordenadas das 540 células supervisionadas
são recuperadas da grade e comparadas com tolerância absoluta de 1e-10 e
tolerância relativa zero. Não são aceitos NaN/Inf ou valores físicos inválidos.

## GeoTIFFs e QA

Os três arquivos têm uma banda `float32`, compressão DEFLATE, NoData `-9999`,
CRS **EPSG:31982**, resolução **30 × 30 m** e **408 colunas × 570 linhas**.

```text
Transform: [30, 0, 425199.046476552, 0, -30, 6784257.17810355]
Bounds [west, south, east, north]:
[425199.046476552, 6767157.17810355, 437439.046476552, 6784257.17810355]
```

Todas as 80.931 células válidas recebem score; as demais 151.629 células da
extensão recebem NoData. O QA reabre os GeoTIFFs, confere a máscara lida pelo
Rasterio, todos os metadados espaciais e igualdade exata dos pixels com os
scores CSV convertidos para float32. Não há scores fora de [0,1].

O CSV preserva float64 com 17 dígitos significativos. A média do ensemble é
calculada antes da conversão para float32. O JSON apresenta tanto estatísticas
do CSV quanto dos pixels float32; o arredondamento do máximo LR para 1 no
GeoTIFF é esperado.

### Distribuição territorial dos susceptibility scores

| Estatística | LR | RF | Ensemble exploratório |
|---|---:|---:|---:|
| Células | 80.931 | 80.931 | 80.931 |
| Mínimo | 1,62093e-10 | 0,021189 | 0,011438 |
| Máximo | 0,999999999777 | 0,931413 | 0,940706 |
| Média | 0,331497 | 0,361140 | 0,346319 |
| Mediana / P50 | 0,263452 | 0,324682 | 0,295497 |
| P5 | 0,012158 | 0,084855 | 0,053509 |
| P25 | 0,084581 | 0,180026 | 0,136491 |
| P75 | 0,544660 | 0,512681 | 0,526927 |
| P95 | 0,845031 | 0,760912 | 0,791239 |

### Positive versus background

Cada grupo tem 270 células. Estes são scores nas próprias amostras de ajuste;
as distribuições são descritivas, sem interpretação causal ou avaliação de
generalização. Todos os percentis, mínimos e máximos estão no CSV e no JSON.

| Modelo | Média positive | Mediana positive | Média background | Mediana background |
|---|---:|---:|---:|---:|
| LR | 0,701761 | 0,760725 | 0,298231 | 0,211876 |
| RF | 0,723428 | 0,750832 | 0,278452 | 0,232935 |
| Ensemble exploratório | 0,712594 | 0,752611 | 0,288341 | 0,225144 |

### Comparação entre superfícies

Sobre todas as 80.931 células, com scores float64:

- Correlação de Pearson LR × RF: **0,918530**.
- Diferença média absoluta: **0,097897**.
- RMSE entre as superfícies: **0,118862**.
- Média de LR − RF: **−0,029642**.
- Intervalo de LR − RF: **[−0,392355; 0,876451]**.

Estes números descrevem concordância entre superfícies e não são métricas
de validação. Os resultados de validação espacial continuam nas Etapas 3 e 4.

## Figuras e limites

Foram produzidos e inspecionados seis PNGs: histogramas com bins e eixos
comuns, distribuições acumuladas por grupo, mapas LR, RF, ensemble e diferença.
Todos os mapas usam coordenadas reais EPSG:31982, pixels sem suavização e
escala visual de 0 a 1. NoData permanece mascarado.

Para representar o sinal de LR − RF mantendo a escala visual solicitada,
o mapa de diferença usa **(LR − RF + 1) / 2**, com 0,5 = igualdade,
0 = diferença −1 e 1 = diferença +1. A legenda declara a transformação;
ela não é um susceptibility score adicional. As estatísticas da diferença
permanecem na escala assinada original.

Há **91 células** com `land_cover` 12, 25 ou 39, ausentes no treinamento.
O encoder validado produz zeros no bloco categórico dessas células. Nenhum
código foi agrupado, substituído ou removido. Essa limitação consta nos
metadados e nos warnings do resumo.

## Arquivos e execução

Resultados em `outputs/models/final/`:

- `logistic.joblib`, `random_forest.joblib`, `final_model_metadata.json`.
- `inference_grid.csv`, `inference_grid_metadata.json`.
- `cell_scores.csv`, `prediction_metadata.json`, `susceptibility_summary.json`.
- `positive_background_score_summary.csv`, `model_surface_comparison.csv`.
- `santa_tereza_susceptibility_logistic.tif`.
- `santa_tereza_susceptibility_random_forest.tif`.
- `santa_tereza_susceptibility_ensemble_mean.tif`.
- `score_histograms.png`, `positive_background_distributions.png`.
- `map_logistic.png`, `map_random_forest.png`, `map_ensemble_mean.png`.
- `map_difference_lr_minus_rf.png`.
- `preservation_manifest.json`, logs de execução e testes.

Execução completa, a partir da raiz:

```powershell
& C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe -B -m src.models.generate_susceptibility
& C:\Users\thiagowallace\georisk-ml\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests
```

O primeiro comando treina, serializa, constrói a grade, infere, exporta e audita.
Cada módulo também é executável com `-m`; `evaluate_susceptibility` audita e
regenera figuras sem retreinar. Reexecuções substituem somente os produtos
da Etapa 5. Um manifest existente nunca é sobrescrito automaticamente.

## Preservação e testes

Antes da implementação foram registrados SHA-256 de 125 arquivos existentes
em `data`, `outputs`, `src`, `tests`, `docs` e `scripts`, excluindo caches.
A verificação posterior confirmou zero alterações. O dataset mantém SHA-256
`334fd5bcd190ffc4ea7ad4d34debecce8d1c4af6910e663411eab6dd5b379162`.
Os resultados das Etapas 1–4 permanecem intactos.

`tests/test_susceptibility_mapping.py` cobre os contratos de features e
parâmetros, scaler/encoder, códigos desconhecidos, grade, cobertura exata,
rejeição de scores inválidos, células repetidas/ausentes/externas, rasters
desalinhados e pixels indevidos fora da máscara. Os testes reais conferem
distribuições, preservação e repetem o ajuste completo e a inferência,
comparando scores exatamente e hashes dos três GeoTIFFs. Testes de produtos
reais são pulados somente quando os artefatos da Etapa 5 ainda não existem.

Reprodutibilidade se refere ao mesmo ambiente e versões registradas. Não
se assume identidade numérica entre outras versões ou plataformas.

A suíte completa foi executada com o Python 3.10.11 da `.venv` oficial:
**130 testes aprovados, nenhum pulado ou com falha**. Houve 101
`PendingDeprecationWarning` da multiplicação Affine/Rasterio: 85 nos testes
anteriores e 16 nos testes da Etapa 5. Nenhum warning de ajuste ou convergência
foi emitido pelos modelos finais. O log completo está em
`outputs/models/final/test_results.log`.
