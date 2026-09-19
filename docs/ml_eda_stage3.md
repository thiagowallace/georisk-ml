# Etapa 3 — análise exploratória e QA estatístico

## Escopo e execução

Status desta implementação: código e testes escritos, mas execução pendente.
A `.venv` ainda referencia uma instalação ausente de Python 3.10. As saídas
em `outputs/eda/`, os resultados estatísticos e a inspeção das figuras ainda
não foram gerados/validados nesta tarefa. Nenhum ambiente alternativo foi usado.

Entrada única: `data/processed/santa_tereza_ml_dataset.csv`. Unidade: uma célula
da grade de 30 × 30 m. Esperam-se 540 células, sendo 270 `landslide` (`target=1`)
e 270 `background` (`target=0`). Background representa pseudoausência, não
ausência confirmada. A proporção 1:1 não estima prevalência territorial.

Executar na raiz, com o ambiente do projeto funcional:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests
.\.venv\Scripts\python.exe -B -m src.features.explore_ml_dataset
```

O script grava somente em `outputs/eda/`, substituindo os arquivos de mesmo
nome em uma nova execução. Não altera o CSV nem dados brutos ou intermediários.
Não treina modelos, não faz seleção de features, divisão treino/teste ou
validação espacial. Não instala dependências.

## QA de entrada

Antes da análise, verificam-se colunas obrigatórias, 540 linhas, balanceamento,
consistência entre `target` e `sample_type`, valores ausentes em todas as
colunas, finitude dos campos numéricos e duplicidades de registros, `cell_id`
e `sample_id`. Valores NoData, códigos de cobertura não positivos/não inteiros,
NDVI fora de [-1,1], slope fora de [0,90] e aspecto fora de [0,360) são rejeitados.
Seno/cosseno são confrontados com o ângulo original. Falhas interrompem a EDA.

Vetores de features idênticos são contados separadamente: podem pertencer a
células diferentes e não são removidos automaticamente. A EDA não substitui a
auditoria geoespacial anterior; distâncias de exclusão e rasters não são relidos.

## Estatísticas e comparações

Para `elevation`, `slope`, `aspect_sin`, `aspect_cos`, `plan_curvature`,
`profile_curvature` e `ndvi_pre_event`, são calculados por classe: quantidade,
média, desvio padrão amostral (`ddof=1`), mínimo, percentis 5/25/50/75/95 e máximo.
Histogramas compartilham intervalos entre as classes e exibem percentuais com
denominador de cada classe. Boxplots preservam todos os valores observados.

Diferenças são sempre **landslide menos background**. A tabela de comparação
inclui diferença de médias, diferença de medianas, diferença média padronizada
(desvio padrão combinado ponderado pelos graus de liberdade) e correlação
rank-biserial (`2U/(n_landslide*n_background)-1`, com empates por postos médios).
Essas medidas descrevem separação amostral; não são importância preditiva,
evidência causal ou teste de significância. Não são calculados p-valores.

`aspect` é circular: 359° está próximo de 1°. Seu resumo específico contém
direção média circular, comprimento resultante médio R e variância circular
1−R. Se R ≤ 1e−12, a direção média é indefinida. A distribuição aparece em
histograma com indicação da quebra 0/360 e em rosas de orientação com norte
no topo, azimute horário e escalas radiais iguais. Não se interpreta média ou
boxplot linear do ângulo bruto.

## Correlação, outliers e MapBiomas

Pearson e Spearman são calculados para as sete variáveis contínuas apropriadas,
tanto na amostra conjunta quanto separadamente em cada classe. `aspect` bruto,
`land_cover`, rótulos e rastreabilidade não entram nas matrizes. Correlações
com |coeficiente| ≥ 0,80 são listadas. Matrizes por classe permitem verificar
se associações da amostra conjunta são influenciadas pela mistura das classes.
Seno/cosseno são derivados da mesma variável circular; interpretar associações
com essa dependência em mente. Variáveis constantes produzem coeficientes
indefinidos (vazio no CSV, null no JSON), nunca zeros artificiais.

Possíveis outliers são sinalizados por variável e classe quando o valor está
fora de [Q1−1,5 IQR, Q3+1,5 IQR]. Em IQR zero, valores diferentes dos quartis
podem ser sinalizados. Esse critério não determina erro de medição, sobretudo
em distribuições assimétricas e componentes limitados de aspecto. Não se aplica
ao ângulo bruto nem aos códigos MapBiomas. Não há remoção, clipping ou
winsorização. A tabela detalhada inclui IDs exclusivamente para inspeção.

`land_cover` mantém códigos originais categóricos: contagem e percentual por
classe, incluindo zeros para combinações não observadas. Não se calcula média
dos códigos nem correlação numérica deles; não há one-hot encoding. Nenhum
nome de classe é inferido sem uma legenda externa verificada.

`sample_id`, `cell_id`, `row`, `col`, `x`, `y`, `landslide_ids` e
`landslide_count` servem apenas ao QA/rastreabilidade e não entram nas análises
de preditores. A lista de variáveis analíticas é explícita, não inferida do tipo
numérico das colunas.

## Saídas em outputs/eda

- `class_distribution.csv` e `.png`: distribuição conjunta dos dois rótulos.
- `descriptive_by_class.csv`: estatísticas das sete variáveis contínuas.
- `continuous_comparison.csv`: diferenças e tamanhos de efeito descritivos.
- `aspect_circular.csv` e `.png`: estatísticas e orientação circular.
- `histograms.png` e `boxplots.png`: comparação das distribuições.
- `correlation_{pearson,spearman}_{pooled,landslide,background}.csv`: seis matrizes.
- `correlations.png`: matrizes conjuntas Pearson e Spearman.
- `high_correlations.csv`: pares com correlação absoluta ≥ 0,80 por escopo/método.
- `outlier_summary.csv` e `outlier_records.csv`: contagens, limites e valores.
- `land_cover_by_class.csv` e `land_cover.png`: composição MapBiomas.
- `statistical_summary.json`: QA, tabelas, metodologia resumida e SHA-256 da entrada.

As tabelas são ordenadas deterministicamente; o resumo não inclui timestamp
variável. Mesmos dados e mesmas versões devem reproduzir os resultados. Testes
comparam as tabelas em memória, inclusive após reordenação das linhas, e os
bytes de CSV/JSON entre execuções. A aparência ou bytes das imagens podem
variar entre versões do Matplotlib; testes verificam sua geração e a inspeção
visual verifica a legibilidade. Nenhuma amostragem aleatória é usada na EDA.

## Testes

`tests/test_ml_eda.py` inclui contrato de 540 linhas e balanceamento,
NaN/inf/NoData, colunas obrigatórias, duplicidades, consistência dos rótulos,
estatísticas de sequências conhecidas, aspecto em torno de 0/360°, outliers,
variáveis constantes, denominadores de cobertura e reprodutibilidade.
Os testes sintéticos independem dos dados geoespaciais. Há também verificação
do CSV local quando disponível e um teste de geração das seis figuras.
