# Etapa 3 — dataset amostral de Machine Learning

Esta etapa constrói e audita amostras de células com ocorrência de deslizamento e
background (pseudoausências). Não treina modelos, não divide treino/teste e não
implementa validação espacial. Os arquivos em `data/raw` e `data/interim` são
somente entradas de leitura.

## Ambiente e execução

Na implementação, `.venv/Scripts/python.exe` não estava funcional: o ambiente
referenciava uma instalação ausente de Python 3.10. A execução dos testes e a
validação completa com os dados reais permanecem pendentes da regularização da
`.venv`. Não foram gerados produtos desta etapa nem executados testes em outro
ambiente. As bibliotecas utilizadas já constam em `requirements.txt`.

Após regularizar o ambiente, executar a partir da raiz do repositório:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests
.\.venv\Scripts\python.exe -B -m src.features.build_ml_dataset
.\.venv\Scripts\python.exe -B -m src.transform.audit_ml_dataset
```

Parâmetros de geração:

```powershell
.\.venv\Scripts\python.exe -B -m src.features.build_ml_dataset --exclusion-radius 60 --seed 42
```

- `--exclusion-radius`: distância em metros, finita e não negativa; default 60.
- `--seed`: inteiro não negativo; default 42.
- `--background-count`: inteiro positivo; quando omitido, igual ao número de
  células positivas.
- `--overwrite`: necessário para substituir produtos já existentes.

O auditor lê os parâmetros do metadata, retorna código de saída diferente de
zero se encontrar inconsistências e não grava arquivos. O gerador prepara os
três arquivos em diretório temporário dentro de `data/processed` antes de mover
os produtos para seus destinos. A substituição dos três destinos não constitui
uma transação única; após qualquer interrupção, executar a auditoria antes de
usar os produtos.

## Entradas e grade

Todas as coordenadas usam EPSG:31982 (SIRGAS 2000 / UTM 22S), em metros.
O DEM define exclusivamente a grade; nenhum raster é reamostrado nesta etapa.
CRS, transformação afim, dimensões e banda única devem coincidir exatamente.
A grade deve ser orientada ao norte e ter pixels de 30 × 30 m.

| Variável | Caminho relativo ao repositório |
|---|---|
| elevation | `data/interim/dem/santa_tereza_dem_30m.tif` |
| slope | `data/interim/terrain_features/santa_tereza_slope.tif` |
| aspect | `data/interim/terrain_features/santa_tereza_aspect.tif` |
| plan_curvature | `data/interim/terrain_features/santa_tereza_plan_curvature.tif` |
| profile_curvature | `data/interim/terrain_features/santa_tereza_profile_curvature.tif` |
| land_cover | `data/interim/land_cover/santa_tereza_land_cover_2023.tif` |
| ndvi_pre_event | `data/interim/ndvi/santa_tereza_ndvi_pre_event_2024.tif` |

Entradas vetoriais:

- `data/interim/study_area/santa_tereza.gpkg`: limite municipal.
- `data/interim/study_area/landslide_initiation_santa_tereza.gpkg`: ocorrências
  que definem as células positivas.
- `data/interim/landslides_rs_2024/landslide_initiation.gpkg`: inventário estadual
  completo para exclusão de background, incluindo pontos externos ao município.

## Unidade amostral e máscara conjunta

Uma linha representa uma célula. A máscara conjunta exige simultaneamente:

1. Centro estritamente no interior do polígono municipal, excluindo centros
   exatamente sobre sua borda; não se usa o buffer de processamento dos rasters.
2. Valor finito nas sete features.
3. Ausência de máscara de invalidade do raster e de NoData declarado.

O código respeita tanto NoData `-9999` das features contínuas quanto NoData `0`
do MapBiomas. Não preenche lacunas nem converte NoData em zero válido.

## Células positivas e rastreabilidade

Cada geometria de ocorrência é convertida para linha/coluna pela transformação
inversa da grade, com a convenção de indexação do Rasterio. Coordenadas de
atributos como `source_x` e `source_y` não são utilizadas para essa conversão.
`landslide_id` deve ser inteiro, não nulo e único no inventário municipal.

Ocorrências na mesma célula geram uma única linha positiva:

- `target = 1`;
- `sample_type = "landslide"`;
- `landslide_count` contém a quantidade de ocorrências;
- `landslide_ids` contém uma lista JSON de inteiros ordenados, por exemplo
  `[123,456]`, serializada como texto no CSV e no GPKG.

Nenhuma ocorrência é descartada silenciosamente. Se seu pixel não pertencer à
máscara conjunta, se o ponto estiver fora da grade ou se o inventário municipal
contiver um ponto fora do município, a construção termina com erro. Uma
ocorrência dentro do município cujo centro de célula fique fora dele também
provoca erro, para revisão explícita da cobertura.

## Background / pseudoausências

Os candidatos partem da máscara conjunta, retirando todas as células positivas.
Calcula-se a distância euclidiana do centro de cada candidato ao ponto de
ocorrência mais próximo, usando a união do inventário municipal com o estadual.
O inventário estadual não é recortado no limite municipal.

Um candidato só permanece se sua distância for **maior ou igual** ao raio de
exclusão. Logo, distância exatamente igual a 60 m é permitida no default;
distância inferior a 60 m é excluída. O cálculo usa busca exata por árvore de
pontos, sem aproximar o círculo por um polígono de buffer.

O sorteio é uniforme, sem reposição, sobre candidatos ordenados por `cell_id`,
com `numpy.random.Generator(PCG64(seed))`. A saída é ordenada por classe
(positivas primeiro) e por `cell_id` dentro da classe. Quantidade solicitada
superior à disponibilidade provoca erro. Por default, a razão é 1:1.

Background recebe `target = 0`, `sample_type = "background"`,
`landslide_count = 0` e `landslide_ids = "[]"`. Estes rótulos não significam
ausência confirmada de deslizamento. A proporção amostral não estima a
prevalência territorial de deslizamentos. O raio de 60 m é um parâmetro inicial,
não uma distância validada empiricamente.

## Colunas e papéis

| Grupo | Colunas e significado |
|---|---|
| Identificadores | `cell_id = row * width + col`; `sample_id = sample_` seguido do `cell_id` com pelo menos seis dígitos |
| Posição | `row`, `col`, com índices iniciados em zero; `x`, `y`, coordenadas do centro em metros |
| Ocorrências | `landslide_count`, `landslide_ids` |
| Rótulos | `target`, `sample_type` |
| Features originais | `elevation`, `slope`, `aspect`, `plan_curvature`, `profile_curvature`, `land_cover`, `ndvi_pre_event` |
| Aspecto circular | `aspect_sin = sin(aspect * pi / 180)`; `aspect_cos = cos(aspect * pi / 180)` |

O valor bruto de `aspect` é preservado em graus, no intervalo [0, 360).
`aspect_sin` e `aspect_cos` pertencem a [-1, 1]. O MapBiomas mantém seu código
original inteiro em `land_cover`; ele é categórico, não ordinal. Não há one-hot
encoding, normalização, imputação ou ajuste estatístico nesta etapa.

A lista explícita de futuros preditores é: `elevation`, `slope`, `aspect_sin`,
`aspect_cos`, `plan_curvature`, `profile_curvature`, `land_cover` e
`ndvi_pre_event`. `aspect` bruto permanece disponível como feature de origem.
IDs, coordenadas, linha/coluna e rastreabilidade das ocorrências **não são
preditores**. Rótulos e contagens de ocorrências também não devem entrar em X.

## Produtos e metadata

Produtos futuros, sob a política atual de exclusão do Git para dados gerados:

- `data/processed/santa_tereza_ml_dataset.csv`, UTF-8, sem índice adicional;
- `data/processed/santa_tereza_ml_samples.gpkg`, camada `ml_samples`, pontos
  localizados nos centros das células;
- `data/processed/santa_tereza_ml_dataset_metadata.json`, UTF-8.

O metadata registra timestamp UTC, CRS, resolução, dimensões e transformação da
grade, contagens de ocorrências/células/classes/candidatos, raio, seed,
estratégia, nomes e caminhos das sete features, colunas e seus papéis, nomes de
saída e versões do ambiente. Registra também hashes SHA-256 das dez entradas.
Os caminhos das fontes são relativos ao repositório. Os inputs são conferidos
antes e depois da construção; mudança durante o processamento interrompe a
gravação.

Reprodutibilidade significa igualdade das amostras, atributos e contagens para
os mesmos dados e parâmetros. Timestamp de geração e bytes internos do GPKG
não precisam ser idênticos entre execuções. As versões utilizadas ficam
registradas; uma mudança de biblioteca deve ser acompanhada de nova auditoria.

## Auditoria e testes

O auditor verifica:

- Total de amostras, distribuição das classes e contagens do metadata.
- Unicidade de `cell_id` e `sample_id`, índices e centros da grade.
- Pertencimento à máscara conjunta e igualdade das sete features com os
  valores dos rasters, sem NoData ou valores não finitos.
- Rastreabilidade completa dos IDs e contagens do inventário municipal.
- Separação entre células positivas e background.
- Distâncias de background por cálculo independente de distâncias par a par,
  em blocos de memória, usando também as ocorrências estaduais.
- Limites e valores calculados dos componentes circulares do aspecto.
- Preservação dos códigos MapBiomas e intervalo físico do NDVI.
- Regeneração duas vezes com a mesma seed e comparação com a tabela salva.
- Integridade das fontes por hash, parâmetros, estratégia e grade do metadata.
- Camada, CRS, geometrias Point válidas, centros únicos e atributos do GPKG
  coincidentes com o CSV.

`tests/test_ml_dataset.py` usa funções centrais com matrizes e pontos sintéticos:
agregação, máscara, limites de distância, pontos externos, amostragem,
componentes do aspecto, alinhamento e detecção de corrupção. Inclui ainda um
round-trip com sete pequenos rasters e vetores exclusivamente em diretório
temporário do pytest. Não depende dos dados reais para esses testes.

## Referência da auditoria anterior e limitações

A auditoria anterior dos dados reais identificou 275 ocorrências em 270 células
positivas, grade de 408 × 570, 81.774 centros municipais e 80.931 células na
máscara conjunta. Com a configuração default, esperam-se 270 backgrounds e
540 linhas. Esses valores são referências de conferência, não resultados de
uma execução desta implementação, nem constantes usadas para forçar a saída.

As 843 células municipais excluídas pela cobertura de curvaturas têm slope
inferior a 0,1 grau. Portanto, o domínio do dataset é o subconjunto com todas
as features válidas, e não toda a área municipal. Não se imputam curvaturas
nessas células. As lacunas do NDVI identificadas anteriormente ficam fora do
município e não afetam essas contagens.

Esta etapa preserva a estrutura necessária para futura avaliação espacial,
mas não cria partições ou modelos. A distância de exclusão não garante
independência espacial entre amostras.
