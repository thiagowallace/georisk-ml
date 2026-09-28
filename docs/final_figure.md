# Figura final de apresentação

`scripts/make_final_figure.py` gera uma peça horizontal de **2400 × 1350 px**
com Matplotlib e Pillow, para README, apresentações e portfólio.

## Uso

Na raiz do repositório, usando o ambiente do projeto:

```powershell
& .\.venv\Scripts\python.exe -B scripts/make_final_figure.py
```

Os caminhos são resolvidos pela localização do script, independentemente do
diretório de execução. Não há dependência de caminhos absolutos ou rede.

Saída única: `outputs/figures/georisk_ml_final_figure.png`.
A pasta é criada automaticamente. Reexecuções substituem somente essa figura.
Nenhum dado, modelo, métrica ou artefato das Etapas 1–5 é alterado.

## Entradas e valores

| Entrada | Uso |
|---|---|
| `outputs/models/final/susceptibility_summary.json` | Células válidas e resolução |
| `outputs/models/final/model_surface_comparison.csv` | Conferência da contagem e concordância com o resumo |
| `outputs/models/final/map_logistic.png` | Mapa principal e cores da legenda original |
| `outputs/models/spatial_validation/summary_metrics.json` | ROC-AUC médio dos folds de LR e RF; número de folds |
| `outputs/models/final/test_results.log` | Contagem registrada de testes aprovados |

As três primeiras entradas são obrigatórias. Na execução verificada, foram
lidos automaticamente **80.931 células**, **30 m**, ROC-AUC LR
**0,8600034295772631**, ROC-AUC RF **0,8475566948632185**, **5 folds** e
**130 testes aprovados**. A figura arredonda os ROC-AUC para três casas.
A contagem de testes se refere ao log da Etapa 5; gerar a figura não executa
a suíte de testes.

ROC-AUC vem da validação espacial da Etapa 4. A correlação entre superfícies
da Etapa 5 é somente conferida e nunca usada como substituta dessa métrica.
O mapa continua representando `landslide susceptibility score`.

## Fallbacks e validação

Se o JSON opcional da validação espacial estiver ausente, usam-se os valores
explicitamente solicitados: LR **0,860** e RF **0,848**. Se o log opcional
estiver ausente, usa-se a contagem solicitada de **130 testes**. Esses
fallbacks ficam documentados no código e são informados no recibo JSON no
terminal. **Nenhum fallback foi necessário na execução verificada.**

Arquivo obrigatório ausente, contagens divergentes, resolução inválida,
ROC-AUC fora de [0,1] ou log de testes com falhas interrompem a execução
com mensagem de erro. Evidência opcional existente, porém inválida, não é
substituída silenciosamente por fallback.

## Composição e reprodução

O script recorta margens, eixos e título do PNG original, preservando os
pixels internos do mapa e sua proporção. O recorte localiza pixels com cor
nos 84% esquerdos da imagem e adiciona margem de 12 pixels. A legenda
horizontal reutiliza as cores da barra original à direita. Essa regra é
específica ao layout do mapa produzido pela Etapa 5; erros de localização
interrompem a geração.

A renderização usa fonte DejaVu Sans fornecida pelo Matplotlib, dimensões,
cores e metadados fixos, sem datas ou aleatoriedade. Insumos e versões das
bibliotecas iguais produzem o mesmo PNG. O recibo informa entradas, valores,
fallbacks, verificação de preservação das entradas e SHA-256 da saída.
