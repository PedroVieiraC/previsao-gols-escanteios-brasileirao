# Previsão pré-jogo de gols e escanteios no Brasileirão

Quanto do total de gols e do total de escanteios de uma partida da Série A (2017–2024) dá para prever com informação disponível antes do jogo, e o quanto isso supera referências ingênuas? Trabalho de conclusão de curso; o artigo em LaTeX está em `artigo/`.

## Dados
3.040 partidas da Série A de 2017 a 2024 (380 por temporada), integradas pelo autor a partir de bases públicas de partidas e de estatísticas. Os CSVs estão em `data/` e o esquema em `data/README.md`.

## Método
- **Variáveis pré-jogo, sem vazamento.** Placar, escanteios, chutes e posse da partida-alvo nunca entram; só o histórico de jogos anteriores, ordenado por data e hora. Médias da temporada, janelas de 5 e 10 jogos e médias com decaimento exponencial, todas encolhidas para uma referência (temporada anterior ou média dos promovidos); pi-ratings; GAP ratings; dias de descanso (Série A e Copa do Brasil); jogos sem público; trocas de técnico; formação anunciada (só para escanteios). Cada módulo de `src/features/` tem teste de vazamento.
- **Validação.** Walk-forward por temporada: teste de 2019 a 2024, treino em todas as temporadas anteriores (2017 e 2018 só no treino). Hiperparâmetros e configuração de variáveis escolhidos só com as temporadas anteriores a cada teste.
- **Modelos.** Média da liga (referência), Poisson ridge, Random Forest e XGBoost; binomial negativa quando há superdispersão.
- **Métricas.** Deviance de Poisson (principal), MAE, RMSE; log loss e Brier para P(gols > 2,5) e P(escanteios > 9,5).
- **Inferência.** Bootstrap pareado por blocos de rodada (temporada, rodada) da diferença de deviance contra a referência.

## Resultados principais (rodada 4c)
Deviance média nas 2.280 partidas de teste; entre colchetes, IC95% da diferença contra a referência.

| alvo | média da liga | melhor modelo | contra a média da liga | contra a média corrente da liga |
|---|---|---|---|---|
| gols | 1,0759 | Random Forest, 1,0574 | -0,0185 [-0,0315; -0,0061] | -0,0187 [-0,0314; -0,0067] |
| escanteios | 1,1838 | média RF+XGB, 1,1731 | -0,0106 [-0,0184; -0,0034] | -0,0061 [-0,0157; 0,0031] |

- **Gols:** o ganho do Random Forest e do XGBoost resiste a referências mais fortes (média da temporada anterior e média corrente da liga).
- **Escanteios:** o ganho é significativo contra a média da liga e contra a da temporada anterior, mas deixa de ser significativo contra a média corrente da liga.
- **Probabilidades:** P(gols > 2,5) supera a frequência de treino com RF e XGBoost; P(escanteios > 9,5) não é superada por nenhum modelo.
- O ganho é pequeno nos dois alvos: a maior parte da variação dos totais é ruído do jogo.

Tabelas completas em `results/modelos/` (`rodada4c_conjunto_comparacao.csv`, `rodada4c_referencias.csv`, `rodada4c_metricas.csv`).

## Estrutura
```
artigo/          artigo em LaTeX (main.tex, src/, figures/, referencias.bib)
data/            base canônica, Copa do Brasil e de-para de técnicos (CSV)
src/features/    variáveis pré-jogo
src/modelos/     tabela de modelagem, walk-forward, rodadas 2 a 4, análises e figuras
results/modelos/ resultados de cada rodada (CSV)
tests/           testes unitários, inclusive de vazamento
```

## Reproduzir
Python 3.13.
```
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.features.historico        # confere a leitura dos dados e uma feature
python -m src.modelos.tabela            # monta a tabela de modelagem (265 colunas de base)
python -m src.modelos.walkforward       # rodada 1: média da liga, Poisson e binomial negativa (~10 min)
python -m src.modelos.rodada2           # rodada 2: seleção estável e Poisson ridge (~10 min)
python -m src.modelos.rodada3           # rodada 3: Random Forest e XGBoost
python -m src.modelos.rodada4           # rodada 4c (final): grades amplas; leva horas
python -m src.modelos.conjunto          # combinação das previsões (alguns minutos)
python -m src.modelos.analise           # bootstrap por blocos, permutação, SHAP, probabilidades (alguns minutos)
python -m src.modelos.referencias       # referências mais fortes (temporada anterior e corrente)
python -m src.modelos.figuras           # figuras do artigo em artigo/figures/
```
A rodada 4 leva horas (centenas de combinações por dobra, em paralelo) e retoma de um cache em `results/modelos/rodada4_cache/` se for interrompida. Os resultados da 4c já estão em `results/modelos/`; rodar essa etapa de novo sobrescreve os CSVs. `ETAPA` (`src/modelos/walkforward.py`) vale `rodada4c` por padrão; `TCC_ETAPA=rodada4d` roda a análise de sensibilidade. Os CSVs de validação, grandes, não estão versionados.
