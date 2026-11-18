# Dados

Base de partidas da Série A do Campeonato Brasileiro, 2017 a 2024, integrada pelo autor a partir de bases públicas de partidas e de estatísticas. Os arquivos são lidos por `ler_tabela` (`src/features/historico.py`).

## `tb_main_brasileirao_tcc.csv`
Base canônica: 3.040 partidas (380 por temporada), uma linha por partida; chave única `(data, mandante, visitante)`.

| coluna | descrição |
|---|---|
| `temporada`, `rodada`, `data`, `hora` | identificação e horário de início |
| `mandante`, `visitante`, `estado_m`, `estado_v` | times (nome padronizado) e UF |
| `tecnico_m`, `tecnico_v` | técnico de cada time na partida |
| `formacao_m`, `formacao_v` | formação inicial |
| `gols_*`, `escanteios_*` | alvos do estudo (total = mandante + visitante) |
| `chutes_*`, `chutes_no_alvo_*`, `posse_*`, `faltas_*`, `amarelos_*` | estatísticas da partida; só entram como histórico de partidas anteriores |
| `estatisticas_disponiveis` | se a partida tem estatísticas |
| `fonte_gols`, `fonte_escanteios` | origem do valor: `ADAODUQUE` (base principal), `KAGGLE_1423`, `MANUAL` ou `COMPLEMENTAR` (outra fonte; ver o artigo) |
| `vencedor` | derivado do placar; nunca é variável preditora |

Valores ausentes ficam vazios e não são imputados na base.

## Arquivos locais não versionados
`norm_copa_brasil.csv` (jogos da Copa do Brasil, usados nos dias de descanso) e `tb_tecnicos_depara.csv` (de-para de técnicos) não são versionados neste repositório. São lidos por `src/features/calendario.py`, `copa.py` e `tecnicos.py` e pelos scripts de `src/modelos/`; sem eles, as variáveis de calendário/Copa e de técnicos não podem ser recalculadas.
