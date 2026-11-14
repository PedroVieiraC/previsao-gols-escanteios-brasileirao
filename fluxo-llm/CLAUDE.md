# TCC: predição de gols e escanteios no Brasileirão (artigo)

Versão limpa das instruções de projeto que o Claude Code lia em toda sessão. As decisões vigentes e o status viviam em arquivos à parte (`docs/decisoes.md`, `docs/status.md`), importados por este.

## Metodologia
- Previsão pré-jogo. Placar, escanteios, chutes, posse e público da partida-alvo são alvo ou pós-jogo; só entram como histórico de jogos anteriores, com `shift(1)`.
- Validação cronológica (TimeSeriesSplit ou walk-forward). Nunca split aleatório como avaliação principal.
- Modelos: Poisson (baseline), Random Forest, XGBoost. Métricas: MAE, RMSE, Poisson deviance.
- Ausência não é zero. Não imputar sem regra documentada.

## SQL
- O autor executa o SQL manualmente para conhecer a base. Por padrão, não execute nem gere DDL/DML (CREATE, DROP, DELETE, UPDATE, ALTER, INSERT). Revisar scripts e propor consultas SELECT é permitido.
- Exceção: quando o autor pedir explicitamente, gere e execute DDL/DML. Salve o script em `sql/` antes de executar e liste os objetos afetados antes e depois.
- DROP só quando necessário, com aviso ao autor, descrição do que será removido e backup antes.
- Ao revisar `sql/`, aponte bugs e riscos; o autor corrige.

## Comandos
- Testes: `python -m unittest discover -s tests -v`
- Nunca ler nem exibir `.env`.

## Ignorar em buscas
- `.venv/`, `.git/`, `__pycache__/` e caches de dados brutos.

## Comportamento
- Para mudança complexa ou análise de dados, proponha o plano e os riscos (vazamento) antes de editar. Itens marcados [PROPOSTA] ou [A DECIDIR] não são decisões: não os implemente.
- Sem recapitulação do pedido e sem cordialidades.
- Ao redigir ou revisar texto do artigo, use a skill `tcc-artigo-redacao`.
- Ao fim de uma sessão que mude o estado do projeto, atualize o arquivo de status.
- Ao concluir uma mudança significativa, registre-a no relatório do dia (skill `relatorio-diario`) e comite (skill `commit-checkpoint`).
- Ao comitar, faça também o push do branch atual; nunca force push.

## Grafo de conhecimento do código
Havia um grafo do projeto (ferramenta graphify). Regra: quando o grafo existe, consultar com `query`, `path` e `explain` antes de buscar em arquivos brutos, e atualizar o grafo depois de mudar código. Passar essa regra a todo subagente que explore código.
