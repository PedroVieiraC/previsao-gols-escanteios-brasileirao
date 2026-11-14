# Fluxo de trabalho com LLM

O código, os resultados e o artigo deste repositório foram produzidos pelo autor com o Claude Code (CLI da Anthropic) como assistente de programação e de redação. O branch `autoral` tem só o projeto; este branch (`main`) acrescenta esta pasta com a descrição do método de trabalho. O que o autor decidiu e o que delegou está na última seção.

## Instruções de projeto
Um `CLAUDE.md` na raiz do projeto privado era lido em toda sessão (versão limpa em `CLAUDE.md`, nesta pasta). Ele fixa as regras que o assistente não pode violar:
- **Vazamento:** nada da partida-alvo entra como variável, só histórico de jogos anteriores com `shift(1)`, ordenado por data e hora (e não por rodada: 75 jogos adiados ficam fora da ordem da rodada).
- **Validação cronológica:** walk-forward, nunca split aleatório como avaliação principal.
- **Ausência não é zero:** sem imputação sem regra documentada.
- **SQL manual:** o autor roda o SQL no pgAdmin para conhecer a base; o assistente só revisa e propõe `SELECT`, salvo pedido explícito (e DROP só com aviso e backup).
- **Decisões:** itens marcados `[PROPOSTA]` ou `[A DECIDIR]` nunca são implementados. As decisões vigentes ficavam em `docs/decisoes.md` (datadas) e o estado em `docs/status.md`, ambos importados pelo `CLAUDE.md`.

## Skills próprias
Em `skills/` há duas cópias de exemplo. O projeto tinha seis:
- `tcc-artigo-redacao`: lê um guia ABNT para artigo antes de qualquer redação, proíbe inventar números, citações e DOI, e checa chaves bibliográficas contra o `.bib`.
- `fluxo_tcc`: confere a tabela de próximos passos da entrega contra o estado real (git, seções vazias, marcadores `\pendente`) e separa o que cabe ao assistente do que cabe ao autor.
- `relatorio-diario`: um relatório por dia, com decisões e números, como fonte para o artigo.
- `commit-checkpoint`: commit local com mensagem curta em português ao fim de uma etapa.
- `project-organization-auditor`: auditoria somente de leitura da organização do repositório, disparada por um hook de pós-commit a cada 10 commits.
- `trabalho-paralelo`: planeja sessões paralelas (ver abaixo).

## Subagentes, sessões paralelas e worktrees
Tarefas grandes e independentes (análise exploratória, redação da Metodologia, de-para dos técnicos) rodaram em sessões paralelas. Na primeira vez, três sessões no mesmo diretório e no mesmo branch se sobrescreveram (relatório do dia, status, banco alterado sem relação com commit). Daí a skill `trabalho-paralelo`: um branch e um `git worktree` por sessão, backup do git e do banco antes, um arquivo de handoff por sessão e um prompt de merge para depois. Uma sessão de auditoria revisou os commits paralelos antes do merge (nenhum revert). Subagentes (analista, programador, testador) também foram usados para planejar, implementar e verificar dentro de uma sessão.

## Grafo de conhecimento do código
O código e os documentos foram indexados em um grafo (graphify) com comunidades e nós centrais. As instruções de projeto mandavam consultar o grafo (`query`, `path`, `explain`) antes de buscar em arquivos brutos e atualizá-lo depois de mudar código, para gastar menos contexto.

## Revisão crítica por uma sessão independente
Em 14/11/2026 uma sessão que não participou do trabalho revisou, só em leitura, o código, os testes e o texto, e escreveu um relatório com oito achados. Ela conferiu a suíte de testes, os números do artigo contra os CSVs e fez um teste de vazamento na base real: perturbou os dados dos jogos a partir de um instante T e as 265 colunas de variáveis dos jogos anteriores ficaram idênticas (nenhum vazamento). O que a revisão mudou:
- **Os escanteios deixaram de ser significativos contra a média corrente da liga.** A "média da liga" era uma constante por dobra; uma referência que acompanha o nível da liga ao longo da temporada (média corrente encolhida para a da temporada anterior) absorve cerca de metade do ganho. Média RF+XGB: -0,0061 [-0,0157; 0,0031]. Nos gols o ganho resiste (Random Forest -0,0187 [-0,0314; -0,0067]).
- A significância dos escanteios variava entre execuções e modelos, e a média RF+XGB foi fixada depois de vistos resultados de teste: o texto passou a dizer isso e a restringir "ambos com significância" aos gols.
- O código, que gerava a 4d em vez da 4c relatada, passou a reproduzir a 4c por padrão (`ETAPA`), e a base canônica passou a ser publicada.

Resumo, Metodologia, Resultados, Discussão e Conclusão foram revistos por causa disso.

## O que o autor decidiu e o que delegou
Decidiu o autor: o tema e os dois alvos (total de gols e de escanteios, em contagem); o período (2017–2024, só a Série A); a hierarquia entre as fontes de dados; as regras de conflito entre fontes (aprovadas por escrito); as famílias de formação tática; o desenho do walk-forward (teste de 2019 a 2024); quais famílias de variáveis estudar (médias com decaimento, pi-ratings, GAP ratings); a 4c como rodada final (a 4d foi só sensibilidade); a publicação da base; o merge no `master`. Conferências manuais de valores (por exemplo, os escanteios de um jogo) também foram do autor.

Delegou ao assistente: a escrita e a execução do código das variáveis, dos modelos, dos testes de vazamento e das rodadas; a ampliação de grades quando a escolha caía na borda (registrada em `docs/decisoes_autonomas.md`, com a ressalva da crítica sobre seleção por teste); a combinação das previsões; a primeira redação de Resumo, Introdução, Resultados, Discussão e Conclusão, que o autor revisa; e rascunhos de relatórios e auditorias.

Limite conhecido: o assistente pode errar fatos e números; por isso cada número do artigo foi conferido contra os CSVs e os testes precisam passar antes de cada commit.
