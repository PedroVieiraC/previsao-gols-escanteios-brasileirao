---
name: fluxo_tcc
description: Mostra e atualiza a tabela dos próximos passos para entregar o TCC (docs/fluxo_tcc.md) - o que falta, quem faz (Claude ou autor), de que depende e o próximo passo. Use quando o usuário digitar /fluxo_tcc ou perguntar "o que falta para o TCC", "próximos passos", "como estamos", "o que falta entregar", "o que eu preciso fazer", e ao fim de uma sessão que conclua um passo da tabela.
---

# Fluxo do TCC

A tabela fica em `docs/fluxo_tcc.md` (versionada). Esta skill confere cada linha contra o estado real do projeto, atualiza a tabela e a mostra ao autor.

## Passos
1. Leia `docs/fluxo_tcc.md`, `docs/status.md` e o fim de `docs/decisoes_autonomas.md`.
2. Confira o estado real, só com leitura:
   - `git log --oneline -10` e `git branch` (merges feitos, branches `par/...` ainda abertos);
   - `docs/handoff/*.md` nos worktrees (`git worktree list`);
   - texto do artigo em `textoTCC/principal/LATEX/src/`:
     - seções vazias (tamanho 0);
     - `\pendente`;
     - marcas `% NÚMEROS DA RODADA`;
     - verbos no futuro na Metodologia e na Introdução;
   - `logs/` das rodadas, se alguma estiver rodando.
3. Atualize as linhas:
   - situação ✅, 🔄, ⏳ ou ⬜;
   - o próximo passo concreto de cada uma.

   Passos novos que aparecerem entram na tabela. Os concluídos vão para "Feito recentemente" (no máximo ~8 linhas; as mais antigas saem). Troque a data e a hora do cabeçalho (`Get-Date -Format "dd/MM/yyyy HH:mm"`).
4. Mostre ao autor:
   - **Bloqueios seus:** primeiro, só as linhas com Quem = **Autor**, cada uma com a pergunta exata a responder;
   - **Próximos passos:** depois, a tabela de passos ativos, sem a seção de feitos;
   - **Fazer agora:** no fim, uma linha com o que o Claude pode fazer em seguida sem esperar ninguém.
5. Não execute os passos por conta própria. A skill só mostra e atualiza a tabela. Execute um passo só se o autor pedir, ou se o pedido da conversa já o incluir.
6. Se a tabela mudou, comite só `docs/fluxo_tcc.md` (skill `commit-checkpoint`, `git add` pelo nome), a não ser que haja outro commit da mesma tarefa logo em seguida.

## Regras
- Use só fatos conferidos no passo 2. Se uma linha não pôde ser conferida, escreva "não verificado".
- Itens que dependem de regra do CEFET-MG, do orientador ou do autor (prazo, template, merge no `master`) nunca viram ✅ sem a confirmação dele.
