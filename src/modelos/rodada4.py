"""Rodada 4: k_enc na seleção, grades amplas nas árvores e número de árvores do XGBoost (decisão de 08/11/2026).

Mudanças em relação às rodadas 2 e 3:
- k_enc entra na seleção do ridge, com a configuração de features (decisão de 03/11, ampliada em 08/11 para
  {0, 2, 5, 10, 20, 40} porque a 1ª execução escolheu o 10, o topo, em quase todas as dobras). Escolher k
  dentro do treino da dobra não vaza; o k = 5 fixo da rodada 2 só economizava tempo.
- Seleção estável como na rodada 2: para o teste T, deviance média das validações v de 2018 a T−1, cada uma
  prevista por um modelo ajustado em temporadas < v. Nenhuma escolha usa T (ver `_checar_sem_teste`).
- Poisson ridge: grade k × forma × frações (× formação) × penalidade.
- Árvores: k e configuração de features (forma, frações, formação) do ridge da mesma dobra, escolhidos só
  com o treino; a grade das árvores fica nos hiperparâmetros (buscar k também nas árvores multiplicaria o
  tempo por 6). RF: min_samples_leaf × max_features × max_depth. XGBoost: amostra fixa (semente 0) de
  max_depth, min_child_weight, subsample, colsample, reg_lambda, reg_alpha e taxa de aprendizado; o número
  de árvores é escolhido entre ARVORES_XGB pelas previsões parciais do mesmo ajuste (sem early stopping).
- Execução leve (`executar`): prioridade baixa; nº de processos lido de CONTROLE a cada tarefa (pode ser
  mudado durante a execução); espera, sem interromper, quando o job passa de 50% da RAM ou sobra pouca
  memória livre; checkpoint em CACHE, retomado se a execução parar.

Uso: python -m src.modelos.rodada4   (grava results/modelos/<ETAPA>_*.csv; ETAPA = rodada4c por padrão, TCC_ETAPA=rodada4d para a 4d)
"""
import itertools
import os
import pickle
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_poisson_deviance
from xgboost import XGBRegressor

from src.features.historico import carregar_canonica
from src.modelos.rodada2 import _ridge, configuracoes, validacoes
from src.modelos.tabela import ALVOS, calcular_base, enxuta, ler_tabela
from src.modelos.walkforward import ETAPA, SAIDA, TESTES, _sem_constantes, metricas

# 80 e 160 entraram na 4b (k = 40 na borda em 4 dobras dos escanteios); 320 e 1000 na 4c (160 na borda em
# 4 dobras). Com k = 1000 a média da temporada é praticamente a referência (temporada anterior).
KS = (0, 2, 5, 10, 20, 40, 80, 160, 320, 1000)
# Bordas da rodada 2 nos dois extremos (1e-4 nos gols, 300 nos escanteios): grade ampliada dos dois lados.
PENALIDADES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
# Ampliadas onde a 1ª execução bateu na borda (folha 30, max_features 0,5, profundidade 4 no RF;
# profundidade 6, min_child_weight 50, subsample 0,6, colsample 0,5 e 10 árvores no XGBoost).
GRADE_RF = [dict(min_samples_leaf=l, max_features=f, max_depth=d)
            for l, f, d in itertools.product((10, 20, 30, 50, 100, 200, 300), (0.1, 0.2, 0.33, 0.5, 0.7, 1.0),
                                             (1, 2, 3, 4, 6, None))]
# XGBoost sem ampliação na 4b: as bordas da taxa (0,1) e das 5 árvores são cobertas pelo outro eixo
# (taxa maior ≈ menos árvores com taxa 0,1, e o escolhido ficou longe de 1.500; poucas árvores ≈ taxa 0,01).
ARVORES_XGB = (5, 10, 25, 50, 100, 200, 400, 700, 1000, 1500)
_ESPACO_XGB = list(itertools.product((1, 2, 3, 4, 6, 8), (1, 5, 20, 50, 100, 200), (0.4, 0.6, 0.8, 1.0),
                                     (0.3, 0.5, 0.8, 1.0), (1.0, 10.0, 100.0), (0.0, 1.0, 10.0), (0.01, 0.03, 0.1)))


# 4d: a 4c ficou na borda da taxa (0,1), do reg_alpha (10) e do reg_lambda (1) em várias dobras. Os eixos
# ganham 0,3, 30 e 0,1, e as 240 novas combinações têm ao menos um desses valores.
_ESPACO_XGB_4D = [c for c in itertools.product((1, 2, 3, 4, 6, 8), (1, 5, 20, 50, 100, 200), (0.4, 0.6, 0.8, 1.0),
                                               (0.3, 0.5, 0.8, 1.0), (0.1, 1.0, 10.0, 100.0), (0.0, 1.0, 10.0, 30.0),
                                               (0.01, 0.03, 0.1, 0.3))
                  if c[4] == 0.1 or c[5] == 30.0 or c[6] == 0.3]


def _amostra_xgb():
    """120 combinações da semente 0 (rodadas 4 e 4b) + 120 novas da semente 1 (4c), sem repetir,
    + 240 da semente 2 na região ampliada (4d)."""
    a = random.Random(0).sample(_ESPACO_XGB, 120)
    usadas = set(a)
    resto = [c for c in _ESPACO_XGB if c not in usadas]
    return a + random.Random(1).sample(resto, 120) + random.Random(2).sample(_ESPACO_XGB_4D, 240)


N_XGB = {"rodada4c": 240, "rodada4d": 480}[ETAPA]  # a 4c usa as 240 primeiras (sementes 0 e 1); a 4d, as 480
GRADE_XGB = [dict(zip(("max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda",
                       "reg_alpha", "learning_rate"), c)) for c in _amostra_xgb()[:N_XGB]]
GRADES = {"random_forest": GRADE_RF, "xgboost": GRADE_XGB}
CFG = ("forma", "f_pi", "f_gap", "formacao")

CACHE = SAIDA / "rodada4_cache"
CONTROLE = CACHE / "processos.txt"  # nº de processos; editar durante a execução
MEM_JOB_MAX = 0.5                   # fração da RAM total que o job (com os filhos) pode usar
MEM_LIVRE_MIN = 1.5 * 2 ** 30       # memória livre mínima no sistema para mandar tarefa nova


# ---------- execução leve ----------

def _baixa_prioridade():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)


def processos() -> int:
    try:
        return max(1, int(CONTROLE.read_text().strip()))
    except (OSError, ValueError):
        return 2


def _memoria_ok() -> bool:
    vm, eu = psutil.virtual_memory(), psutil.Process()
    rss = 0
    for p in [eu, *eu.children(recursive=True)]:
        try:
            rss += p.memory_info().rss
        except psutil.Error:
            pass
    try:  # ajustável durante a execução, como processos.txt (09/11: 1,5 GB travou a 4c com outros programas abertos)
        livre_min = float(CONTROLE.with_name("memoria_livre_gb.txt").read_text().strip()) * 2 ** 30
    except (OSError, ValueError):
        livre_min = MEM_LIVRE_MIN
    return vm.available > livre_min and rss < MEM_JOB_MAX * vm.total


def _gravar(arq, feitos):
    tmp = arq.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(feitos))
    os.replace(tmp, arq)


def executar(tarefas: dict, nome: str) -> dict:
    """tarefas {chave: () -> (função, args)}, montadas só na hora de enviar (poupa memória).
    Resultados em CACHE/<nome>.pkl a cada 10 tarefas; na retomada, as já feitas são puladas."""
    CACHE.mkdir(parents=True, exist_ok=True)
    arq = CACHE / f"{nome}.pkl"
    feitos = pickle.loads(arq.read_bytes()) if arq.exists() else {}
    pendentes = [c for c in tarefas if c not in feitos][::-1]
    print(f"{nome}: {len(tarefas) - len(pendentes)} de {len(tarefas)} já feitas", flush=True)
    rodando, novos, inicio = {}, 0, time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_baixa_prioridade) as ex:
        while pendentes or rodando:
            while pendentes and len(rodando) < processos() and _memoria_ok():
                c = pendentes.pop()
                f, args = tarefas[c]()
                rodando[ex.submit(f, *args)] = c
            if not rodando:
                time.sleep(10)  # memória curta: espera, não interrompe
                continue
            prontos, _ = wait(rodando, timeout=10, return_when=FIRST_COMPLETED)
            for fut in prontos:
                feitos[rodando.pop(fut)] = fut.result()
                novos += 1
                if novos % 10 == 0:
                    _gravar(arq, feitos)
                    print(f"{nome}: {len(feitos)}/{len(tarefas)} ({processos()} processos, "
                          f"{(time.time() - inicio) / 60:.0f} min)", flush=True)
    _gravar(arq, feitos)
    return feitos


# ---------- modelos ----------

def modelo_arvore(nome: str, hp: dict, n_jobs: int = 1):
    if nome == "random_forest":
        return RandomForestRegressor(n_estimators=300, criterion="poisson", n_jobs=n_jobs, random_state=0, **hp)
    hp = {k: v for k, v in hp.items() if k != "n_arvores"}
    return XGBRegressor(objective="count:poisson", n_estimators=max(ARVORES_XGB), n_jobs=n_jobs, random_state=0, **hp)


def prever_arvore(nome: str, m, X, n_arvores=None) -> np.ndarray:
    p = m.predict(X) if nome == "random_forest" else m.predict(X, iteration_range=(0, int(n_arvores)))
    return np.clip(p, 1e-6, None)


def tabela(bases, alvo, k, cfg, nome="poisson_ridge"):
    """(X, linhas sem nulo) do modelo; o XGBoost recebe as colunas com NULL, mas usa as mesmas linhas."""
    ok = enxuta(bases[k], alvo, **cfg).notna().all(axis=1)
    X = enxuta(bases[k], alvo, **cfg, com_nulos=(nome == "xgboost"))
    return (X if nome == "xgboost" else X.astype(float)), ok


def _checar_sem_teste(temp, linhas, limite):
    assert temp[linhas].max() < limite, "linha de temporada proibida no ajuste"


def _notas_ridge(X, ok, y, temp, vs):
    out = {}
    for v in vs:
        aj, va = (temp < v) & ok, temp == v
        _checar_sem_teste(temp, aj, v)
        cols = _sem_constantes(X, aj)
        for pen in PENALIDADES:
            out[(pen, v)] = mean_poisson_deviance(y[va], _ridge(X.loc[aj, cols], y[aj], pen)(X.loc[va, cols]))
    return out


def _notas_arvore(nome, hp, X, ok, y, temp, vs):
    out = {}
    for v in vs:
        aj, va = (temp < v) & ok, temp == v
        _checar_sem_teste(temp, aj, v)
        m = modelo_arvore(nome, hp).fit(X[aj], y[aj])
        if nome == "random_forest":
            out[(None, v)] = mean_poisson_deviance(y[va], prever_arvore(nome, m, X[va]))
        else:
            for n in ARVORES_XGB:
                out[(n, v)] = mean_poisson_deviance(y[va], prever_arvore(nome, m, X[va], n))
    return out


def ajustar_final(nome, escolha: dict, bases, jogos, alvo):
    """Reajusta no treino inteiro (< teste) o modelo escolhido; devolve (prever(Z) -> μ, X, linhas de teste, modelo)."""
    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    temp, t = jogos["temporada"], int(escolha["teste"])
    cfg = {c: escolha[c] for c in CFG}
    cfg["f_pi"], cfg["f_gap"] = int(cfg["f_pi"]), int(cfg["f_gap"])
    X, ok = tabela(bases, alvo, int(escolha["k"]), cfg, nome)
    tr, te = (temp < t) & ok, temp == t
    _checar_sem_teste(temp, tr, t)
    assert ok[te].all(), f"feature nula no teste {t}"
    if nome == "poisson_ridge":
        cols = _sem_constantes(X, tr)
        prever = _ridge(X.loc[tr, cols], y[tr], float(escolha["penalidade"]))
        return (lambda Z: prever(Z[cols])), X[cols], te, None
    hp = {h: escolha[h] for h in GRADES[nome][0]}
    hp = {h: (None if pd.isna(v) else (int(v) if h in ("min_samples_leaf", "max_depth") else v)) for h, v in hp.items()}
    m = modelo_arvore(nome, hp, n_jobs=processos()).fit(X[tr], y[tr])
    n = escolha.get("n_arvores")
    return (lambda Z: prever_arvore(nome, m, Z, n)), X, te, m


def _melhor(notas: dict, t: int):
    """Chave com a menor deviance média nas validações de t; notas: {chave: {v: deviance}}."""
    medias = {c: np.mean([d[v] for v in validacoes(t)]) for c, d in notas.items()}
    return min(medias.items(), key=lambda kv: kv[1])


def avaliar(jogos, bases, alvo, testes=TESTES):
    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    temp = jogos["temporada"]
    vs = validacoes(max(testes))
    assert max(vs) < max(testes)
    # 1) Ridge: k × configuração × penalidade, em todas as validações.
    tarefas = {(k, *c.values()): (lambda k=k, c=c: (_notas_ridge, (*tabela(bases, alvo, k, c), y, temp, vs)))
               for k in KS for c in configuracoes(alvo)}
    notas_r, validacao = {}, []
    for chave, r in executar(tarefas, f"{alvo}_ridge").items():
        for (pen, v), d in r.items():
            notas_r.setdefault((*chave, pen), {})[v] = d
            validacao.append({"alvo": alvo, "modelo": "poisson_ridge", "k": chave[0], **dict(zip(CFG, chave[1:])),
                              "penalidade": pen, "v": v, "deviance": d})
    escolhas = []
    for t in testes:
        chave, nota = _melhor(notas_r, t)
        escolhas.append({"alvo": alvo, "teste": t, "modelo": "poisson_ridge", "k": chave[0],
                         **dict(zip(CFG, chave[1:5])), "penalidade": chave[5], "deviance_validacao": nota})
    print(alvo, "ridge:", pd.DataFrame(escolhas)[["teste", "k", *CFG, "penalidade"]].to_string(index=False), flush=True)
    # 2) Árvores: k e configuração do ridge da dobra; hiperparâmetros só nas validações que cada dobra usa.
    ultimo_teste = {}
    for e in escolhas:
        kc = (e["k"], *(e[c] for c in CFG))
        ultimo_teste[kc] = max(ultimo_teste.get(kc, 0), e["teste"])
    for nome, grade in GRADES.items():
        tarefas = {(kc, tuple(hp.items())): (lambda kc=kc, hp=hp: (
            _notas_arvore, (nome, hp, *tabela(bases, alvo, kc[0], dict(zip(CFG, kc[1:])), nome), y, temp,
                            validacoes(ultimo_teste[kc]))))
                   for kc in ultimo_teste for hp in grade}
        notas = {}
        for (kc, hp), r in executar(tarefas, f"{alvo}_{nome}").items():
            for (n, v), d in r.items():
                notas.setdefault((kc, hp, n), {})[v] = d
                validacao.append({"alvo": alvo, "modelo": nome, "k": kc[0], **dict(zip(CFG, kc[1:])), **dict(hp),
                                  "n_arvores": n, "v": v, "deviance": d})
        for t in testes:
            e_r = next(e for e in escolhas if e["teste"] == t and e["modelo"] == "poisson_ridge")
            kc = (e_r["k"], *(e_r[c] for c in CFG))
            (_, hp, n), nota = _melhor({c: d for c, d in notas.items() if c[0] == kc}, t)
            escolhas.append({"alvo": alvo, "teste": t, "modelo": nome, "k": kc[0], **dict(zip(CFG, kc[1:])),
                             **dict(hp), "n_arvores": n, "deviance_validacao": nota})
        print(alvo, nome, "escolhas:", pd.DataFrame([e for e in escolhas if e["modelo"] == nome]).drop(
            columns=["alvo", "modelo"]).to_string(index=False), flush=True)
    # 3) Teste: reajuste no treino inteiro de cada dobra.
    linhas, previsoes = [], []
    for t in testes:
        te = temp == t
        p_t = {"ingenua": np.full(int(te.sum()), y[temp < t].mean())}
        for e in (e for e in escolhas if e["teste"] == t):
            prever, X, te_, _ = ajustar_final(e["modelo"], e, bases, jogos, alvo)
            p_t[e["modelo"]] = prever(X[te_])
        for nome, p in p_t.items():
            linhas.append({"alvo": alvo, "teste": t, "modelo": nome, **metricas(y[te], p)})
        previsoes.append(pd.DataFrame({"alvo": alvo, "teste": t, "rodada": jogos.loc[te, "rodada"].to_numpy(),
                                       "y": y[te].to_numpy(), **p_t}))
    return pd.DataFrame(linhas), pd.DataFrame(escolhas), pd.concat(previsoes), pd.DataFrame(validacao)


def no_limite(escolhas: pd.DataFrame) -> list[str]:
    """Hiperparâmetros escolhidos na borda da grade (pede ampliar a grade de novo)."""
    def bordas(valores):
        return (min(valores), max(valores))
    por_modelo = {"poisson_ridge": {"penalidade": bordas(PENALIDADES), "k": bordas(KS)},
                  # max_depth None (sem limite) fica acima de 6, então só o menor é borda.
                  "random_forest": {"min_samples_leaf": bordas({g["min_samples_leaf"] for g in GRADE_RF}),
                                    "max_features": bordas({g["max_features"] for g in GRADE_RF}),
                                    "max_depth": (min(g["max_depth"] for g in GRADE_RF if g["max_depth"]),)},
                  "xgboost": {**{h: bordas({g[h] for g in GRADE_XGB}) for h in GRADE_XGB[0]},
                              "n_arvores": bordas(ARVORES_XGB)}}
    avisos = []
    for _, e in escolhas.iterrows():
        for h, borda in por_modelo.get(e["modelo"], {}).items():
            if pd.notna(e.get(h)) and e[h] in borda:
                avisos.append(f"{e['alvo']} {e['teste']} {e['modelo']}: {h} = {e[h]} no limite")
    return avisos


def carregar_bases():
    jogos = carregar_canonica()
    copa, depara = ler_tabela("norm_copa_brasil"), ler_tabela("tb_tecnicos_depara")
    return jogos, {k: calcular_base(jogos, copa, depara, k) for k in KS}


def main():
    _baixa_prioridade()
    jogos, bases = carregar_bases()
    SAIDA.mkdir(parents=True, exist_ok=True)
    partes = [avaliar(jogos, bases, alvo) for alvo in ALVOS]
    for i, nome in enumerate(("metricas", "escolhas", "previsoes", "validacao")):
        pd.concat(p[i] for p in partes).to_csv(SAIDA / f"{ETAPA}_{nome}.csv", index=False)
    res = pd.concat(p[0] for p in partes)
    print(res.groupby(["alvo", "modelo"])[["mae", "rmse", "deviance"]].mean().round(4).to_string())
    print("\n".join(no_limite(pd.concat(p[1] for p in partes))) or "nenhuma escolha no limite")


if __name__ == "__main__":
    main()
