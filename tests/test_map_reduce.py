"""
tests/test_map_reduce.py
Motore map-reduce (stadio 2): map con sentinella NIENTE + reduce gerarchico.
"""
from types import SimpleNamespace

import pytest

from modules.map_reduce import MapReduceEngine, MapReduceResult


class _FakeLLM:
    """
    LLM finto: distingue map e reduce dal system prompt e risponde con un testo
    pilotato dal `responder` (callable(system, user) -> str). Registra le chiamate.
    """
    def __init__(self, responder):
        self._responder = responder
        self.map_calls = 0
        self.reduce_calls = 0

    async def chat(self, messages, role, *, system=None, options=None, model=None):
        user = messages[0].content
        if system and system.startswith("Fondi"):
            self.reduce_calls += 1
        else:
            self.map_calls += 1
        return SimpleNamespace(content=self._responder(system, user))


def _blocks(n):
    return [
        {"text": f"testo blocco {i}", "page_start": i, "page_end": i, "n_chunks": 1}
        for i in range(1, n + 1)
    ]


def _is_map(system):
    return bool(system) and system.startswith("Analizzi")


class TestMap:
    async def test_scarta_blocchi_niente(self):
        # Il blocco 2 non e' rilevante: il map risponde NIENTE -> scartato.
        def responder(system, user):
            if _is_map(system):
                return "NIENTE" if "pagine 2-2" in user else "estratto rilevante"
            return "risposta finale"
        llm = _FakeLLM(responder)
        eng = MapReduceEngine(llm)
        res = await eng.run(question="q", blocks=_blocks(3))
        assert isinstance(res, MapReduceResult)
        assert res.n_blocks == 3
        assert res.n_partials == 2          # 1 e 3, non 2
        assert llm.map_calls == 3           # il map gira su tutti i blocchi
        assert res.content == "risposta finale"

    async def test_niente_case_insensitive_e_spazi(self):
        def responder(system, user):
            return "  niente  " if _is_map(system) else "X"
        eng = MapReduceEngine(_FakeLLM(responder))
        res = await eng.run(question="q", blocks=_blocks(2))
        assert res.n_partials == 0


class TestReduce:
    async def test_zero_parziali_non_chiama_reduce(self):
        def responder(system, user):
            return "NIENTE" if _is_map(system) else "non dovrebbe accadere"
        llm = _FakeLLM(responder)
        res = await eng_run(llm, _blocks(4))
        assert res.n_partials == 0
        assert llm.reduce_calls == 0
        assert "Non ho trovato" in res.content

    async def test_reduce_singolo_quando_pochi(self):
        def responder(system, user):
            return "estratto" if _is_map(system) else "finale"
        llm = _FakeLLM(responder)
        res = await eng_run(llm, _blocks(3))
        assert res.n_partials == 3
        assert llm.reduce_calls == 1        # tutti i parziali in un solo reduce
        assert res.content == "finale"

    async def test_reduce_gerarchico_quando_tanti(self):
        # Parziali da ~23 char ("[pagine i-i] ABCDEFGHIJ"); budget 50 ne fa stare
        # 2 per gruppo, non 3 -> piu' gruppi -> reduce intermedi + reduce finale.
        def responder(system, user):
            if _is_map(system):
                return "ABCDEFGHIJ"          # parziale corto ma non vuoto
            return "INT"                       # intermedi cortissimi -> 1 gruppo finale
        llm = _FakeLLM(responder)
        eng = MapReduceEngine(llm, reduce_block_chars=50)
        res = await eng.run(question="q", blocks=_blocks(5))
        assert res.n_partials == 5
        # 5 parziali -> gruppi [2,2,1] -> 3 reduce intermedi + 1 reduce finale.
        assert llm.reduce_calls >= 2         # cascata avvenuta
        assert res.content == "INT"


class TestConteggi:
    async def test_n_llm_calls(self):
        def responder(system, user):
            return "estratto" if _is_map(system) else "finale"
        llm = _FakeLLM(responder)
        res = await eng_run(llm, _blocks(3))
        # 3 map + 1 reduce = 4
        assert res.n_llm_calls == llm.map_calls + llm.reduce_calls == 4


async def eng_run(llm, blocks, **kw):
    return await MapReduceEngine(llm, **kw).run(question="q", blocks=blocks)
