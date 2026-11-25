"""O detalhe de cada tipo de episódio: o contrato entre quem produz a condição e o pop-up."""
from __future__ import annotations

from datetime import date
from typing import get_args

import pytest
from pydantic import ValidationError

from src.notificacoes.schemas import (
    DETALHES,
    Condicao,
    DetalheEntregaAtrasada,
    DetalheRuptura,
    TipoEpisodio,
)


def test_todo_tipo_de_episodio_tem_o_modelo_do_detalhe() -> None:
    assert set(DETALHES) == set(get_args(TipoEpisodio))


def test_o_detalhe_vira_o_json_guardado() -> None:
    condicao = Condicao(
        tipo="entrega_atrasada",
        papel_destino="comprador",
        detalhe=DetalheEntregaAtrasada(
            fornecedor_nome="Katrina", data_prevista_entrega=date(2026, 11, 1), dias_de_atraso=3, skus=["A", "B"]
        ),
    )
    assert condicao.detalhe == {
        "fornecedor_nome": "Katrina",
        "data_prevista_entrega": "2026-11-01",
        "dias_de_atraso": 3,
        "skus": ["A", "B"],
    }


def test_o_detalhe_de_outro_tipo_e_recusado() -> None:
    detalhe = DetalheRuptura(produto_nome="Toalha", cor="Bege", tamanho="70x140", disponivel=0, cobertura_dias=0.0)
    with pytest.raises(ValidationError, match="DetalheEstoqueDivergente"):
        Condicao(tipo="estoque_divergente", sku_code="X", papel_destino="comprador", detalhe=detalhe)


def test_campo_a_mais_no_detalhe_e_recusado() -> None:
    with pytest.raises(ValidationError):
        DetalheRuptura(
            produto_nome="Toalha", cor="Bege", tamanho="70x140", disponivel=0, cobertura_dias=0.0, cobertura=1
        )
