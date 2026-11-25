"""Conversores de DTOs de domínio para DTOs HTTP, usados por mais de um router ou endpoint.

Quando o DTO HTTP é o de domínio com menos campos (sem o `usuario_id`, por exemplo), o
conversor lê do domínio os campos do DTO HTTP (`from_attributes`): campo novo no domínio
só aparece na API quando entra no DTO HTTP, e campo do DTO HTTP que sumiu do domínio falha
aqui. Os conversores escritos à mão são os que mudam a forma."""
from __future__ import annotations

from src.ai.schemas import (
    SinaisDoSKU,
    SinalCorpus,
    SugestaoComSinais,
    TrechoClassificado,
    VerificacaoCitacao,
)
from src.api.schemas import (
    AnaliseSKUResponse,
    AvisoResponse,
    CobrancaEntregaResponse,
    DecisaoCompraResponse,
    FornecedorResponse,
    GiroResponse,
    SinaisDoSKUResponse,
    SinalCorpusResponse,
    SugestaoComSinaisResponse,
    SugestaoPedidoResponse,
    TrechoClassificadoResponse,
    VerificacaoCitacaoResponse,
    VerificacaoGondolaResponse,
)
from src.catalog.schemas import FornecedorParaSKU
from src.ficha_sku.schemas import Ficha
from src.painel.schemas import Aviso, CobrancaEntrega, DecisaoCompra
from src.purchasing.schemas import SugestaoPedido
from src.reposicao.schemas import VerificacaoGondola


def fornecedor_to_response(f: FornecedorParaSKU) -> FornecedorResponse:
    return FornecedorResponse.model_validate(f, from_attributes=True)


def ficha_to_response(ficha: Ficha) -> AnaliseSKUResponse:
    return AnaliseSKUResponse(
        sku_code=ficha.sku.sku_code,
        produto_id=ficha.sku.produto_id,
        produto_nome=ficha.sku.produto_nome,
        categoria=ficha.sku.categoria,
        cor=ficha.sku.cor,
        tamanho=ficha.sku.tamanho,
        estoque=ficha.estoque,
        em_transito_unidades=ficha.em_transito,
        giro=GiroResponse(
            unidades_por_mes=ficha.giro.unidades_por_mes,
            meses_considerados=ficha.giro.meses_considerados,
        ),
        cobertura=ficha.cobertura,
        fornecedores=[fornecedor_to_response(f) for f in ficha.fornecedores],
    )


def sugestao_to_response(sugestao: SugestaoPedido) -> SugestaoPedidoResponse:
    return SugestaoPedidoResponse(
        sku_code=sugestao.sku_code,
        quantidade=sugestao.quantidade,
        motivo=sugestao.motivo,
        fornecedor=(
            fornecedor_to_response(sugestao.fornecedor)
            if sugestao.fornecedor is not None
            else None
        ),
        valor_estimado_centavos=sugestao.valor_estimado_centavos,
        calculo=sugestao.calculo,
        alertas=sugestao.alertas,
        politica_versao=sugestao.politica_versao,
    )


def sinal_to_response(sinal: SinalCorpus) -> SinalCorpusResponse:
    return SinalCorpusResponse.model_validate(sinal, from_attributes=True)


def sugestao_com_sinais_to_response(com_sinais: SugestaoComSinais) -> SugestaoComSinaisResponse:
    return SugestaoComSinaisResponse(
        sugestao=sugestao_to_response(com_sinais.sugestao), sinais=_sinais_to_response(com_sinais.sinais)
    )


def sinais_do_sku_to_response(sinais: SinaisDoSKU) -> SinaisDoSKUResponse:
    return SinaisDoSKUResponse(sku_code=sinais.sku_code, sinais=_sinais_to_response(sinais.sinais))


def _sinais_to_response(sinais: list[SinalCorpus] | None) -> list[SinalCorpusResponse] | None:
    return None if sinais is None else [sinal_to_response(s) for s in sinais]


def verificacao_to_response(verificacao: VerificacaoCitacao) -> VerificacaoCitacaoResponse:
    return VerificacaoCitacaoResponse.model_validate(verificacao, from_attributes=True)


def trecho_to_response(trecho: TrechoClassificado) -> TrechoClassificadoResponse:
    return TrechoClassificadoResponse.model_validate(trecho, from_attributes=True)


def aviso_to_response(aviso: Aviso) -> AvisoResponse:
    return AvisoResponse.model_validate(aviso, from_attributes=True)


def decisao_to_response(decisao: DecisaoCompra) -> DecisaoCompraResponse:
    return DecisaoCompraResponse.model_validate(decisao, from_attributes=True)


def cobranca_to_response(cobranca: CobrancaEntrega) -> CobrancaEntregaResponse:
    return CobrancaEntregaResponse.model_validate(cobranca, from_attributes=True)


def verificacao_gondola_to_response(v: VerificacaoGondola) -> VerificacaoGondolaResponse:
    return VerificacaoGondolaResponse.model_validate(v, from_attributes=True)
