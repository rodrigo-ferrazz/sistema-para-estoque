"""DTOs de domínio do módulo `notificacoes`."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from src.catalog.schemas import SKU
from src.usuarios.schemas import Papel

TipoEpisodio = Literal[
    "ruptura",
    "entrega_atrasada",
    "aviso",
    "queda_de_venda",
    "estoque_divergente",
    "decisao_sobre_aviso",
    "gondola_vazia",
    "verificacao_sobre_aviso",
]


class Detalhe(BaseModel):
    """Os números do momento de um episódio, como o pop-up de notificações (`notificacoes.js`)
    os lê. Um modelo por `TipoEpisodio`, em `DETALHES`: quem produz a condição monta o
    modelo do tipo, e um campo trocado falha aqui, não na tela."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DetalheDoSku(Detalhe):
    produto_nome: str
    cor: str
    tamanho: str


def nome_do_sku(sku: SKU) -> dict[str, str]:
    """Os campos de `DetalheDoSku`, para montar o detalhe com `**nome_do_sku(sku)`."""
    return {"produto_nome": sku.produto_nome, "cor": sku.cor, "tamanho": sku.tamanho}


class DetalheRuptura(DetalheDoSku):
    disponivel: int
    cobertura_dias: float | None


class DetalheEntregaAtrasada(Detalhe):
    fornecedor_nome: str
    data_prevista_entrega: date
    dias_de_atraso: int
    skus: list[str]


class DetalheAviso(DetalheDoSku):
    """`tipo` é o `TipoAviso` do `painel`; `quantidade`, as peças que faltaram no
    `faltou_na_gondola`."""

    tipo: str
    avisado_por: str
    comentario: str | None
    quantidade: int | None


class DetalheQuedaDeVenda(DetalheDoSku):
    disponivel: int
    venda_diaria_base: float
    vendido_na_janela: int
    dias_observados: int


class DetalheEstoqueDivergente(DetalheDoSku):
    disponivel_no_erp: int
    verificado_por: str
    comentario: str | None


class DetalheDecisaoSobreAviso(DetalheDoSku):
    """`tipo` é o `TipoDecisao`; `tipo_do_aviso`, o `TipoAviso` que a decisão fechou."""

    tipo: str
    quantidade: int | None
    motivo: str | None
    decidido_por: str
    tipo_do_aviso: str


class DetalheGondolaVazia(DetalheDoSku):
    setor: str
    avisado_por: str
    comentario: str | None
    disponivel_no_erp: int
    pecas: int | None


class DetalheVerificacaoSobreAviso(DetalheDoSku):
    """`resultado` é o `ResultadoVerificacao` da `reposicao`."""

    resultado: str
    comentario: str | None
    verificado_por: str


DETALHES: dict[TipoEpisodio, type[Detalhe]] = {
    "ruptura": DetalheRuptura,
    "entrega_atrasada": DetalheEntregaAtrasada,
    "aviso": DetalheAviso,
    "queda_de_venda": DetalheQuedaDeVenda,
    "estoque_divergente": DetalheEstoqueDivergente,
    "decisao_sobre_aviso": DetalheDecisaoSobreAviso,
    "gondola_vazia": DetalheGondolaVazia,
    "verificacao_sobre_aviso": DetalheVerificacaoSobreAviso,
}


class Condicao(BaseModel):
    """O que vale agora para um SKU ou para um pedido. A condição é o `tipo` com o
    `sku_code` e o `pedido_id`; `detalhe` guarda os números do momento, em JSON. Com
    `usuario_destino`, só essa pessoa vê a notificação; sem ele, todo o papel de destino.

    Quem produz a condição passa o `Detalhe` do tipo (ver `DETALHES`), que vira o JSON
    guardado. Um `Detalhe` de outro tipo é recusado."""

    model_config = ConfigDict(frozen=True)

    tipo: TipoEpisodio
    sku_code: str | None = None
    pedido_id: UUID | None = None
    papel_destino: Papel
    usuario_destino: UUID | None = None
    detalhe: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _detalhe_do_tipo(cls, dados: Any) -> Any:
        if not isinstance(dados, dict) or not isinstance(detalhe := dados.get("detalhe"), Detalhe):
            return dados
        esperado = DETALHES[dados["tipo"]]
        if type(detalhe) is not esperado:
            raise ValueError(f"O episódio '{dados['tipo']}' leva o detalhe {esperado.__name__}, não {type(detalhe).__name__}.")
        return {**dados, "detalhe": detalhe.model_dump(mode="json")}

    @property
    def chave(self) -> tuple[str, str | None, UUID | None]:
        return (self.tipo, self.sku_code, self.pedido_id)


class Episodio(BaseModel):
    """Intervalo em que uma condição valeu. Aberto enquanto `fechado_em` é nulo. Um evento
    de uma vez só, como o aviso da equipe de vendas, nasce fechado."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    tipo: TipoEpisodio
    sku_code: str | None
    pedido_id: UUID | None
    papel_destino: Papel
    usuario_destino: UUID | None
    aberto_em: datetime
    fechado_em: datetime | None
    detalhe: dict[str, Any]

    @property
    def chave(self) -> tuple[str, str | None, UUID | None]:
        return (self.tipo, self.sku_code, self.pedido_id)


class Notificacao(BaseModel):
    """Um episódio visto pelo usuário de destino ou, sem ele, por um usuário do papel de
    destino."""

    model_config = ConfigDict(frozen=True)

    episodio: Episodio
    lida: bool


class CaixaDeNotificacoes(BaseModel):
    """As notificações mais recentes primeiro e o total de não lidas, que pode passar do
    tamanho da lista."""

    model_config = ConfigDict(frozen=True)

    notificacoes: list[Notificacao]
    nao_lidas: int
