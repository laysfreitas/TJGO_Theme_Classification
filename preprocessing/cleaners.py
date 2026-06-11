"""Funções puras de limpeza de petições.

Cada função recebe `str` e devolve `str`. Não há I/O, não há estado
global, não há logging implícito. Isso permite composição, teste e
paralelização triviais.

Ordem canônica de aplicação (ver ``noise-patterns.md``, seção
"Sequência recomendada"):

1. :func:`normalizar_unicode` (N7)
2. :func:`remover_marcadores_exportacao` (N1)
3. :func:`separar_e_consolidar_documentos` (Q1) — opera sobre o texto
   **antes** da remoção de marcadores se houver mais de um par.
4. :func:`remover_urls_camscanner` (N3)
5. :func:`remover_urls_genericas` (N8)
6. :func:`cortar_cabecalho_institucional` (N2)
7. :func:`cortar_assinatura_final` (N4)
8. :func:`remover_codigos_controle` (N6)
9. :func:`colapsar_espacos` (N7 — passe final)

Funções auxiliares de decisão (`contem_apenas_url`, `deve_descartar_subdocumento`)
vivem aqui também para facilitar reuso em notebooks e testes.
"""

from __future__ import annotations

import re
import unicodedata

SEPARADOR_NEUTRO: str = " ||| "
"""Separador usado ao concatenar sub-documentos não-triviais."""

LIMITE_REDUCAO_ETAPA: float = 0.70
"""Teto de redução aceitável em uma única etapa de corte (D13).

Se ``cortar_cabecalho_institucional`` ou ``cortar_assinatura_final`` derem
uma saída com menos de ``(1 - LIMITE_REDUCAO_ETAPA)`` do tamanho da
entrada, a etapa é revertida (rollback) e o texto de entrada é preservado.

Fundamentação: peças de contrarrazões do MP (padrão frequente no dataset)
combinam ofício de remessa curto + peça substantiva longa. A última ocorrência
de "pede deferimento" pode estar no ofício, fazendo `rfind` descartar o miolo
argumentativo. Assinaturas e cabeçalhos legítimos têm <30% do texto; qualquer
corte >70% é sinal forte de falso positivo."""


def _guard_rail_reducao(
    texto_entrada: str,
    texto_saida: str,
    *,
    limite: float = LIMITE_REDUCAO_ETAPA,
) -> str:
    """Reverter corte se a redução for excessiva (D13).

    Retorna ``texto_saida`` se a redução for ≤ ``limite``; caso contrário
    devolve ``texto_entrada`` (rollback da etapa). Textos vazios não são
    afetados — rollback só dispara quando há algo para preservar.
    """

    if not texto_entrada:
        return texto_saida
    reducao = 1 - len(texto_saida) / len(texto_entrada)
    if reducao > limite:
        return texto_entrada
    return texto_saida

MARCADOR_INICIO: str = ">>>>>inicio<<<<<"
MARCADOR_FIM_RE: str = r"#####fim#####id:\d+"

_RE_BLOCO = re.compile(
    rf"{re.escape(MARCADOR_INICIO)}\s*(.*?)\s*{MARCADOR_FIM_RE}",
    flags=re.DOTALL,
)
_RE_MARCADOR_INICIO = re.compile(re.escape(MARCADOR_INICIO))
_RE_MARCADOR_FIM = re.compile(MARCADOR_FIM_RE)

_RE_URL_CAMSCANNER = re.compile(
    r"https?://v3\.camscanner\.com/user/download\S*",
    flags=re.IGNORECASE,
)
_RE_URL_GENERICA = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")

_RE_OAB = re.compile(
    r"OAB[/\- ]?(?:GO|SP|MG|RJ|DF|BA|PR|RS|SC|PE|CE|PA|MA|MT|MS|AM|ES|AL|PI|"
    r"RN|PB|SE|AC|AP|RO|RR|TO)\s*n?[º°]?\s*[\d.\-/]+",
    flags=re.IGNORECASE,
)

_RE_JURSP = re.compile(r"JURSP\s*-\s*\d+v\d+\s*-\s*[\d.]+", flags=re.IGNORECASE)
_RE_IMPRESSAO = re.compile(
    r"IMPRESS[ÃA]O:\s*\d{1,2}/\d{1,2}/\d{2,4}\s*-\s*\d{1,2}:\d{2}:\d{2}"
    r"(?:\s*-\s*[^\n]*)?",
    flags=re.IGNORECASE,
)
_RE_PAGINACAO = re.compile(r"P[ÁA]GINA:\s*\d+/?\d*", flags=re.IGNORECASE)
_RE_CODIGO_VALIDACAO = re.compile(
    r"C[ÓO]DIGO DE VALIDA[ÇC][ÃA]O:\s*[a-f0-9]{16,}",
    flags=re.IGNORECASE,
)

_RE_WHITESPACE = re.compile(r"\s+")

_ENDERECAMENTO_JUDICIAL: tuple[str, ...] = (
    "EXMO.",
    "EXMA.",
    "Mm(a). Juiz(íza)",
    "EXCELENTÍSSIMO",
    "EXCELENTISSIMO",
    "EXCELENTÍSSIMA",
    "EXCELENTISSIMA",
    "AO JUÍZO",
    "AO JUIZO",
    "AO EGRÉGIO",
    "AO EGREGIO",
    "AO DOUTO JUÍZO",
    "AO DOUTO JUIZO",
    "AO DOUTO JUÍZO DE DIREITO",
    "AO MERITÍSSIMO",
    "AO MERITISSIMO",
    "MERITÍSSIMO",
    "MERITISSIMO",
    "EGRÉGIO TRIBUNAL",
    "EGREGIO TRIBUNAL",
)

_PEDE_DEFERIMENTO: tuple[str, ...] = (
    "PEDE DEFERIMENTO",
    "PEDE-SE DEFERIMENTO",
    "PEDE O DEFERIMENTO",
    "P. DEFERIMENTO",
    "P.DEFERIMENTO",
    "NESTES TERMOS, PEDE",
    "NESTES TERMOS PEDE",
    "TERMOS EM QUE",
    "TERMOS EM QUE PEDE",
    "TERMOS EM QUE PEDE DEFERIMENTO",
    "NESTES TERMOS",
)


def normalizar_unicode(texto: str) -> str:
    """Normalizar ``texto`` para Unicode NFKC.

    Remove também ``\\xa0`` (non-breaking space) e outros espaços
    exóticos (U+2007, U+202F) convertendo para espaço simples.
    Trata o ruído N7 do catálogo de ruídos.
    """

    if not texto:
        return ""
    normalizado = unicodedata.normalize("NFKC", texto)
    normalizado = normalizado.replace("\u00a0", " ")
    normalizado = normalizado.replace("\u2007", " ")
    normalizado = normalizado.replace("\u202f", " ")
    return normalizado


def colapsar_espacos(texto: str) -> str:
    """Colapsar qualquer sequência de whitespace para um único espaço.

    Preserva conteúdo. Passe final após todas as remoções.
    """

    if not texto:
        return ""
    return _RE_WHITESPACE.sub(" ", texto).strip()


def contar_marcadores(texto: str) -> int:
    """Contar quantos pares ``>>>>>inicio<<<<<`` aparecem no texto."""

    if not texto:
        return 0
    return len(_RE_MARCADOR_INICIO.findall(texto))


def extrair_subdocumentos(texto: str) -> list[str]:
    """Extrair o conteúdo de cada par inicio/fim como sub-documento.

    Se o texto não tiver marcadores, devolve uma lista com o texto
    original (tratando-o como subdocumento único).
    """

    if not texto:
        return []
    blocos = _RE_BLOCO.findall(texto)
    if not blocos:
        return [texto]
    return [bloco.strip() for bloco in blocos]


def remover_marcadores_exportacao(texto: str) -> str:
    """Remover os marcadores ``>>>>>inicio<<<<<`` e ``#####fim#####id:<N>``.

    Trata o ruído N1. Aplica-se apenas a textos com **um único** par
    de marcadores. Para múltiplos, use
    :func:`separar_e_consolidar_documentos` antes.
    """

    if not texto:
        return ""
    texto = _RE_MARCADOR_INICIO.sub(" ", texto)
    texto = _RE_MARCADOR_FIM.sub(" ", texto)
    return texto


def contem_apenas_url(texto: str, *, limite_chars: int = 50) -> bool:
    """Indicar se ``texto`` é trivialmente ruído (só URLs / muito curto).

    Critérios:
    - após remover URLs e whitespace, restam menos de ``limite_chars``
      caracteres úteis.
    """

    if not texto:
        return True
    sem_url = _RE_URL_GENERICA.sub(" ", texto)
    sem_url = _RE_EMAIL.sub(" ", sem_url)
    util = _RE_WHITESPACE.sub("", sem_url)
    return len(util) < limite_chars


def deve_descartar_subdocumento(texto: str, *, limite_chars: int = 200) -> bool:
    """Decidir se um sub-documento é trivial e deve ser descartado.

    Regra N11: descartar se tem menos de ``limite_chars`` chars **ou**
    se contém apenas URLs.
    """

    if texto is None:
        return True
    texto_limpo = colapsar_espacos(texto)
    if len(texto_limpo) < limite_chars:
        return True
    if contem_apenas_url(texto_limpo):
        return True
    return False


def separar_e_consolidar_documentos(
    texto: str,
    *,
    limite_chars_subdoc: int = 200,
    separador: str = SEPARADOR_NEUTRO,
) -> str:
    """Separar múltiplos pares inicio/fim em sub-documentos e consolidar.

    Trata Q1 (ver ``quality-issues.md``):

    - Se há 0 ou 1 marcador de início: remove os marcadores e devolve
      o texto inteiro.
    - Se há 2+: extrai cada sub-documento, descarta os triviais
      (``deve_descartar_subdocumento``) e concatena os substantivos
      com ``separador``.
    """

    if not texto:
        return ""

    n = contar_marcadores(texto)
    if n <= 1:
        return remover_marcadores_exportacao(texto)

    subdocs = extrair_subdocumentos(texto)
    substantivos = [
        s for s in subdocs if not deve_descartar_subdocumento(s, limite_chars=limite_chars_subdoc)
    ]
    if not substantivos:
        return remover_marcadores_exportacao(texto).strip()
    return separador.join(substantivos)


def remover_urls_camscanner(texto: str) -> str:
    """Remover ocorrências de ``https://v3.camscanner.com/...``. Trata N3."""

    if not texto:
        return ""
    return _RE_URL_CAMSCANNER.sub(" ", texto)


def remover_urls_genericas(texto: str) -> str:
    """Remover qualquer URL ``http(s)://...``. Trata N8.

    Deve rodar **depois** de :func:`remover_urls_camscanner` apenas
    por clareza de logs — o resultado é idempotente.
    """

    if not texto:
        return ""
    return _RE_URL_GENERICA.sub(" ", texto)


def cortar_cabecalho_institucional(texto: str) -> str:
    """Cortar prefixo anterior ao primeiro endereçamento judicial. Trata N2.

    Estratégia: procura a primeira ocorrência (case-insensitive) de
    qualquer termo de :data:`_ENDERECAMENTO_JUDICIAL` e devolve o
    texto **a partir** dela. Se nenhum for encontrado (ex.: sub-documento
    que é anexo técnico), devolve o texto inalterado.
    """

    if not texto:
        return ""
    texto_upper = texto.upper()
    melhor_idx: int | None = None
    for marcador in _ENDERECAMENTO_JUDICIAL:
        idx = texto_upper.find(marcador)
        if idx != -1 and (melhor_idx is None or idx < melhor_idx):
            melhor_idx = idx
    if melhor_idx is None:
        return texto
    saida = texto[melhor_idx:]
    return _guard_rail_reducao(texto, saida)


def tem_pede_deferimento(texto: str) -> bool:
    """Indicar se ``texto`` contém alguma das fórmulas de encerramento.

    Útil para medir a cobertura de :func:`cortar_assinatura_final`:
    petições sem fórmula não sofrem corte de sufixo (apenas a remoção
    de matrícula OAB remanescente). Não faz corte nem altera o texto.
    """

    if not texto:
        return False
    texto_upper = texto.upper()
    for marcador in _PEDE_DEFERIMENTO:
        if marcador in texto_upper:
            return True
    return False


def cortar_assinatura_final(texto: str) -> str:
    """Cortar o sufixo após a última fórmula de deferimento. Trata N4.

    Após cortar, também remove linhas contendo matrícula OAB que
    eventualmente tenham sobrado no corpo.
    """

    if not texto:
        return ""
    entrada = texto
    texto_upper = texto.upper()
    melhor_idx: int | None = None
    for marcador in _PEDE_DEFERIMENTO:
        idx = texto_upper.rfind(marcador)
        if idx != -1 and (melhor_idx is None or idx > melhor_idx):
            melhor_idx = idx + len(marcador)
    if melhor_idx is not None:
        cortado = texto[:melhor_idx]
        texto = _guard_rail_reducao(entrada, cortado)
    texto = _RE_OAB.sub(" ", texto)
    return texto


def remover_codigos_controle(texto: str) -> str:
    """Remover códigos internos de controle: JURSP, paginação, impressão, validação.

    Trata N6 e parte de N8 (código de validação).
    """

    if not texto:
        return ""
    texto = _RE_JURSP.sub(" ", texto)
    texto = _RE_IMPRESSAO.sub(" ", texto)
    texto = _RE_PAGINACAO.sub(" ", texto)
    texto = _RE_CODIGO_VALIDACAO.sub(" ", texto)
    return texto

