"""CLI de Universal RAG (Typer + Rich).

Comandos:
    ingest SOURCE    ingesta un archivo/URL/patrón
    text TEXT        ingesta texto directo
    query QUESTION   pregunta con respuesta y citas
    chat             chat interactivo en la terminal
    eval GOLDEN      evalúa la recuperación con un golden set (JSONL)
    stats            estadísticas
    documents        lista documentos
    delete DOC_ID    elimina un documento
    clear            vacía el índice
    serve            arranca la API REST (con mini chat UI en /)
    mcp              arranca el servidor MCP
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from universal_rag import __version__
from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline

app = typer.Typer(
    name="universal-rag",
    help="Universal RAG — pipeline plugable de RAG con citas verificables.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


def _pipeline() -> RAGPipeline:
    return RAGPipeline(settings=Settings())


def _print_ingest(result) -> None:  # noqa: ANN001
    if result.errors:
        for error in result.errors:
            console.print(f"[red]✗ {error}[/red]")
    if result.documents:
        console.print(
            f"[green]✓[/green] Conector '{result.connector}': {result.documents} documento(s), "
            f"{result.chunks} fragmento(s)"
            + (f", {result.skipped} omitido(s)" if result.skipped else "")
        )


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Ruta, globo (docs/*.pdf) o URL a ingestar"),
    title: str | None = typer.Option(None, "--title", "-t", help="Título del documento"),
    connector: str | None = typer.Option(None, "--connector", "-c", help="Forzar conector concreto"),
) -> None:
    """Ingesta un archivo, patrón o URL en la base de conocimiento."""
    pipeline = _pipeline()
    options = {"title": title} if title else {}
    _print_ingest(pipeline.ingest(source, connector=connector, **options))


@app.command()
def text(
    content: str = typer.Argument(..., help="Contenido de texto a ingestar"),
    title: str | None = typer.Option(None, "--title", "-t", help="Título del documento"),
) -> None:
    """Ingesta texto directo como documento."""
    _print_ingest(_pipeline().ingest_text(content, title=title))


@app.command()
def query(
    question: str = typer.Argument(..., help="Pregunta en lenguaje natural"),
    k: int | None = typer.Option(None, "--k", "-k", min=1, max=50, help="Fragmentos a recuperar"),
) -> None:
    """Responde una pregunta con citas a las fuentes."""
    answer = _pipeline().query(question, k=k)
    console.print(Panel(answer.answer, title=f"Respuesta · {answer.model} · {answer.latency_ms:.0f} ms"))
    if answer.citations:
        table = Table(title="Fuentes citadas", show_lines=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Fuente", style="magenta")
        table.add_column("Título")
        table.add_column("Momento", style="green", overflow="fold")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Cita", overflow="fold")
        for i, citation in enumerate(answer.citations, start=1):
            table.add_row(
                str(i),
                citation.source,
                citation.title or "—",
                citation.timestamp or "—",
                f"{citation.score:.3f}",
                citation.quote,
            )
        console.print(table)


@app.command()
def chat(k: int | None = typer.Option(None, "--k", "-k", min=1, max=50)) -> None:
    """Chat interactivo en la terminal ('salir' para terminar)."""
    pipeline = _pipeline()
    console.print(
        f"[cyan]Universal RAG[/cyan] · búsqueda [bold]{pipeline.retrieval_label()}[/bold] · "
        "escribe [bold]salir[/bold] para terminar"
    )
    while True:
        try:
            question = console.input("\n[bold green]Tú ▸ [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit", "q"}:
            break
        answer = pipeline.query(question, k=k)
        console.print(Panel(answer.answer, title=f"{answer.model} · {answer.latency_ms:.0f} ms"))
        for i, citation in enumerate(answer.citations, start=1):
            moment = f" {citation.timestamp}" if citation.timestamp else ""
            console.print(f"  [cyan][{i}][/] {citation.title or citation.source}{moment}")
            console.print(f"      [dim]{citation.quote}[/dim]")
    console.print("[dim]¡Hasta luego![/dim]")


@app.command("eval")
def eval_command(
    golden: Path = typer.Argument(..., exists=True, dir_okay=False, help="Golden set JSONL/JSON"),
    k: int = typer.Option(5, "--k", "-k", min=1, max=50, help="Top-k a evaluar"),
    json_output: bool = typer.Option(False, "--json", help="Reporte JSON completo"),
) -> None:
    """Evalúa la calidad de recuperación con un golden set."""
    from universal_rag.core.evaluation import evaluate, load_golden

    cases = load_golden(golden)
    if not cases:
        console.print("[red]El golden set no contiene casos válidos.[/red]")
        raise typer.Exit(code=1)
    report = evaluate(_pipeline(), cases, k=k)
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False, indent=2))
        return
    table = Table(title=f"Evaluación · {report['cases']} caso(s) · k={k}")
    table.add_column("Consulta", overflow="fold")
    table.add_column("Relevantes")
    table.add_column("Hits", justify="right")
    table.add_column("P", justify="right", style="green")
    table.add_column("R", justify="right", style="cyan")
    table.add_column("MRR", justify="right", style="magenta")
    for case in report["per_case"]:
        table.add_row(
            case["query"],
            ", ".join(case["relevant"]) or "—",
            str(case["hits"]),
            f"{case['precision']:.2f}",
            f"{case['recall']:.2f}",
            f"{case['mrr']:.2f}",
        )
    console.print(table)
    console.print(
        f"[bold]Agregado[/bold] · precision [green]{report['precision']:.3f}[/green] · "
        f"recall [cyan]{report['recall']:.3f}[/cyan] · MRR [magenta]{report['mrr']:.3f}[/magenta] · "
        f"hit_rate {report['hit_rate']:.3f}"
    )


@app.command()
def stats() -> None:
    """Muestra estadísticas del pipeline."""
    data = _pipeline().stats()
    table = Table(title=f"Universal RAG v{__version__}")
    table.add_column("Clave", style="cyan")
    table.add_column("Valor")
    for key, value in data.items():
        if key == "connector_report":
            continue
        table.add_row(key, str(value))
    console.print(table)


@app.command("documents")
def documents() -> None:
    """Lista los documentos ingestado."""
    rows = _pipeline().documents()
    if not rows:
        console.print("[yellow]Sin documentos ingestado.[/yellow]")
        return
    table = Table(title=f"{len(rows)} documento(s)")
    table.add_column("ID", style="cyan")
    table.add_column("Fuente", style="magenta")
    table.add_column("Título")
    table.add_column("Fragmentos", justify="right")
    table.add_column("URI", overflow="fold")
    for row in rows:
        table.add_row(row["document_id"], row["source"], row.get("title") or "—", str(row["chunks"]), row["uri"])
    console.print(table)


@app.command()
def delete(document_id: str = typer.Argument(..., help="ID del documento a eliminar")) -> None:
    """Elimina un documento y sus fragmentos."""
    deleted = _pipeline().delete_document(document_id)
    console.print(f"[green]✓ Eliminado[/green] {document_id}" if deleted else f"[red]✗ No encontrado:[/red] {document_id}")


@app.command()
def clear(confirm: bool = typer.Option(False, "--yes", "-y", prompt="¿Vaciar TODO el índice?")) -> None:
    """Vacía completamente el índice."""
    pipeline = _pipeline()
    pipeline.clear()
    console.print("[green]✓ Índice vaciado[/green]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8100, "--port", "-p"),
) -> None:
    """Arranca la API REST (FastAPI + uvicorn)."""
    import uvicorn

    console.print(f"[cyan]Universal RAG API[/cyan] en http://{host}:{port} (docs: /docs)")
    uvicorn.run("universal_rag.api.app:app", host=host, port=port, factory=False)


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-T", help="stdio | streamable-http | sse"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8200, "--port", "-p"),
) -> None:
    """Arranca el servidor MCP (para Claude Desktop / Universal MCP)."""
    import sys

    from universal_rag.mcp_server import main as mcp_main

    sys.argv = ["universal-rag-mcp", "--transport", transport, "--host", host, "--port", str(port)]
    mcp_main()


if __name__ == "__main__":  # pragma: no cover
    app()
