"""Demo end-to-end de Universal RAG (Fase 1).

Ejecuta:

    python examples/demo.py

Flujo:
    1. Ingesta de textos directos (conector text).
    2. Ingesta de archivos (conector file) — se generan en una carpeta temporal.
    3. Consultas con citas verificables.
    4. Estadísticas del pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline

console = Console()

SOLAR = """\
La energía solar fotovoltaica convierte la luz del sol en electricidad mediante celdas \
de silicio. Un panel doméstico típico de 450 vatios produce entre 1,8 y 2,5 kWh al día \
en climas soleados.

Los sistemas de bombeo solar permiten extraer agua sin baterías: el panel alimenta \
directamente la bomba durante las horas de sol y el agua se almacena en un tanque.

El mantenimiento de un sistema solar es mínimo: limpieza de paneles cada tres meses y \
revisión anual del inversor. La vida útil típica supera los 25 años.
"""

WIND = """\
La energía eólica aprovecha el viento mediante aerogeneradores. Un aerogenerador moderno \
de 3 MW abastece a unas 1.500 familias. Los parques eólicos offshore producen un 40 por \
ciento más que los terrestres por la constancia del viento marino.
"""


def main() -> None:
    console.print(Panel.fit("🌍 Universal RAG — demo Fase 1", style="cyan"))

    # Pipeline con almacén en memoria y embeddings deterministas (demo instantánea).
    # Para uso real: Settings() a secas usa ChromaDB persistente + sentence-transformers.
    with tempfile.TemporaryDirectory() as tmp:
        docs_dir = Path(tmp)

        # --- 1. Ingesta de archivos reales (conector file) ------------------
        (docs_dir / "manual_solar.txt").write_text(SOLAR, encoding="utf-8")
        (docs_dir / "manual_eolico.md").write_text(f"# Eólica\n\n{WIND}", encoding="utf-8")

        settings = Settings(store_provider="memory", embedding_provider="hash", llm_provider="echo")
        rag = RAGPipeline(settings=settings)

        console.print("\n[bold]1) Ingesta de archivos[/bold]")
        for path in sorted(docs_dir.glob("*")):
            result = rag.ingest(str(path))
            console.print(
                f"   ✅ {path.name}: {result.documents} documento(s), {result.chunks} fragmento(s)"
            )

        # --- 2. Ingesta de texto directo (conector text) --------------------
        console.print("\n[bold]2) Ingesta de texto directo[/bold]")
        result = rag.ingest_text(
            "El inversor híbrido gestiona paneles solares y baterías a la vez, priorizando "
            "el autoconsumo y exportando el excedente a la red.",
            title="Inversores híbridos",
        )
        console.print(f"   ✅ texto inline: {result.chunks} fragmento(s)")

        # --- 3. Consultas con citas -----------------------------------------
        console.print("\n[bold]3) Consultas con citas[/bold]")
        for question in (
            "¿cada cuánto se limpian los paneles solares?",
            "¿cuántas familias abastece un aerogenerador de 3 MW?",
        ):
            answer = rag.query(question)
            console.print(Panel(answer.answer, title=f"❓ {question} · {answer.model}"))
            for i, citation in enumerate(answer.citations, start=1):
                console.print(
                    f"   [{i}] {citation.source} · {citation.title} · score={citation.score:.3f}"
                )

        # --- 4. Estadísticas -------------------------------------------------
        console.print("\n[bold]4) Estadísticas[/bold]")
        table = Table()
        table.add_column("Clave", style="cyan")
        table.add_column("Valor")
        for key, value in rag.stats().items():
            if key != "connector_report":
                table.add_row(key, str(value))
        console.print(table)


if __name__ == "__main__":
    main()
