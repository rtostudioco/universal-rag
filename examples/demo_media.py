"""Demo Fase 2: audio/video → transcripción local → citas por minuto.

Uso:

    python examples/demo_media.py /ruta/al/podcast.mp3
    python examples/demo_media.py /ruta/al/clase.mp4 --language es

Requiere el extra de media:  pip install "universal-rag[media]"
El primer uso descarga el modelo Whisper (default: base, ~75 MB).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de audio/video con citas por minuto")
    parser.add_argument("media", help="Ruta del archivo (MP3, WAV, M4A, MP4, MKV…)")
    parser.add_argument("--language", "-l", default=None, help='Idioma ("es", "en"…; default: auto)')
    parser.add_argument("--asr-model", default=None, help="Modelo Whisper (tiny/base/small/medium)")
    parser.add_argument("--question", "-q", default=None, help="Pregunta a hacer tras la ingesta")
    args = parser.parse_args()

    media = Path(args.media)
    if not media.exists():
        console.print(f"[red]No existe:[/red] {media}")
        raise SystemExit(1)

    rag = RAGPipeline(settings=Settings(store_provider="memory", embedding_provider="hash", llm_provider="echo"))

    console.print(Panel.fit(f"🎙️ Universal RAG — media demo: {media.name}", style="cyan"))
    result = rag.ingest(str(media), language=args.language, asr_model=args.asr_model)
    if result.errors or result.documents == 0:
        for error in result.errors:
            console.print(f"[red]✗ {error}[/red]")
        raise SystemExit(1)
    console.print(
        f"[green]✓[/green] Conector '{result.connector}': {result.documents} documento, "
        f"{result.chunks} fragmento(s) temporales"
    )

    doc = rag.documents()[-1]
    console.print(f"    uri: {doc['uri']}")

    question = args.question or "¿cuáles son los puntos principales del audio?"
    answer = rag.query(question)
    console.print(Panel(answer.answer, title=f"❓ {question}"))
    for i, citation in enumerate(answer.citations, start=1):
        minute = f" · {citation.timestamp}" if citation.timestamp else ""
        console.print(f"   [{i}] {citation.source} · {citation.title}{minute}")
        console.print(f"       «{citation.quote[:160]}»")


if __name__ == "__main__":
    main()
