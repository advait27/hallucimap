"""hallucimap CLI — scan and visualize LLM hallucination risk.

Commands
--------
hallucimap scan   Run a full probe scan against a model.
hallucimap show   Render an existing RiskAtlas as a heatmap.

Examples
--------
    hallucimap scan --model gpt-4o --domains science,history --output atlas.json
    hallucimap show atlas.json --browser
    hallucimap show atlas.json --save map.html
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="hallucimap",
    help="Cartograph hallucination risk across an LLM's knowledge space.",
    add_completion=False,
)
console = Console()

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_SUPPORTED_MODELS = {
    "gpt-4o": "openai",
    "gpt-4-turbo": "openai",
    "gpt-3.5-turbo": "openai",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-opus-4-6": "anthropic",
}

_DEFAULT_DOMAINS = "science,history,law,medicine,finance,geography"


def _make_adapter(model: str) -> object:
    """Instantiate the correct adapter for a given model ID."""
    provider = _SUPPORTED_MODELS.get(model)
    if provider is None:
        # Fall back: assume HuggingFace local model
        from hallucimap.models import HFAdapter

        console.print(f"[yellow]Unknown model '{model}' — treating as HuggingFace local model.[/]")
        return HFAdapter(model=model)

    if provider == "openai":
        from hallucimap.models import OpenAIAdapter

        return OpenAIAdapter(model=model)
    else:
        from hallucimap.models import AnthropicAdapter

        return AnthropicAdapter(model=model)


# ------------------------------------------------------------------ #
# scan                                                                 #
# ------------------------------------------------------------------ #


@app.command()
def scan(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model ID to probe."),
    domains: str = typer.Option(
        _DEFAULT_DOMAINS,
        "--domains",
        "-d",
        help="Comma-separated domain list.",
    ),
    samples: int = typer.Option(5, "--samples", "-n", help="Consistency samples per question."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output atlas JSON path."),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Max concurrent API calls."),
    temperature: float = typer.Option(0.9, "--temperature", "-t", help="Sampling temperature."),
) -> None:
    """Run a hallucination risk scan against a model.

    Probes the model across the specified domains, scores each response for
    hallucination risk, and writes a :class:`RiskAtlas` JSON to ``--output``.

    Examples
    --------
        hallucimap scan --model gpt-4o --domains science,history --samples 5
    """
    from hallucimap.core.atlas import RiskAtlas, ScanSession
    from hallucimap.core.scorer import HallucinationScorer
    from hallucimap.probes import DomainProbe, FactualProbe

    output_path = output or Path(f"atlas_{model.replace('/', '_')}.json")
    domain_list = [d.strip() for d in domains.split(",") if d.strip()]

    console.rule(f"[bold cyan]hallucimap scan[/] — model: {model}")
    console.print(f"  Domains   : {', '.join(domain_list)}")
    console.print(f"  Samples   : {samples}")
    console.print(f"  Output    : {output_path}")

    adapter = _make_adapter(model)

    async def _run() -> None:
        atlas = RiskAtlas(model_id=model)
        session = ScanSession(model_id=model)
        scorer = HallucinationScorer(
            adapter=adapter,  # type: ignore[arg-type]
            n_samples=samples,
            temperature=temperature,
        )

        all_probe_results = []

        # Calibration: always run factual probes first
        console.print("\n[bold]Running calibration (factual) probes…[/]")
        factual_probe = FactualProbe()
        factual_results = await factual_probe.run_all(
            adapter,  # type: ignore[arg-type]
            concurrency=concurrency,
        )
        all_probe_results.extend(factual_results)

        # Domain probes
        for domain in domain_list:
            console.print(f"[bold]Probing domain:[/] {domain}")
            probe = DomainProbe(domain=domain)
            results = await probe.run_all(
                adapter,  # type: ignore[arg-type]
                concurrency=concurrency,
            )
            all_probe_results.extend(results)

        # Score everything
        console.print(f"\nScoring {len(all_probe_results)} probe results…")
        questions = [
            (r.question, r.domain, r.subdomain) for r in all_probe_results
        ]
        references = [r.reference for r in all_probe_results]
        scored = await scorer.score_batch(
            questions,
            references=references,
            concurrency=concurrency,
        )

        atlas.update(scored, session=session)
        atlas.save(output_path)

        # Summary table
        _print_summary(atlas)
        console.print(f"\n[green]Atlas saved to:[/] {output_path}")

    asyncio.run(_run())


# ------------------------------------------------------------------ #
# show                                                                 #
# ------------------------------------------------------------------ #


@app.command()
def show(
    atlas_path: Path = typer.Argument(..., help="Path to an atlas JSON file."),
    browser: bool = typer.Option(False, "--browser", help="Open the heatmap in a browser."),
    save: Path | None = typer.Option(None, "--save", "-s", help="Save heatmap as HTML."),
) -> None:
    """Render a saved RiskAtlas as an interactive heatmap.

    Examples
    --------
        hallucimap show atlas_gpt-4o.json --browser
        hallucimap show atlas_gpt-4o.json --save map.html
    """
    from hallucimap.core.atlas import RiskAtlas
    from hallucimap.viz import HeatmapRenderer

    if not atlas_path.exists():
        console.print(f"[red]File not found:[/] {atlas_path}")
        raise typer.Exit(code=1)

    atlas = RiskAtlas.load(atlas_path)
    _print_summary(atlas)

    renderer = HeatmapRenderer(atlas)

    if save:
        renderer.save(save)
        console.print(f"[green]Heatmap saved to:[/] {save}")
    if browser:
        renderer.show()
    if not browser and not save:
        console.print("[yellow]Tip:[/] pass --browser to open or --save <path> to export.")


# ------------------------------------------------------------------ #
# Private                                                              #
# ------------------------------------------------------------------ #


def _print_summary(atlas: object) -> None:
    from hallucimap.core.atlas import RiskAtlas

    assert isinstance(atlas, RiskAtlas)
    hot = atlas.hottest_cells(n=10)

    table = Table(title=f"Risk Atlas: {atlas.model_id}", show_header=True)
    table.add_column("Domain", style="cyan")
    table.add_column("Subdomain", style="magenta")
    table.add_column("Risk", justify="right", style="red")
    table.add_column("Confidence", justify="right")
    table.add_column("Samples", justify="right")

    for cell in hot:
        risk_str = f"{cell.risk_score:.3f}"
        conf_str = f"{cell.confidence:.2f}"
        table.add_row(cell.domain, cell.subdomain, risk_str, conf_str, str(cell.sample_count))

    console.print(table)
    console.print(atlas.summary())


if __name__ == "__main__":
    app()
