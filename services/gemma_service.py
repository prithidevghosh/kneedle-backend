from gemma_client import call_gemma4
from models import GaitMetrics, AnalysisResponse


def run_gemma_analysis(
    metrics: GaitMetrics,
    frames_b64: list[str],
    age: str,
    knee: str,
    lang: str,
    session_number: int
) -> AnalysisResponse:
    """Call Gemma 4 12b via Ollama with frames + metrics. Returns structured report."""
    return call_gemma4(metrics, frames_b64, age, knee, lang, session_number)
