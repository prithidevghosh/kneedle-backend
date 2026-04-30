from fastapi import APIRouter

from handlers.diagnostics_handler import handle_health, handle_test_ollama

router = APIRouter()


@router.get("/health")
def health():
    """
    Server status check. The app calls this on startup to verify the
    laptop server is reachable on the local network before showing
    the record button.
    """
    return handle_health()


@router.get("/test-ollama")
def test_ollama():
    """
    Checks if Gemma 4 12b is loaded and ready in Ollama.
    Called once when the app first connects.
    If this returns gemma4_ready=false, the app shows an "AI not ready" state.
    """
    return handle_test_ollama()
