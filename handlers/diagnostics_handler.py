def handle_health() -> dict:
    return {
        "status": "running",
        "service": "Kneedle Backend",
        "model": "gemma4:e4b",
        "message": "গেটমিত্র সার্ভার চালু আছে"
    }


def handle_test_ollama() -> dict:
    try:
        import ollama
        models = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]
        gemma_available = any("gemma4" in m for m in model_names)
        return {
            "status": "ok" if gemma_available else "model_missing",
            "available_models": model_names,
            "gemma4_ready": gemma_available
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
