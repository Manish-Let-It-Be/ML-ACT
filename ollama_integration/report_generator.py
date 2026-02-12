import requests
import json


def generate_ai_report(dataset_info, algorithms, metrics_table, ollama_url="http://localhost:11434/api/generate", model="mistral"):
    prompt = f"""You are a Machine Learning expert. Analyze the following experiment results and provide a comprehensive report.

Dataset Information:
{dataset_info}

Algorithms Used:
{', '.join(algorithms)}

Performance Metrics:
{metrics_table}

Please provide:
1. Which algorithm performed best and why
2. Bias vs Variance analysis for each model
3. Overfitting detection analysis
4. Suggestions for improvement
5. Overall recommendation

Be specific with numbers and provide actionable insights."""

    try:
        response = requests.post(
            ollama_url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated."), None
        return None, f"Ollama returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Ollama is not running. Start it with: ollama serve"
    except requests.exceptions.Timeout:
        return None, "Ollama request timed out. Try a smaller model."
    except Exception as e:
        return None, str(e)


def check_ollama_status(ollama_url="http://localhost:11434"):
    try:
        r = requests.get(ollama_url, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models(ollama_url="http://localhost:11434"):
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []
