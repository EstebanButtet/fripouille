"""FRP-IA-14 : mesures reproductibles, sans assertions de performance.

Base jetable, fixtures publiques ; --ollama active deux tours réseau réels.
Sortie JSON explicite ; aucune lecture de la base personnelle.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import statistics
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
from urllib.request import urlopen

from assistant_ia.application import build_default_runtime
from assistant_ia.intelligence.model_client import OllamaModelClient
from assistant_ia.intelligence.prompt import build_conversation_prompt
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.learning.models import ExperienceProvenance
from assistant_ia.learning.outcomes import ExperienceOutcome


def measure(function, count):
    values = []
    for _ in range(count):
        start = perf_counter()
        function()
        values.append((perf_counter() - start) * 1000)
    return {"count": count, "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--ollama", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000:
        parser.error("iterations: 1..1000")
    report = {"milestone": "FRP-IA-14", "python": platform.python_version(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "fixture": "100 memories, 3 confirmed rules; temporary SQLite", "measurements": {}}
    measurements = report["measurements"]
    with TemporaryDirectory(prefix="frp-ia14-benchmark-") as directory:
        database = SQLiteDatabase(Path(directory) / "bench.db")
        runtime = build_default_runtime(database=database)
        core = runtime.assistant
        client = core._model_client
        learning = core.behavioral_learning_service
        for index in range(100):
            MemoryRepository(database).save_memory(f"Fixture explication mécanique maquette numéro {index}.")
        for index in range(3):
            attempt = learning.begin_active_person_attempt(context="explication mécanique", objective="comprendre", strategy=f"Exemple {index}.")
            experience = learning.record_active_person_outcome(attempt, ExperienceOutcome("success", "reported_result", "fixture"), provenance=ExperienceProvenance(source_type="manual_entry"))
            candidate = learning.propose_active_person_lesson(source_experiences=(experience,), context_pattern="explication mécanique", proposed_strategy=f"Exemple {index}.", rationale="fixture")
            learning.confirm_active_person_lesson(candidate, "confirmation de fixture")
        query = "Explique la mécanique de ma maquette en une phrase."
        person = core.person_context.active_person_id
        cognitive = client._cognitive_context_provider
        social = client._retrieve_social_context()
        memories = client._retrieve_contextual_memories(query)
        snapshot = cognitive.build(query, person)
        operations = {
            "cognitive_context": lambda: cognitive.build(query, person),
            "social_context_sqlite": client._retrieve_social_context,
            "memory_retrieval_sqlite": lambda: client._retrieve_contextual_memories(query),
            "rules_sqlite": lambda: learning.repository.list_rules(person_id=person, limit=50),
            "prompt": lambda: build_conversation_prompt(client._identity, client._person_context, client._capability_context, social_context=social, contextual_memories=memories, cognitive_context=snapshot),
        }
        for name, operation in operations.items():
            measurements[name] = measure(operation, args.iterations)
        def simulated_request(client, payload):
            if "format" in payload:
                return "fixture", json.dumps({"name": "conversation", "parameters": {}, "conversation": {"mode": "standard", "target_text": None}})
            return "fixture", "Une maquette illustre le fonctionnement mécanique."
        with ExitStack() as stack:
            stack.enter_context(patch.object(OllamaModelClient, "_request_ollama", simulated_request))
            for name in ("memory_candidates", "profile_fact_candidates"):
                cls = "OllamaMemoryCandidateAnalyzer" if name == "memory_candidates" else "OllamaProfileFactCandidateAnalyzer"
                stack.enter_context(patch(f"assistant_ia.intelligence.{name}.{cls}._request_ollama", return_value='{"candidates": []}'))
            measurements["turn_without_llm"] = measure(lambda: runtime.process_message(query), args.iterations)
        if args.ollama:
            report["ollama_requests"] = []
            def measured_open(request, timeout):
                start = perf_counter()
                response = urlopen(request, timeout=timeout)
                class Response:
                    def __enter__(self): return self
                    def __exit__(self, *unused): response.close()
                    def read(self):
                        raw = response.read()
                        data, payload = json.loads(raw), json.loads(request.data)
                        entry = {"elapsed_ms": (perf_counter()-start)*1000,
                                 "num_ctx": payload.get("options", {}).get("num_ctx"),
                                 "input_characters": sum(len(m["content"]) for m in payload["messages"]),
                                 "structured": "format" in payload}
                        entry.update({key: data.get(key) for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "prompt_eval_count", "eval_count")})
                        report["ollama_requests"].append(entry)
                        print(json.dumps(entry), flush=True)
                        return raw
                return Response()
            runtime.reset_conversation()
            with ExitStack() as stack:
                for module in ("model_client", "memory_candidates", "profile_fact_candidates"):
                    stack.enter_context(patch(f"assistant_ia.intelligence.{module}.urlopen", measured_open))
                try:
                    measurements["real_turn"] = measure(lambda: runtime.process_message(query), 2)
                except Exception as error:
                    report["ollama_error"] = f"{type(error).__name__}: {error}"
            report["first_usable_output"] = "Non-streaming client: only complete validated response is usable."
    if platform.system() == "Windows":
        measurements["powershell_speech_initialization"] = measure(lambda: subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "Add-Type -AssemblyName System.Speech"],
            check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW), 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(measurements, indent=2), flush=True)


if __name__ == "__main__":
    main()
