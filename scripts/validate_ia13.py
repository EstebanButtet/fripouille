"""FRP-IA-13 : essai explicite avec Ollama réel, dans une base jetable.

Usage : python scripts/validate_ia13.py --output <rapport.json>
Aucun lancement d'application externe, périphérique ou base personnelle.
Les souvenirs et preuves initiaux sont des fixtures annoncées comme telles.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch
from urllib.request import urlopen

from assistant_ia.application import build_default_runtime
from assistant_ia.learning.models import ExperienceProvenance
from assistant_ia.learning.outcomes import ExperienceOutcome
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.context import ActivePersonContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"milestone": "FRP-IA-13", "model": "qwen3.5:9b", "database": "temporary",
              "fixtures": "Fictional person, memory and explicitly confirmed reported experience.",
              "turns": [], "requests": []}

    def measured_urlopen(request, timeout):
        payload = json.loads(request.data)
        response = urlopen(request, timeout=timeout)

        class MeasuredResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                response.close()

            def read(self):
                raw = response.read()
                data = json.loads(raw)
                report["requests"].append({
                    "phase": "interpretation" if "format" in payload else "conversation",
                    "characters": sum(len(m["content"]) for m in payload["messages"]),
                    "num_ctx": payload["options"]["num_ctx"],
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                    "done_reason": data.get("done_reason"),
                })
                return raw
        return MeasuredResponse()

    with TemporaryDirectory(prefix="frp-ia-13-") as directory:
        db = SQLiteDatabase(Path(directory) / "validation.db")
        db.initialize()
        person = PersonRepository(db).create_person("Testeur FRP IA")
        runtime = build_default_runtime(database=db, person_context=ActivePersonContext(
            assistant_name="Fripouille", default_person=person))
        core = runtime.assistant
        learning = core.behavioral_learning_service
        memory_repo = MemoryRepository(db)
        memory = memory_repo.save_memory("La maquette Orion du testeur est construite en carton bleu.")
        memory_repo.link_person(memory.id, person.id)
        fixture_attempt = learning.begin_active_person_attempt(context="explication mécanique", objective="comprendre", strategy="Commencer par une analogie concrète.")
        fixture = learning.record_active_person_outcome(fixture_attempt, ExperienceOutcome("success", "reported_result", "Fixture de validation, pas un essai humain."),
                  provenance=ExperienceProvenance(source_type="manual_entry", source_reference="FRP-IA-13 validation fixture"))
        candidate = learning.propose_active_person_lesson(source_experiences=(fixture,), context_pattern="explication mécanique", proposed_strategy="Commencer par une analogie concrète.", rationale="Fixture de validation explicite.")
        rule = learning.confirm_active_person_lesson(candidate, "Confirmation de fixture de test logiciel.")

        def turn(label, message):
            start = monotonic()
            entry = {"label": label, "message": message}
            try:
                entry["response"] = runtime.process_message(message)
                action = core.last_action_result
                trace = core.last_cognitive_trace
                entry.update(intent=core.last_intent.name if core.last_intent else None,
                             action_status=action.status if action else None,
                             attempted=action.attempted if action else None,
                             state=core.internal_state.snapshot.guidance,
                             role=core.roles.active_id,
                             memory_selected=bool(trace and memory.id in trace.memory_ids),
                             rule_selected=bool(trace and rule.id in trace.cognitive.rule_ids),
                             state_in_prompt=trace.cognitive.state_guidance if trace else None)
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
            entry["elapsed_seconds"] = round(monotonic() - start, 2)
            report["turns"].append(entry)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            return entry

        with patch("assistant_ia.intelligence.model_client.urlopen", measured_urlopen):
            turn("conversation_person_memory", "Que sais-tu de ma maquette Orion ? Réponds en une phrase.")
            turn("confirmed_rule", "Donne une explication de mécanique : à quoi sert un levier ? Deux phrases.")
            attempt = learning.begin_active_person_attempt(context="validation tâche", objective="créer", strategy="registre")
            entry = turn("authorized_action", "Crée une tâche intitulée Vérification FRP IA temporaire.")
            if core.last_action_result is not None and core.last_action_result.attempted:
                experience = learning.record_active_person_action_result(attempt, core.last_action_result)
                report["verified_experience"] = experience.outcome_status
            turn("role_activation", "/role guide")
            turn("failed_action", "Termine la tâche numéro 999999.")
            turn("role_and_state", "Explique-moi simplement comment fonctionne un engrenage, en deux phrases.")
        with db.connect() as connection:
            report["schema_version"] = connection.execute("SELECT version FROM schema_version").fetchone()[0]
            report["foreign_key_check"] = connection.execute("PRAGMA foreign_key_check").fetchall()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
