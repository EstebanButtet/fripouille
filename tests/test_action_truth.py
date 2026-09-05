"""FRP-IA-14 : réponse libre et paramètres persistants non inventés."""
import json
import unittest
from unittest.mock import Mock, patch
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.intelligence.model_client import OllamaModelClient
from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.action_truth import CLARIFY_ACTION, grounded_action, truthful_conversation

BENCHMARK_TRANSCRIPT = "Il répond est aujourd'hui je voudrais que tu fais une tâche pour demain à 15h00 puis rappelle moins que le projet de reboisement correctement elle reste savoir si tu comprends bien"


class ActionTruthTests(unittest.TestCase):
    def test_reported_regression_never_claims_creation_or_keeps_arnaud(self):
        client=Mock()
        client.generate_response.return_value=ModelResponse("Je vais créer une tâche : Rappeler Arnaud concernant le projet de reboisement.", "test", Intent("conversation"))
        core=AssistantCore(model_client=client)
        self.assertEqual(core.process_message(BENCHMARK_TRANSCRIPT),CLARIFY_ACTION)
        self.assertIsNone(core.last_action_result)
        self.assertNotIn("Arnaud",core.context.messages[-1].content)

    def test_free_text_claims_are_filtered(self):
        for content in ("J'ai créé une tâche.", "Tâche ajoutée : appel.", "Je te rappellerai demain.", "C'est noté !"):
            self.assertEqual(truthful_conversation(content,"Bonjour"),CLARIFY_ACTION)

    def test_negative_and_explanation_remain(self):
        for content in ("Je n'ai créé aucune tâche.","Je peux créer une tâche si tu me le demandes.","Une pile produit du courant."):
            self.assertEqual(truthful_conversation(content,"Explique"),content)

    def test_invented_name_in_title_is_rejected_before_action(self):
        response=json.dumps({"name":"create_task","parameters":{"title":"Rappeler Arnaud"},"conversation":{"mode":"standard","target_text":None}})
        with patch.object(OllamaModelClient,"_request_ollama",return_value=("test",response)) as request:
            result=OllamaModelClient().generate_response((ConversationMessage("user","Crée une tâche pour rappeler mon projet."),))
        self.assertEqual(result.intent.name,"conversation")
        self.assertEqual(result.content,CLARIFY_ACTION)
        request.assert_called_once()

    def test_grounded_title_and_missing_date(self):
        self.assertTrue(grounded_action(Intent("create_task",{"title":"Appeler Arnaud"}),"Crée une tâche Appeler Arnaud"))
        self.assertFalse(grounded_action(Intent("create_task",{"title":"Appeler Arnaud","due_at":"2026-10-01"}),"Crée une tâche Appeler Arnaud"))
