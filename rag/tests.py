"""
Run with:  python manage.py test
Needs no real MongoDB, internet or API key: MongoDB is mongomock, embeddings are a
deterministic hashing embedder, and the LLM is faked.
"""
import json
import re
import tempfile
import zlib
from unittest.mock import patch

import mongomock
import numpy as np
from bson import ObjectId
from django.test import Client, SimpleTestCase, override_settings

from rag.management.commands.seed_sample_data import SAMPLE
from rag.services import (document_converter, embedding_service, embedding_store, ingestion_service,
                          llm_service, rag_service, relationships, retrieval, structured_query, vector_index)
from rag.services.exceptions import InvalidRequestError, VectorStoreError
from rag.services.retrieval import rrf_fuse


def fake_embed_texts(texts, batch_size=64):
    """Bag-of-words hashing embedder: texts sharing words get higher cosine similarity."""
    out = np.zeros((len(texts), 128), dtype="float32")
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if re.fullmatch(r"[0-9a-f]{24}", token):      # random ObjectIds are noise: keep the fake embedder deterministic
                continue
            out[i, zlib.crc32(token.encode()) % 128] += 1
        out[i] /= np.linalg.norm(out[i]) or 1
    return out


def _mongomock_add_replace_shim():
    """Newer pymongo passes `sort=` to the bulk builder; older mongomock doesn't accept it (test-only gap)."""
    from mongomock.collection import BulkOperationBuilder

    original = BulkOperationBuilder.add_replace

    def add_replace(self, selector, replacement, upsert=False, collation=None, hint=None, sort=None):
        return original(self, selector, replacement, upsert=upsert, collation=collation, hint=hint)

    return patch.object(BulkOperationBuilder, "add_replace", add_replace)


class MongoCase(SimpleTestCase):
    """Base: fresh in-memory MongoDB with the sample data, fake embedder, temp index dir."""

    structured = False     # exact-query path off by default so the similarity-search tests stay focused

    def setUp(self):
        self.db = mongomock.MongoClient()["testdb"]
        for name, docs in SAMPLE.items():
            self.db[name].insert_many([dict(d) for d in docs])
        tmp = tempfile.mkdtemp()
        patches = [
            patch("rag.services.mongo_service.get_db", return_value=self.db),
            patch("rag.services.mongo_service.get_embeddings_collection", return_value=self.db["rag_embeddings"]),
            patch("rag.services.mongo_service.ping", return_value=True),
            patch("rag.services.embedding_service.embed_texts", side_effect=fake_embed_texts),
            _mongomock_add_replace_shim(),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        overrides = override_settings(VECTOR_STORE_DIR=tmp, MIN_SCORE=0.0, EMBEDDING_PROVIDER="local",
                                      STRUCTURED_QUERIES=self.structured)
        overrides.enable()
        self.addCleanup(overrides.disable)
        vector_index.reset_for_tests()
        structured_query.reset_for_tests()
        embedding_service.clear_query_cache()
        rag_service._answer_cache.clear()

    def ingest(self, **kwargs):
        return ingestion_service.run_ingestion(log=lambda m: None, **kwargs)

    def emb(self, source, doc_filter=None):
        return self.db["rag_embeddings"].find_one({"_id": {"$regex": f"^{source}:"}, **(doc_filter or {})})


class ConverterTests(SimpleTestCase):
    def test_types_are_converted_to_readable_text(self):
        from datetime import datetime
        oid = ObjectId()
        doc = {"_id": ObjectId(), "name": "Ann", "tags": ["a", "b"], "active": True,
               "joined": datetime(2026, 8, 12, 6, 38), "owner": oid,
               "address": {"city": "Paris", "geo": {"lat": 1.5}},
               "password": "secret", "empty": "", "blob": b"x", "items": [{"sku": "A1"}, {"sku": "B2"}]}
        text = "\n".join(document_converter.convert_document("people", doc).lines)
        for expected in ("Name: Ann", "Tags: a, b", "Active: Yes", "Joined: August 12, 2026 06:38 UTC",
                         f"Owner ID: {oid}", "Address City: Paris", "Address Geo Lat: 1.5",
                         "Items 1 Sku: A1", "Items 2 Sku: B2"):
            self.assertIn(expected, text)
        for missing in ("secret", "Password", "Empty", "Blob"):
            self.assertNotIn(missing, text)

    def test_metadata_has_primary_id_collection_fields_and_references(self):
        oid, doc_id = ObjectId(), ObjectId()
        record = document_converter.to_records("things", {"_id": doc_id, "label": "X", "ref": oid})[0]
        meta = record["metadata"]
        self.assertEqual(record["_id"], f"things:{doc_id}:0")
        self.assertEqual((meta["collection"], meta["primary_id"]), ("things", str(doc_id)))
        self.assertIn("label", meta["field_names"])
        self.assertEqual(meta["reference_ids"], [str(oid)])

    def test_long_text_is_chunked_and_nothing_is_lost(self):
        words = " ".join(f"w{i}" for i in range(1500)) + " THE_END"
        records = document_converter.to_records("docs", {"_id": "d1", "name": "Report", "body": words})
        self.assertGreater(len(records), 5)
        self.assertTrue(all(len(r["text"]) < 1000 for r in records))
        self.assertIn("THE_END", records[-1]["text"])
        self.assertIn("(Report)", records[1]["text"])
        self.assertEqual(len({r["_id"] for r in records}), len(records))

    def test_empty_document_gives_no_records(self):
        self.assertEqual(document_converter.to_records("c", {"_id": ObjectId()}), [])


class RelationshipTests(MongoCase):
    def test_guess_collections(self):
        names = ["users", "workspaces", "categories", "addresses"]
        g = relationships.guess_collections
        self.assertEqual(g("userId", names), ["users"])
        self.assertEqual(g("workspace", names), ["workspaces"])
        self.assertEqual(g("category_id", names), ["categories"])
        self.assertEqual(g("address", names), ["addresses"])
        self.assertEqual(g("createdBy", names), [])

    def test_detects_targets_by_name_and_by_probing(self):
        resolver = relationships.RelationshipResolver(self.db, list(SAMPLE))
        resolver.prefetch("memberships", list(self.db["memberships"].find()))
        resolver.prefetch("projects", list(self.db["projects"].find()))
        resolver.prefetch("tasks", list(self.db["tasks"].find()))
        t = resolver._targets
        self.assertEqual(t[("memberships", "user")], "users")
        self.assertEqual(t[("memberships", "workspace")], "workspaces")
        self.assertEqual(t[("projects", "workspaceId")], "workspaces")
        self.assertEqual(t[("projects", "createdBy")], "users")       # found by probing
        self.assertEqual(t[("tasks", "assignedTo")], "users")         # found by probing
        self.assertEqual(t[("tasks", "projectId")], "projects")

    def test_string_ids_are_treated_as_references_when_they_resolve(self):
        self.db["notes"].insert_one({"text": "hello", "author": str(ObjectId("b00000000000000000000001"))})
        resolver = relationships.RelationshipResolver(self.db, list(SAMPLE) + ["notes"])
        doc = self.db["notes"].find_one()
        resolver.prefetch("notes", [doc])
        text = "\n".join(document_converter.convert_document("notes", doc, resolver).lines)
        self.assertIn("Author: Aisha Khan", text)


class IngestionTests(MongoCase):
    def test_ingest_stores_text_metadata_relationships_and_vector_together(self):
        summary = self.ingest()
        self.assertEqual(summary["mode"], "incremental")
        doc = self.emb("memberships")
        self.assertEqual(len(doc["embedding"]), 128)
        self.assertIn("User: Aisha Khan", self.db["rag_embeddings"].find_one(
            {"_id": f"memberships:{self.db['memberships'].find_one()['_id']}:0"})["text"])
        self.assertIn("Workspace: Acme Corp", doc["text"])
        self.assertEqual({r["ref_collection"] for r in doc["relationships"]}, {"users", "workspaces"})
        self.assertEqual(doc["metadata"]["model"], "local:sentence-transformers/all-MiniLM-L6-v2")
        for rel in doc["relationships"]:
            self.assertEqual(set(rel), {"field", "ref_id", "ref_collection", "ref_name"})

    def test_sensitive_fields_never_reach_embeddings(self):
        self.ingest()
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "notarealhash|Password"}}), 0)

    def test_second_run_embeds_nothing_then_only_changes_and_deletions(self):
        first = self.ingest()
        total = sum(s["embedded"] for s in first["collections"].values())
        self.assertGreater(total, 10)
        second = self.ingest()
        self.assertEqual(sum(s["embedded"] for s in second["collections"].values()), 0)

        self.db["projects"].update_one({"project": "ShopEase"}, {"$set": {"status": "completed"}})
        self.db["tasks"].delete_one({"title": "Fix login bug"})
        third = self.ingest()
        self.assertEqual(third["collections"]["projects"]["embedded"], 1)
        self.assertEqual(third["collections"]["tasks"]["removed"], 1)
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "Fix login bug"}}), 0)

    def test_renaming_a_referenced_document_refreshes_the_documents_that_point_to_it(self):
        self.ingest()
        self.db["workspaces"].update_one({"name": "Acme Corp"}, {"$set": {"name": "Acme Global"}})
        summary = self.ingest()
        self.assertGreaterEqual(summary["collections"]["memberships"]["embedded"], 1)
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "Acme Global"}}) > 0, True)

    def test_full_mode_reembeds_everything(self):
        self.ingest()
        again = self.ingest(full=True)
        self.assertEqual(sum(s["unchanged"] for s in again["collections"].values()), 0)

    def test_unknown_collection_is_a_clean_error(self):
        from rag.services.exceptions import IngestionError
        with self.assertRaises(IngestionError):
            self.ingest(collections=["nope"])

    def test_embeddings_collection_is_never_ingested_itself(self):
        self.ingest()
        self.assertNotIn("rag_embeddings", ingestion_service.mongo_service.list_collections())


class SearchTests(MongoCase):
    def setUp(self):
        super().setUp()
        self.ingest()

    def test_vector_search_finds_the_right_record(self):
        hits = retrieval.search("Health4Travel completed", top_k=3, hybrid=False)
        self.assertTrue(any(h.collection == "projects" and h.display_name == "Health4Travel" for h in hits))
        self.assertTrue(all(h.vector_score is not None for h in hits))

    def test_collection_filter(self):
        hits = retrieval.search("Acme Corp", top_k=5, collections=["workspaces"], hybrid=False)
        self.assertTrue(hits and all(h.collection == "workspaces" for h in hits))

    def test_keyword_failure_falls_back_to_vector_search(self):
        hits = retrieval.search("Health4Travel completed", top_k=3, hybrid=True)  # mongomock has no $text
        self.assertTrue(hits)

    def test_rrf_fusion_rewards_agreement(self):
        fused = rrf_fuse([("a", .9), ("b", .8)], [("b", 5.0), ("c", 4.0)], k=3, keyword_weight=1.0, rrf_k=60)
        self.assertEqual(fused[0][0], "b")
        self.assertEqual({i for i, _ in fused}, {"a", "b", "c"})

    def test_multi_hop_follows_relationships_forward_and_reverse(self):
        hits = [h for h in retrieval.search("Rohan Mehta backend developer", top_k=8, hybrid=False)
                if h.collection == "users"][:1]
        self.assertEqual(hits[0].display_name, "Rohan Mehta")
        related = retrieval.expand(hits, hops=1, max_related=10)
        self.assertTrue(any(r.collection == "tasks" and r.via["direction"] == "reverse" for r in related))

        task = [h for h in retrieval.search("Fix login bug", top_k=8, hybrid=False) if h.collection == "tasks"][0]
        forward = {(r.collection, r.display_name) for r in retrieval.expand([task], hops=1, max_related=10)
                   if r.via["direction"] == "forward"}
        self.assertIn(("projects", "ShopEase"), forward)
        self.assertIn(("users", "Rohan Mehta"), forward)

    def test_two_hops_reach_the_workspace_through_the_project(self):
        task = [h for h in retrieval.search("Fix login bug", top_k=8, hybrid=False) if h.collection == "tasks"][0]
        related = retrieval.expand([task], hops=2, max_related=10)
        self.assertIn("workspaces", {r.collection for r in related})


class AskTests(MongoCase):
    def setUp(self):
        super().setUp()
        self.ingest()

    def test_ask_sends_context_with_related_records_and_returns_sources(self):
        with patch.object(llm_service, "generate", return_value="ShopEase is in progress.") as gen:
            result = rag_service.ask("What is the status of ShopEase?", hops=1)
        self.assertEqual(result["answer"], "ShopEase is in progress.")
        context = gen.call_args[0][1]
        self.assertIn("ShopEase", context)
        self.assertIn("[1]", context)
        self.assertTrue(any(s["collection"] == "projects" for s in result["sources"]))
        self.assertIn("search", result["timings_ms"])

    def test_not_found_answer_has_no_sources(self):
        with patch.object(llm_service, "generate", return_value=llm_service.NOT_FOUND_ANSWER):
            self.assertEqual(rag_service.ask("What is the CEO's phone number?")["sources"], [])

    def test_answers_are_cached_until_the_index_changes(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            rag_service.ask("Which projects are completed?")
            second = rag_service.ask("which   projects are COMPLETED?")
            self.assertTrue(second["cached"])
            self.assertEqual(gen.call_count, 1)
            self.db["projects"].update_one({"project": "ShopEase"}, {"$set": {"status": "paused"}})
            self.ingest()
            rag_service.ask("Which projects are completed?")
            self.assertEqual(gen.call_count, 2)

    @override_settings(MAX_CONTEXT_CHARS=300)
    def test_context_respects_the_character_budget(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            rag_service.ask("Tell me about tasks and projects", top_k=10)
        self.assertLess(len(gen.call_args[0][1]), 900)

    def test_list_question_detection(self):
        for yes in ("List all tasks", "which projects have been finished?", "How many tasks are there?",
                    "Show me medium priority tasks", "What are the open tasks", "every user in Acme"):
            self.assertTrue(rag_service.is_list_question(yes), yes)
        for no in ("What is the status of ShopEase?", "Who is the manager of Acme?", "which is the best project"):
            self.assertFalse(rag_service.is_list_question(no), no)

    def test_list_questions_retrieve_more_than_the_selected_mode(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            wide = rag_service.ask("List all records", top_k=2, hops=0)
            narrow = rag_service.ask("What is the status of ShopEase?", top_k=2, hops=0)
        self.assertGreater(len(wide["sources"]), 5)
        self.assertTrue(wide["list_mode"])
        self.assertEqual(wide["records_used"], len(wide["sources"]))
        self.assertLessEqual(len(narrow["sources"]), 2)
        self.assertFalse(narrow["list_mode"])
        self.assertEqual(gen.call_args_list[0].kwargs["max_tokens"], 2048)
        self.assertIsNone(gen.call_args_list[1].kwargs["max_tokens"])

    @override_settings(LIST_TOP_K=0)
    def test_list_widening_can_be_turned_off(self):
        with patch.object(llm_service, "generate", return_value="ok"):
            self.assertLessEqual(len(rag_service.ask("List all records", top_k=2, hops=0)["sources"]), 2)

    def test_validation(self):
        for bad in ("  ", "x" * 501, 5):
            with self.assertRaises(InvalidRequestError):
                rag_service.validate_question(bad)

    def test_missing_index_is_a_clean_error(self):
        self.db["rag_embeddings"].delete_many({})
        vector_index.reset_for_tests()
        import shutil
        from django.conf import settings
        shutil.rmtree(settings.VECTOR_STORE_DIR, ignore_errors=True)
        with self.assertRaises(VectorStoreError):
            rag_service.ask("anything at all")

    @override_settings(LLM_API_KEY="")
    def test_missing_llm_key_is_a_clean_error(self):
        from rag.services.exceptions import ConfigError
        llm_service._client = None
        with self.assertRaises(ConfigError):
            rag_service.ask("Which projects are completed?")


class ApiTests(MongoCase):
    def setUp(self):
        super().setUp()
        self.client = Client()

    def post(self, url, payload, **extra):
        return self.client.post(url, json.dumps(payload), content_type="application/json", **extra)

    def test_index_page(self):
        self.assertContains(self.client.get("/"), "Ask your database")

    def test_ask_endpoint_end_to_end(self):
        self.ingest()
        with patch.object(llm_service, "generate", return_value="Health4Travel is completed."):
            response = self.post("/api/ask/", {"question": "Which projects have been finished?"})
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["answer"], "Health4Travel is completed.")
        self.assertTrue(data["sources"])

    def test_search_endpoint(self):
        self.ingest()
        data = self.post("/api/search/", {"query": "Acme Corp workspace", "top_k": 3, "hybrid": False,
                                          "expand": True}).json()
        self.assertLessEqual(len(data["results"]), 3)
        self.assertIn("related", data)
        self.assertIn("relationships", data["results"][0])

    def test_bad_input_returns_json_errors(self):
        self.assertEqual(self.client.post("/api/ask/", "nope", content_type="application/json").status_code, 400)
        self.assertEqual(self.post("/api/ask/", {"question": "hello", "top_k": 999}).status_code, 400)
        self.assertEqual(self.post("/api/search/", {"query": "x", "collections": "users"}).status_code, 400)
        self.assertEqual(self.client.get("/api/ask/").status_code, 405)

    def test_ask_before_ingest_explains_what_to_do(self):
        response = self.post("/api/ask/", {"question": "Which projects are completed?"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("ingest", response.json()["error"])

    @override_settings(INGEST_TOKEN="s3cret")
    def test_ingest_requires_the_token(self):
        self.assertEqual(self.post("/api/ingest/", {}).status_code, 401)
        self.assertEqual(self.client.get("/api/ingest/", HTTP_AUTHORIZATION="Bearer wrong").status_code, 401)
        with patch("rag.views.ingestion_service.start_job", return_value={"id": "x", "state": "running"}):
            response = self.post("/api/ingest/", {"full": True}, HTTP_AUTHORIZATION="Bearer s3cret")
        self.assertEqual(response.status_code, 202)

    @override_settings(INGEST_TOKEN="", DEBUG=False)
    def test_ingest_is_disabled_without_a_token_in_production(self):
        self.assertEqual(self.post("/api/ingest/", {}).status_code, 401)

    def test_health(self):
        self.ingest()
        data = self.client.get("/api/health/").json()
        self.assertEqual(data["status"], "ok")
        self.assertGreater(data["embeddings"], 0)
        self.assertTrue(data["index"]["ready"])


DEPARTMENTS = ["Engineering", "Human Resources", "Finance", "Marketing", "Sales", "Legal", "Operations", "Support",
               "Design", "Data Science", "Security", "Procurement", "Facilities", "Training", "Research",
               "Quality Assurance", "Logistics", "Public Relations", "Compliance", "Administration"]


class StructuredQueryTests(MongoCase):
    """Exact answers for list/count questions (the '20 departments but I only see 6, 10, 8, 13' problem)."""

    structured = True

    def setUp(self):
        super().setUp()
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])

    def ask(self, question, plan=None, **kw):
        with patch.object(llm_service, "plan_query", return_value=plan), \
             patch.object(llm_service, "generate", return_value="ok") as gen:
            result = rag_service.ask(question, **kw)
        return result, gen

    def test_all_20_departments_are_listed_every_time_even_without_the_planner(self):
        for question in ("List all departments", "Show me the department names", "What are our departments?",
                         "how many departments do we have"):
            result, gen = self.ask(question)                       # planner returns nothing -> name-match fallback
            self.assertEqual(result["mode"], "structured", question)
            self.assertEqual(result["total_matching"], 20, question)
            context = gen.call_args[0][1]
            self.assertIn("Exact number of records in the collection: 20", context)
            self.assertIn("the list is complete", context)
            for name in DEPARTMENTS:
                self.assertIn(name, context)
            self.assertNotIn("Document ID", context)
            self.assertEqual(len(result["sources"]), 20)

    def test_planner_filters_give_exact_counts_and_lists(self):
        plan = {"collection": "tasks", "operation": "list",
                "filters": [{"field": "priority", "op": "eq", "value": "MEDIUM"}]}      # case-insensitive
        result, gen = self.ask("Show me all medium priority tasks", plan)
        self.assertEqual(result["total_matching"], 2)
        self.assertEqual(result["conditions"], "priority = MEDIUM")
        context = gen.call_args[0][1]
        self.assertIn("Design landing page", context)
        self.assertIn("Fix login bug", context)
        self.assertNotIn("Set up payment gateway", context)
        self.assertIn("Exact number of matching records in the database: 2", context)

    def test_filter_by_a_persons_name_on_a_reference_field(self):
        plan = {"collection": "tasks", "operation": "list",
                "filters": [{"field": "assignedTo", "op": "eq", "value": "Rohan Mehta"}]}
        result, gen = self.ask("What tasks are assigned to Rohan Mehta?", plan)
        self.assertEqual(result["total_matching"], 2)
        context = gen.call_args[0][1]
        self.assertIn("Fix login bug", context)
        self.assertIn("Assigned To: Rohan Mehta", context)            # names are shown, not ids
        self.assertNotIn("Write release notes", context)

    def test_count_operation_sends_only_a_few_examples(self):
        plan = {"collection": "departments", "operation": "count", "filters": []}
        result, gen = self.ask("How many departments are there?", plan)
        self.assertEqual(result["total_matching"], 20)
        context = gen.call_args[0][1]
        self.assertIn("Exact number of matching records in the database: 20", context)
        self.assertIn("Only 10 examples are listed", context)
        self.assertEqual(result["records_used"], 10)

    @override_settings(STRUCT_MAX_RECORDS=3)
    def test_big_results_are_cut_but_the_total_stays_exact(self):
        plan = {"collection": "departments", "operation": "list", "filters": []}
        result, gen = self.ask("List all departments", plan)
        context = gen.call_args[0][1]
        self.assertEqual(result["total_matching"], 20)
        self.assertIn("Only the first 3 of 20", context)
        self.assertIn("NOT complete", context)

    def test_a_condition_is_never_silently_dropped(self):
        bad_field = {"collection": "tasks", "operation": "list",
                     "filters": [{"field": "colour", "op": "eq", "value": "red"}]}
        self.assertIsNone(structured_query.validate_plan(bad_field, structured_query.build_catalog()))
        bad_op = {"collection": "tasks", "filters": [{"field": "priority", "op": "$where", "value": "1"}]}
        self.assertIsNone(structured_query.validate_plan(bad_op, structured_query.build_catalog()))
        bad_num = {"collection": "tasks", "filters": [{"field": "priority", "op": "gt", "value": "abc"}]}
        self.assertIsNone(structured_query.validate_plan(bad_num, structured_query.build_catalog()))
        nested = {"collection": "tasks", "filters": [{"field": "priority", "op": "eq", "value": {"$ne": 1}}]}
        self.assertIsNone(structured_query.validate_plan(nested, structured_query.build_catalog()))
        unknown = {"collection": "nope", "filters": []}
        self.assertIsNone(structured_query.validate_plan(unknown, structured_query.build_catalog()))

    def test_invalid_plan_on_a_small_collection_reads_everything_and_says_conditions_were_not_applied(self):
        bad = {"collection": "tasks", "operation": "list", "filters": [{"field": "colour", "op": "eq", "value": "red"}]}
        result, gen = self.ask("Show me the tasks with a red label", bad)
        context = gen.call_args[0][1]
        self.assertEqual(result["mode"], "structured")
        self.assertIn("Conditions were NOT applied by the system", context)

    def test_zero_matches_fall_back_to_similarity_search(self):
        self.ingest()
        plan = {"collection": "tasks", "operation": "list", "filters": [{"field": "priority", "op": "eq", "value": "urgent"}]}
        result, _ = self.ask("Show me all urgent tasks", plan)
        self.assertNotEqual(result.get("mode"), "structured")

    def test_sensitive_fields_never_reach_the_llm(self):
        plan = {"collection": "users", "operation": "list", "filters": []}
        _, gen = self.ask("List all users", plan)
        context = gen.call_args[0][1]
        self.assertIn("Aisha Khan", context)
        self.assertNotIn("notarealhash", context)

    def test_ordinary_questions_do_not_use_the_exact_path(self):
        self.ingest()
        with patch.object(structured_query, "try_structured") as spy, \
             patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("What is the status of ShopEase?")
        spy.assert_not_called()

    @override_settings(STRUCTURED_QUERIES=False)
    def test_can_be_turned_off(self):
        self.ingest()
        with patch.object(structured_query, "try_structured") as spy, patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("List all departments")
        spy.assert_not_called()

    def test_collection_filter_from_the_api_skips_the_exact_path(self):
        self.ingest()
        with patch.object(structured_query, "try_structured") as spy, patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("List all departments", collections=["departments"])
        spy.assert_not_called()

    def test_numeric_comparison_and_in_operator(self):
        plan = {"collection": "departments", "operation": "list", "filters": [{"field": "floor", "op": "gte", "value": 3}]}
        result, _ = self.ask("Which departments are on floor 3 or above?", plan)
        self.assertEqual(result["total_matching"], 8)                 # floors 3 and 4: 4 + 4
        plan = {"collection": "departments", "operation": "list",
                "filters": [{"field": "name", "op": "in", "value": ["Finance", "legal"]}]}
        result, _ = self.ask("List the Finance and Legal departments", plan)
        self.assertEqual(result["total_matching"], 2)

    def test_planner_output_parsing(self):
        parse = llm_service._parse_json
        self.assertEqual(parse('Sure! {"collection": "tasks"} done'), {"collection": "tasks"})
        self.assertIsNone(parse("no json here"))
        self.assertIsNone(parse("{broken"))

    def test_llm_failure_in_planner_still_gives_the_complete_list(self):
        from rag.services.exceptions import LLMError
        with patch.object(llm_service, "plan_query", side_effect=LLMError()), \
             patch.object(llm_service, "generate", return_value="ok") as gen:
            result = rag_service.ask("List all departments")
        self.assertEqual(result["total_matching"], 20)
        self.assertIn("Finance", gen.call_args[0][1])


class SpeedAndResilienceTests(MongoCase):
    """The exact path must stay fast and must never fail just because the language model does."""

    structured = True

    def setUp(self):
        super().setUp()
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])

    def ask(self, question, plan=None, generate="ok", **kw):
        with patch.object(llm_service, "plan_query", return_value=plan) as pq, \
             patch.object(llm_service, "generate", return_value=generate) as gen:
            return rag_service.ask(question, **kw), pq, gen

    # ---- the reported failure: "give me list of all departments" -> LLM error
    def test_llm_failure_still_returns_the_complete_list(self):
        from rag.services.exceptions import LLMError
        with patch.object(llm_service, "plan_query", return_value=None), \
             patch.object(llm_service, "generate", side_effect=LLMError()):
            result = rag_service.ask("give me list of all departments")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["total_matching"], 20)
        for name in DEPARTMENTS:
            self.assertIn(f"- {name}", result["answer"])
        self.assertIn("20 matching record(s)", result["answer"])
        self.assertEqual(len(result["sources"]), 20)

    def test_degraded_count_and_truncated_list(self):
        from rag.services.exceptions import LLMError
        plan = {"collection": "departments", "operation": "count", "filters": []}
        with patch.object(llm_service, "plan_query", return_value=plan), patch.object(llm_service, "generate", side_effect=LLMError()):
            self.assertIn("20 matching record(s)", rag_service.ask("How many departments are there?")["answer"])
        with override_settings(STRUCT_MAX_RECORDS=3), patch.object(llm_service, "plan_query", return_value=None), \
             patch.object(llm_service, "generate", side_effect=LLMError()):
            answer = rag_service.ask("List all departments")["answer"]
        self.assertIn("Showing the first 3 of 20", answer)

    def test_configuration_errors_are_not_hidden_by_the_fallback(self):
        from rag.services.exceptions import ConfigError
        with patch.object(llm_service, "plan_query", return_value=None), \
             patch.object(llm_service, "generate", side_effect=ConfigError("bad key")):
            with self.assertRaises(ConfigError):
                rag_service.ask("List all departments")

    # ---- speed
    def test_catalog_is_saved_and_reloaded_without_probing(self):
        from pathlib import Path
        from django.conf import settings
        structured_query.build_catalog()
        self.assertTrue((Path(settings.VECTOR_STORE_DIR) / "catalog.json").exists())
        structured_query._memory.clear()
        with patch.object(relationships.RelationshipResolver, "_infer", side_effect=AssertionError("probed")):
            catalog = structured_query.build_catalog()
        self.assertEqual(catalog["tasks"]["fields"]["priority"]["types"], {"str"})
        self.assertEqual(catalog["tasks"]["fields"]["assignedTo"]["ref"], "users")
        self.assertIn("departments", catalog)

    def test_ingest_writes_the_catalog(self):
        from pathlib import Path
        from django.conf import settings
        self.ingest()
        self.assertTrue((Path(settings.VECTOR_STORE_DIR) / "catalog.json").exists())

    def test_reference_filters_need_no_probing_at_question_time(self):
        structured_query.build_catalog()
        structured_query.reset_for_tests()       # simulate a server restart: only the saved file remains
        plan = {"collection": "tasks", "operation": "list",
                "filters": [{"field": "assignedTo", "op": "eq", "value": "Rohan Mehta"}]}
        with patch.object(relationships.RelationshipResolver, "_infer", side_effect=AssertionError("probed")):
            result, _, gen = self.ask("What tasks are assigned to Rohan Mehta?", plan)
        self.assertEqual(result["total_matching"], 2)
        self.assertIn("Assigned To: Rohan Mehta", gen.call_args[0][1])

    def test_the_plan_is_cached_so_the_planner_runs_once(self):
        plan = {"collection": "departments", "operation": "list", "filters": []}
        _, pq, _ = self.ask("List all departments", plan)
        with patch.object(llm_service, "plan_query", side_effect=AssertionError("planner called again")), \
             patch.object(llm_service, "generate", return_value="ok"):
            again = rag_service.ask("list   ALL departments")
        self.assertEqual(again["total_matching"], 20)
        self.assertEqual(pq.call_count, 1)

    def test_a_failed_planner_call_is_not_cached(self):
        from rag.services.exceptions import LLMError
        with patch.object(llm_service, "plan_query", side_effect=LLMError()), patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("List all departments")
        plan = {"collection": "departments", "operation": "count", "filters": []}
        result, pq, _ = self.ask("List all departments", plan)
        self.assertEqual(pq.call_count, 1)
        self.assertEqual(result["records_used"], 10)

    def test_show_limits_the_fields_sent_to_the_llm(self):
        plan = {"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}
        _, _, gen = self.ask("Give me the names of all departments", plan)
        context = gen.call_args[0][1]
        self.assertIn("[1] Name: Engineering", context)
        self.assertNotIn("Floor", context)
        plan["show"] = ["nonexistent"]
        structured_query._plans.clear()
        _, _, gen = self.ask("Give me the names of all departments again", plan)
        self.assertIn("Floor", gen.call_args[0][1])         # unknown "show" fields are ignored, not fatal

    def test_timings_show_where_the_time_goes(self):
        result, _, _ = self.ask("List all departments", {"collection": "departments", "filters": []})
        self.assertEqual(set(result["timings_ms"]), {"plan", "db", "llm"})


class LlmErrorMessageTests(SimpleTestCase):
    """Every failure of the provider becomes a specific, actionable message (and never a traceback)."""

    def raised_by(self, name, status=None, message="boom"):
        error = type(name, (Exception,), {})(message)
        error.status_code = status
        client = type("C", (), {})()
        client.chat = type("Ch", (), {})()
        client.chat.completions = type("Co", (), {"create": lambda self, **kw: (_ for _ in ()).throw(error)})()
        with patch.object(llm_service, "_get_client", return_value=client):
            try:
                llm_service._chat([{"role": "user", "content": "x"}], 10)
            except Exception as exc:
                return exc
        raise AssertionError("no error raised")

    def test_messages(self):
        from rag.services.exceptions import ConfigError, LLMError
        cases = [("AuthenticationError", 401, ConfigError, "API key was rejected"),
                 ("PermissionDeniedError", 403, ConfigError, "denied"),
                 ("NotFoundError", 404, ConfigError, "was not found"),
                 ("RateLimitError", 429, LLMError, "rate-limited"),
                 ("APIStatusError", 413, LLMError, "too large"),
                 ("APITimeoutError", None, LLMError, "took longer"),
                 ("APIConnectionError", None, LLMError, "Could not connect"),
                 ("InternalServerError", 503, LLMError, "temporary problem (HTTP 503)"),
                 ("BadRequestError", 400, LLMError, "rejected the request: boom"),
                 ("WeirdError", None, LLMError, "unavailable right now")]
        for name, status, kind, text in cases:
            error = self.raised_by(name, status)
            self.assertIsInstance(error, kind, name)
            self.assertIn(text, error.message, name)

    def test_reasoning_effort_is_dropped_if_the_provider_rejects_it(self):
        from types import SimpleNamespace
        calls = []

        class BadRequestError(Exception):
            pass

        class Completions:
            def create(self, **kw):
                calls.append(dict(kw))
                if "reasoning_effort" in kw:
                    raise BadRequestError("unknown parameter: reasoning_effort")
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch.object(llm_service, "_get_client", return_value=client), override_settings(LLM_REASONING_EFFORT="low"):
            self.assertEqual(llm_service._chat([{"role": "user", "content": "x"}], 10), "hi")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", calls[1])

    def test_planner_uses_its_own_model_when_configured(self):
        from types import SimpleNamespace
        seen = []

        class Completions:
            def create(self, **kw):
                seen.append(kw["model"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"collection": null}'))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch.object(llm_service, "_get_client", return_value=client), \
             override_settings(PLANNER_MODEL="small-fast", LLM_MODEL="big-slow"):
            llm_service.plan_query("q", "catalog")
            llm_service._chat([{"role": "user", "content": "x"}], 5)
        self.assertEqual(seen, ["small-fast", "big-slow"])


# =============================================================== Ollama (local model)
import httpx  # noqa: E402

OLLAMA = dict(LLM_PROVIDER="ollama", LLM_MODEL="gemma3:4b", OLLAMA_HOST="http://ollama.test:11434",
              OLLAMA_NUM_CTX=8192, OLLAMA_KEEP_ALIVE="30m", LLM_TIMEOUT=30, LLM_TEMPERATURE="0",
              LLM_MAX_TOKENS=600, LLM_API_KEY="")


def fake_ollama(handler):
    """Make llm_service talk to an in-process fake Ollama instead of the network."""
    def factory():
        return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    return patch.object(llm_service, "_http", factory)


def reply(content, **extra):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "done": True, **extra})


@override_settings(**OLLAMA)
class OllamaClientTests(SimpleTestCase):
    def test_request_uses_native_api_with_context_window_and_no_api_key(self):
        seen = {}

        def handler(request):
            seen["url"], seen["json"] = str(request.url), json.loads(request.content)
            return reply("  There are 20 departments.  ")

        with fake_ollama(handler):
            answer = llm_service.generate("How many?", "EXACT QUERY RESULT ...")
        body = seen["json"]
        self.assertEqual(answer, "There are 20 departments.")
        self.assertEqual(seen["url"], "http://ollama.test:11434/api/chat")
        self.assertEqual(body["model"], "gemma3:4b")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["options"], {"num_ctx": 8192, "num_predict": 600, "temperature": 0.0})
        self.assertNotIn("format", body)
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertIn("EXACT QUERY RESULT", body["messages"][0]["content"])
        self.assertLess(len(body["messages"][0]["content"]), 2200)        # compact prompt for a small model
        self.assertIn("Question: How many?", body["messages"][1]["content"])

    def test_max_tokens_override_is_passed_as_num_predict(self):
        seen = {}
        with fake_ollama(lambda r: (seen.update(json.loads(r.content)), reply("ok"))[1]):
            llm_service.generate("q", "ctx", max_tokens=900)
        self.assertEqual(seen["options"]["num_predict"], 900)

    def test_planner_asks_for_json_mode_and_parses_the_plan(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}')

        with fake_ollama(handler):
            plan = llm_service.plan_query("list departments", "- departments (20 records): name")
        self.assertEqual(seen["format"], "json")
        self.assertEqual(plan["collection"], "departments")
        self.assertIn("Catalog:", seen["messages"][1]["content"])

    def test_hosted_prompt_is_unchanged_for_other_providers(self):
        with override_settings(LLM_PROVIDER="openai"):
            self.assertIs(llm_service._system_prompt(), llm_service.SYSTEM_PROMPT)
        self.assertIs(llm_service._system_prompt(), llm_service.SYSTEM_PROMPT_COMPACT)
        self.assertIn(llm_service.NOT_FOUND_ANSWER, llm_service.SYSTEM_PROMPT_COMPACT)

    def test_failures_become_actionable_messages(self):
        from rag.services.exceptions import ConfigError, LLMError

        def down(request):
            raise httpx.ConnectError("refused")

        with fake_ollama(down), self.assertRaises(LLMError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("Could not reach Ollama at http://ollama.test:11434", cm.exception.message)
        self.assertIn("ollama serve", cm.exception.message)

        missing = lambda r: httpx.Response(404, json={"error": 'model "gemma3:4b" not found, try pulling it first'})
        with fake_ollama(missing), self.assertRaises(ConfigError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("ollama pull gemma3:4b", cm.exception.message)

        def slow(request):
            raise httpx.ReadTimeout("slow")

        with fake_ollama(slow), self.assertRaises(LLMError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("took longer than 30s", cm.exception.message)

        broken = lambda r: httpx.Response(500, json={"error": "model runner has unexpectedly stopped"})
        with fake_ollama(broken), self.assertRaises(LLMError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("HTTP 500", cm.exception.message)
        self.assertIn("unexpectedly stopped", cm.exception.message)

        with fake_ollama(lambda r: httpx.Response(200, text="not json")), self.assertRaises(LLMError):
            llm_service.generate("q", "c")

    def test_status_reports_running_and_model_installed(self):
        def tags(models):
            return patch("httpx.get", return_value=httpx.Response(200, json={"models": [{"name": m} for m in models]}))

        with tags(["gemma3:4b", "llama3:8b"]):
            self.assertEqual(llm_service.status(), {"provider": "ollama", "model": "gemma3:4b",
                                                    "reachable": True, "model_available": True})
        with tags(["llama3:8b"]):
            self.assertEqual(llm_service.status()["model_available"], False)
        with patch("httpx.get", side_effect=httpx.ConnectError("down")):
            info = llm_service.status()
        self.assertEqual((info["reachable"], info["model_available"]), (False, False))
        with override_settings(LLM_PROVIDER="openai"):
            self.assertEqual(llm_service.status(), {"provider": "openai", "model": "gemma3:4b"})

    def test_warm_up_loads_the_model_with_an_empty_message_list(self):
        seen = {}
        with fake_ollama(lambda r: (seen.update(json.loads(r.content)), reply(""))[1]):
            llm_service.warm_up()
        self.assertEqual((seen["model"], seen["messages"], seen["keep_alive"]), ("gemma3:4b", [], "30m"))
        with override_settings(LLM_PROVIDER="openai"), patch.object(llm_service, "_ollama_post") as post:
            llm_service.warm_up()
        post.assert_not_called()


class OllamaEndToEndTests(MongoCase):
    """Whole pipeline with a fake Ollama: planner (JSON mode) then answer, on the 20-departments case."""

    structured = True

    def setUp(self):
        super().setUp()
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])
        overrides = override_settings(**OLLAMA)
        overrides.enable()
        self.addCleanup(overrides.disable)
        self.requests = []

    def handler(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("format") == "json":
            return reply('{"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}')
        return reply("There are 20 departments, listed below.")

    def test_list_question_runs_planner_then_answer_with_all_names(self):
        with fake_ollama(self.handler):
            result = rag_service.ask("give me list of all departments")
        self.assertEqual(result["mode"], "structured")
        self.assertEqual(result["total_matching"], 20)
        self.assertFalse(result["degraded"])
        planner, answer = self.requests
        self.assertEqual(planner["format"], "json")
        user_prompt = answer["messages"][1]["content"]
        for name in DEPARTMENTS:
            self.assertIn(name, user_prompt)
        self.assertNotIn("Floor", user_prompt)                         # only the requested field was sent
        self.assertEqual(answer["options"]["num_ctx"], 8192)

    def test_when_ollama_is_down_the_complete_list_is_still_returned(self):
        def down(request):
            raise httpx.ConnectError("refused")

        with fake_ollama(down):
            result = rag_service.ask("give me list of all departments")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["total_matching"], 20)
        for name in DEPARTMENTS:
            self.assertIn(f"- {name}", result["answer"])

    @override_settings(STRUCT_PLANNER="off")
    def test_planner_can_be_switched_off_for_speed(self):
        with fake_ollama(self.handler):
            result = rag_service.ask("give me list of all departments")
        self.assertEqual(result["total_matching"], 20)
        self.assertEqual(len(self.requests), 1)                        # only the answer call, no planning call
        self.assertNotIn("format", self.requests[0])

    def test_health_reports_the_local_model(self):
        with patch("httpx.get", return_value=httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})):
            health = rag_service.health()
        self.assertEqual(health["llm"]["model_available"], True)
        self.assertEqual(health["llm_model"], "gemma3:4b")

    def test_catalog_text_shrinks_for_small_models(self):
        catalog = structured_query.build_catalog()
        full = structured_query.catalog_text(catalog, limit=100000)
        tiny = structured_query.catalog_text(catalog, limit=len(full) // 2)
        self.assertLessEqual(len(tiny), len(full) // 2)
        no_values = structured_query.catalog_text(catalog, limit=len(full) - 1)
        self.assertNotIn("[Engineering", no_values)


# =============================================================== Gemini (Google, OpenAI-compatible endpoint)
from io import StringIO  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from django.core.management import call_command  # noqa: E402
from django.core.management.base import CommandError  # noqa: E402

GEMINI = dict(LLM_PROVIDER="gemini", LLM_MODEL="gemini-3.5-flash-lite", LLM_API_KEY="SECRET-KEY-123",
              LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/", LLM_TEMPERATURE="0",
              LLM_MAX_TOKENS_PARAM="max_tokens", LLM_REASONING_EFFORT="", LLM_MAX_TOKENS=600)


def fake_openai_client(content="ok", finish_reason="stop", error=None, seen=None):
    """Stands in for the openai client; records the request fields."""
    class Completions:
        def create(self, **kw):
            if seen is not None:
                seen.append(kw)
            if error is not None:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                            message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def provider_error(name, status, message="boom"):
    error = type(name, (Exception,), {})(message)
    error.status_code = status
    return error


@override_settings(**GEMINI)
class GeminiTests(SimpleTestCase):
    def test_request_never_sends_temperature_and_uses_max_tokens(self):
        seen = []
        with patch.object(llm_service, "_get_client", return_value=fake_openai_client("Hello", seen=seen)):
            self.assertEqual(llm_service.generate("q", "ctx", max_tokens=900), "Hello")
            llm_service.plan_query("q", "catalog")
        for request in seen:
            self.assertNotIn("temperature", request)            # deprecated on Gemini 3.x, even though LLM_TEMPERATURE=0
            self.assertNotIn("max_completion_tokens", request)
            self.assertNotIn("reasoning_effort", request)       # default thinking level is the fastest
            self.assertEqual(request["model"], "gemini-3.5-flash-lite")
            self.assertEqual(request["messages"][-1]["role"], "user")   # a trailing model turn is rejected by Gemini 3.5
        self.assertEqual(seen[0]["max_tokens"], 900)

    def test_other_providers_still_send_temperature_and_max_completion_tokens(self):
        seen = []
        with override_settings(LLM_PROVIDER="openai", LLM_MAX_TOKENS_PARAM="max_completion_tokens"), \
             patch.object(llm_service, "_get_client", return_value=fake_openai_client("Hi", seen=seen)):
            llm_service.generate("q", "ctx")
        self.assertEqual(seen[0]["temperature"], 0.0)
        self.assertIn("max_completion_tokens", seen[0])

    def test_empty_replies_are_reported_not_shown_as_not_found(self):
        from rag.services.exceptions import LLMError
        cases = [("length", "ran out of output tokens"), ("content_filter", "safety filter"), ("stop", "empty answer")]
        for reason, text in cases:
            client = fake_openai_client(content=None, finish_reason=reason)
            with patch.object(llm_service, "_get_client", return_value=client), self.assertRaises(LLMError) as cm:
                llm_service.generate("q", "ctx")
            self.assertIn(text, cm.exception.message, reason)

    def test_misconfiguration_messages_name_the_endpoint(self):
        from rag.services.exceptions import ConfigError
        cases = [(provider_error("NotFoundError", 404), "was not found at generativelanguage.googleapis.com"),
                 (provider_error("BadRequestError", 400, "API key not valid. Please pass a valid API key."),
                  "rejected by generativelanguage.googleapis.com"),
                 (provider_error("AuthenticationError", 401), "rejected by generativelanguage.googleapis.com")]
        for error, text in cases:
            with patch.object(llm_service, "_get_client", return_value=fake_openai_client(error=error)), \
                 self.assertRaises(ConfigError) as cm:
                llm_service.generate("q", "ctx")
            self.assertIn(text, cm.exception.message)
            self.assertNotIn("SECRET-KEY-123", cm.exception.message)

    def test_a_groq_url_with_a_gemini_model_is_explained(self):
        from rag.services.exceptions import ConfigError
        error = provider_error("NotFoundError", 404, "The model `gemini-3.5-flash-lite` does not exist")
        with override_settings(LLM_BASE_URL="https://api.groq.com/openai/v1"), \
             patch.object(llm_service, "_get_client", return_value=fake_openai_client(error=error)), \
             self.assertRaises(ConfigError) as cm:
            llm_service.generate("q", "ctx")
        self.assertIn("api.groq.com", cm.exception.message)
        self.assertIn("LLM_BASE_URL belongs to the same provider", cm.exception.message)

    @override_settings(LLM_API_KEY="")
    def test_missing_key_message_mentions_gemini_key_name(self):
        from rag.services.exceptions import ConfigError
        llm_service._client = None
        with self.assertRaises(ConfigError) as cm:
            llm_service.generate("q", "ctx")
        self.assertIn("GEMINI_API_KEY", cm.exception.message)


@override_settings(**GEMINI)
class CheckLlmCommandTests(SimpleTestCase):
    def run_command(self, client):
        out = StringIO()
        with patch.object(llm_service, "_get_client", return_value=client):
            try:
                call_command("check_llm", stdout=out)
            except CommandError as exc:
                return out.getvalue(), exc
        return out.getvalue(), None

    def test_success_report_never_prints_the_key(self):
        client = fake_openai_client('{"collection": "departments", "operation": "list", "filters": []}')
        output, error = self.run_command(client)
        self.assertIsNone(error)
        self.assertIn("Provider        : gemini", output)
        self.assertIn("Model           : gemini-3.5-flash-lite", output)
        self.assertIn("generativelanguage.googleapis.com", output)
        self.assertIn("Temperature     : not sent", output)
        self.assertIn("OK      1/2", output)
        self.assertIn("OK      2/2", output)
        self.assertIn("configured correctly", output)
        self.assertNotIn("SECRET-KEY-123", output)

    def test_failure_report_says_what_is_wrong(self):
        client = fake_openai_client(error=provider_error("NotFoundError", 404))
        output, error = self.run_command(client)
        self.assertIsNotNone(error)
        self.assertIn("FAILED  1/2", output)
        self.assertIn("was not found at generativelanguage.googleapis.com", output)
        self.assertNotIn("SECRET-KEY-123", output)

    def test_plain_text_plan_is_a_warning_not_a_failure(self):
        output, error = self.run_command(fake_openai_client("sure, here you go"))
        self.assertIsNone(error)
        self.assertIn("WARNING 2/2", output)


class GeminiEndToEndTests(MongoCase):
    """Whole pipeline through the (fake) Gemini endpoint, on the 20-departments case."""

    structured = True

    def setUp(self):
        super().setUp()
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])
        overrides = override_settings(**GEMINI)
        overrides.enable()
        self.addCleanup(overrides.disable)

    def test_list_question_end_to_end(self):
        seen = []

        class Completions:
            def create(self, **kw):
                seen.append(kw)
                if "JSON query plan" in kw["messages"][0]["content"]:
                    content = '{"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}'
                else:
                    content = "There are 20 departments."
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch.object(llm_service, "_get_client", return_value=client):
            result = rag_service.ask("give me list of all departments")
        self.assertEqual(result["answer"], "There are 20 departments.")
        self.assertEqual(result["total_matching"], 20)
        self.assertEqual(len(seen), 2)                                   # planner + answer
        for request in seen:
            self.assertNotIn("temperature", request)
            self.assertIn("max_tokens", request)
        for name in DEPARTMENTS:
            self.assertIn(name, seen[1]["messages"][1]["content"])

    def test_gemini_outage_still_returns_the_complete_list(self):
        client = fake_openai_client(error=provider_error("InternalServerError", 503))
        with patch.object(llm_service, "_get_client", return_value=client):
            result = rag_service.ask("give me list of all departments")
        self.assertTrue(result["degraded"])
        for name in DEPARTMENTS:
            self.assertIn(f"- {name}", result["answer"])

    def test_an_invalid_key_is_not_hidden_by_the_fallback(self):
        from rag.services.exceptions import ConfigError
        client = fake_openai_client(error=provider_error("BadRequestError", 400, "API key not valid"))
        with patch.object(llm_service, "_get_client", return_value=client), self.assertRaises(ConfigError) as cm:
            rag_service.ask("give me list of all departments")
        self.assertIn("rejected by generativelanguage.googleapis.com", cm.exception.message)
