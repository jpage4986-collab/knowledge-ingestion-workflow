import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from job_history import JobHistoryStore

class ActualRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = JobHistoryStore(Path(self.temp.name) / 'jobs.json')
        self.history = patch.object(server, 'HISTORY', self.store)
        self.history.start()
        self.jobs = patch.dict(server._ingestion_jobs, {}, clear=True)
        self.jobs.start()
        self.registry = patch.object(server, 'SANDBOX_REGISTRY_FILE', Path(self.temp.name) / 'sandbox-registry.json')
        self.registry.start()
        server._register_sandbox('new-sandbox')

    def tearDown(self):
        self.registry.stop()
        self.jobs.stop()
        self.history.stop()
        self.temp.cleanup()

    def dataset(self):
        return {'id': 'new-sandbox', 'name': 'HIH-SANDBOX-test',
                'description': '[knowledge-workbench-sandbox:v1]',
                'embedding_model': 'text-embedding-v4@Tongyi-Qianwen',
                'parser_config': {'layout_recognize': 'DeepDOC'}}

    def job(self, uploaded=True):
        return {'id': 'job-test', 'dataset_id': 'new-sandbox', 'mode': 'local',
                'status': 'running', 'stage': 'parse', 'created_at': 1,
                'updated_at': 2, 'parallelism': 3, 'local_file_count': 1,
                'items': [{'id': 'item1', 'title': 'demo.md', 'status': 'running',
                           'document_ids': ['doc1'] if uploaded else [], 'checks': {}}]}

    def test_actual_non_agnes_embedding_is_accepted(self):
        with patch.object(server, '_raw_datasets', return_value=[self.dataset()]):
            result = server._model_gate('new-sandbox')
        self.assertTrue(result['ready'])
        self.assertEqual(result['source'], 'ragflow_dataset_metadata')
        self.assertEqual(result['version'], 'text-embedding-v4@Tongyi-Qianwen')

    def test_missing_actual_embedding_does_not_accept_manual_confirmation(self):
        dataset = self.dataset()
        dataset['embedding_model'] = ''
        manual = Path(self.temp.name) / 'model-gate.json'
        manual.write_text(json.dumps({'confirmed': True, 'provider': 'agnes', 'version': 'fake'}))
        with patch.object(server, 'MODEL_GATE_FILE', manual), patch.object(server, '_raw_datasets', return_value=[dataset]):
            self.assertFalse(server._load_model_gate_file('new-sandbox')['ready'])

    def test_all_updates_persist(self):
        server._ingestion_jobs['job-test'] = self.job()
        server._update_job('job-test', message='saved')
        server._update_job_item('job-test', 'item1', progress=76)
        server._update_item_check('job-test', 'item1', 'parse', 'passed', 'done')
        saved = self.store.snapshot()['job-test']
        self.assertEqual(saved['message'], 'saved')
        self.assertEqual(saved['items'][0]['progress'], 76)
        self.assertEqual(saved['items'][0]['checks']['parse']['state'], 'passed')

    def test_restart_keeps_document_ids(self):
        self.store.record(self.job())
        recovered = self.store.load_interrupted()[0]
        self.assertEqual(recovered['status'], 'interrupted')
        self.assertEqual(recovered['items'][0]['document_ids'], ['doc1'])

    def test_polling_history_does_not_interrupt_active_task(self):
        job = self.job()
        self.store.record(job)
        server._ingestion_jobs[job['id']] = job
        self.assertEqual(server.restore_recoverable_jobs(), [])
        self.assertEqual(self.store.snapshot()[job['id']]['status'], 'running')

    def test_missing_local_bytes_rejected(self):
        result = server._safe_resume_job(self.job(uploaded=False))
        self.assertFalse(result['resumable'])
        self.assertIn('本地原始字节已丢失', result['message'])

    def test_partial_upload_does_not_report_all_items_completed(self):
        job = self.job()
        job['items'].append({'id': 'item2', 'status': 'queued', 'document_ids': [], 'checks': {}})
        result = server._safe_resume_job(job)
        self.assertTrue(result['resumable'])
        self.assertEqual(result['items'][1]['status'], 'needs_review')

    def test_unstarted_remote_document_remains_needs_review(self):
        server._ingestion_jobs['job-test'] = self.job()
        with patch.object(server, '_raw_datasets', return_value=[self.dataset()]), \
             patch.object(server, '_fetch_document_page', return_value=([{'id': 'doc1', 'run': 'UNSTART'}], 1)), \
             patch.object(server, '_trigger_parse') as parse:
            server._review_uploaded_documents('job-test', True)
        parse.assert_not_called()
        self.assertEqual(self.store.snapshot()['job-test']['status'], 'completed_with_review')

    def test_resume_schedules_actual_worker(self):
        job = self.job()
        job['status'] = 'interrupted'
        self.store.record(job)
        with patch.object(server, '_raw_datasets', return_value=[self.dataset()]), patch.object(server._ingestion_executor, 'submit') as submit:
            result = server._start_resume(job['id'])
        self.assertTrue(result['resumed'])
        submit.assert_called_once_with(server._review_uploaded_documents, job['id'], True)

    def test_resumed_review_reads_chunks_without_upload_or_parse(self):
        server._ingestion_jobs['job-test'] = self.job()
        docs = [{'id': 'doc1', 'run': 'DONE', 'chunk_count': 1}]
        with patch.object(server, '_raw_datasets', return_value=[self.dataset()]), \
             patch.object(server, '_fetch_document_page', return_value=(docs, 1)), \
             patch.object(server, 'get_document_chunks', return_value={'items': [{'content': 'actual source evidence ' * 8}]}) as chunks, \
             patch.object(server, '_upload_to_sandbox') as upload, patch.object(server, '_trigger_parse') as parse:
            server._review_uploaded_documents('job-test', True)
        upload.assert_not_called()
        parse.assert_not_called()
        chunks.assert_called_once()
        self.assertEqual(self.store.snapshot()['job-test']['status'], 'completed')

    def test_old_library_write_rejected(self):
        old = dict(self.dataset(), id='old-library', name='EM-论文与方法', description='formal')
        with patch.object(server, '_raw_datasets', return_value=[old]):
            with self.assertRaises(server.ServiceError) as caught:
                server._sandbox_dataset('old-library')
        self.assertEqual(caught.exception.status, 403)

    def test_chunk_page_is_not_bounding_box_coordinates(self):
        chunk = server._serialize_chunk({'id': 'c', 'content': 'evidence', 'positions': [[3, 159, 451, 123, 243]]})
        self.assertEqual(chunk['page'], 3)

    def test_uploaded_paper_keeps_original_online_source(self):
        job = self.job()
        job['items'][0].update(source='arxiv', pdf_url='https://arxiv.org/pdf/2506.16665v3')
        server._ingestion_jobs[job['id']] = job
        doc = server._serialize_document({'id': 'doc1', 'source_type': 'local'})
        self.assertEqual(doc['source_type'], 'arxiv')
        self.assertEqual(doc['source_url'], 'https://arxiv.org/pdf/2506.16665v3')

    def test_old_harness_marker_alone_does_not_authorize_writes(self):
        old = dict(self.dataset(), id='old-harness')
        with patch.object(server, '_raw_datasets', return_value=[old]):
            with self.assertRaises(server.ServiceError) as caught:
                server._sandbox_dataset('old-harness')
        self.assertEqual(caught.exception.status, 403)

    def test_new_sandbox_registration_persists(self):
        server._register_sandbox('owned-after-restart')
        self.assertIn('owned-after-restart', server._registered_sandbox_ids())

    def test_empty_source_search_is_not_reported_as_completed(self):
        job = dict(self.job(), mode='online', query='missing source', limit=1)
        server._ingestion_jobs[job['id']] = job
        with patch.object(server, '_raw_datasets', return_value=[self.dataset()]), \
             patch.object(server, '_all_documents', return_value=[]), \
             patch.object(server, 'discover_sources', return_value={'items': [], 'errors': []}):
            server._run_ingestion_job(job['id'], [])
        self.assertEqual(self.store.snapshot()[job['id']]['status'], 'failed')
        self.assertIn('未发现', self.store.snapshot()[job['id']]['message'])

if __name__ == '__main__':
    unittest.main()
