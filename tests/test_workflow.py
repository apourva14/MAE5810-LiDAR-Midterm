"""Exercise reproducible orchestration without rerunning expensive studies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from lidar_project.workflow import Task, Workflow, WorkflowError


class DeliverableWorkflow(unittest.TestCase):
    def fixture(self, root, *, fail=False):
        root = Path(root)
        (root / "worker.py").write_text(
            "from pathlib import Path\n"
            "import json,sys\n"
            "p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)\n"
            "p.write_text(json.dumps({'created':True}))\n"
            + ("raise SystemExit(7)\n" if fail else "print('complete')\n"), encoding="utf-8")
        source = root / "scientific_input.py"
        source.write_text("value = 1\n", encoding="utf-8")
        sample = Task("sample", "Example data", "data", (),
                      ("worker.py", "data with spaces/result.json"),
                      ("data with spaces/result.json",), ("scientific_input.py",), ("data with spaces",))
        view = Task("view", "Example media", "media", ("sample",),
                    ("worker.py", "outputs/view.json"), ("outputs/view.json",), ("worker.py",))
        return Workflow(root, python=sys.executable, tasks=(sample, view))

    def test_runs_from_any_cwd_and_reuses_unchanged_stages(self):
        with tempfile.TemporaryDirectory(prefix="lidar project ") as root:
            workflow = self.fixture(root)
            events = []
            workflow.event_callback = events.append
            first = workflow.run("view")
            self.assertEqual([r["task"] for r in first["stages"]], ["sample", "view"])
            path = Path(root) / "data with spaces/result.json"
            original_mtime = path.stat().st_mtime_ns
            second = workflow.run("all")
            self.assertTrue(all(r["reused"] for r in second["stages"]))
            self.assertEqual(path.stat().st_mtime_ns, original_mtime)
            self.assertTrue(any(e["type"] == "log" and e["line"] == "complete" for e in events))
            self.assertTrue(all(t["verified"] for t in workflow.status()))

    def test_changed_numerical_sources_require_explicit_fresh_and_keep_backup(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            workflow.run("sample")
            (Path(root) / "scientific_input.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "fresh rebuild"):
                workflow.run("sample")
            result = workflow.run("sample", fresh=True)
            self.assertFalse(result["stages"][0]["reused"])
            backups = list((Path(root) / "build/backups").rglob("result.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), {"created": True})

    def test_failed_fresh_generation_restores_previous_data_and_blocks_dependent(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            workflow.run("sample")
            original = (Path(root) / "data with spaces/result.json").read_bytes()
            failing = self.fixture(root, fail=True)
            with self.assertRaisesRegex(WorkflowError, "exit 7"):
                failing.run("view", fresh=True)
            self.assertEqual((Path(root) / "data with spaces/result.json").read_bytes(), original)
            self.assertFalse((Path(root) / "outputs/view.json").exists())
            self.assertTrue(list((Path(root) / "build/failed").rglob("result.json")))

    def test_existing_saved_data_adoption_checks_archives_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            directory = Path(root) / "data with spaces"
            directory.mkdir()
            (directory / "result.json").write_text('{"created":true}')
            with zipfile.ZipFile(directory / "raw.npz", "w") as archive:
                archive.writestr("sample.npy", b"kept raw measurements")
            result = workflow.run("sample")
            self.assertTrue(result["stages"][0]["reused"])
            manifest = json.loads(workflow.manifest_path.read_text())
            self.assertIn("original source revision unavailable", manifest["tasks"]["sample"]["provenance"])
            (directory / "raw.npz").write_bytes(b"truncated")
            with self.assertRaisesRegex(WorkflowError, "failed validation"):
                workflow.run("sample")

    def test_saved_data_edit_is_not_silently_adopted_again(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            workflow.run("sample")
            (Path(root) / "data with spaces/result.json").write_text('{"created":false}')
            self.assertEqual(workflow.status()[0]["status"], "stale")
            with self.assertRaisesRegex(WorkflowError, "changed after"):
                workflow.run("sample")

    def test_existing_presentation_without_source_record_is_regenerated(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            output = Path(root) / "outputs/view.json"
            output.parent.mkdir()
            output.write_text('{"stale":true}')
            result = workflow.run("view")
            self.assertFalse(result["stages"][-1]["reused"])
            self.assertEqual(json.loads(output.read_text()), {"created": True})

    def test_unknown_tasks_and_paths_cannot_become_shell_commands(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            with self.assertRaisesRegex(WorkflowError, "Unknown"):
                workflow.run("sample; arbitrary command")
            with self.assertRaisesRegex(WorkflowError, "fresh must"):
                workflow.run("sample", fresh="yes")
            bad = Task("bad", "Bad path", "data", (), (), ("../outside.txt",))
            unsafe = Workflow(root, tasks=(bad,))
            with self.assertRaisesRegex(WorkflowError, "escapes"):
                unsafe.task_catalog()


if __name__ == "__main__":
    unittest.main()
