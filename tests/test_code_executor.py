import os
import signal
import tempfile
import unittest

from code_executor import execute_python_code


class ExecutePythonCodeTests(unittest.TestCase):
    def test_captures_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as workdir:
            result = execute_python_code(
                "import sys\nprint('stdout-ok')\nprint('stderr-ok', file=sys.stderr)",
                cwd=workdir,
            )

        self.assertIn("stdout-ok", result)
        self.assertIn("stderr-ok", result)
        self.assertNotIn("[CODE EXECUTION ERROR]", result)

    def test_reports_nonzero_exit_status_and_traceback(self):
        with tempfile.TemporaryDirectory() as workdir:
            result = execute_python_code(
                "raise RuntimeError('candidate failed')", cwd=workdir
            )

        self.assertIn("[CODE EXECUTION ERROR]", result)
        self.assertIn("status 1", result)
        self.assertIn("RuntimeError: candidate failed", result)

    @unittest.skipUnless(os.name == "posix", "signals are POSIX-specific")
    def test_reports_signal_exit_status(self):
        with tempfile.TemporaryDirectory() as workdir:
            result = execute_python_code(
                "import os, signal\nos.kill(os.getpid(), signal.SIGABRT)",
                cwd=workdir,
            )

        self.assertIn("[CODE EXECUTION ERROR]", result)
        self.assertIn(f"status {-signal.SIGABRT}", result)

    def test_times_out_and_removes_temporary_script(self):
        with tempfile.TemporaryDirectory() as workdir:
            result = execute_python_code(
                "import time\ntime.sleep(10)", timeout=0.05, cwd=workdir
            )
            remaining = os.listdir(workdir)

        self.assertIn("exceeded the timeout", result)
        self.assertEqual([], remaining)


if __name__ == "__main__":
    unittest.main()
