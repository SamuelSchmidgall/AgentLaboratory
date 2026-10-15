import os
import signal
import subprocess
import sys
import tempfile


def _terminate_process(proc):
    """Terminate a timed-out process and, on POSIX, its child processes."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            proc.kill()
        proc.wait()


def execute_python_code(code, timeout=600, cwd=None):
    """Execute Python in a clean child interpreter and report its real status."""
    workdir = os.path.abspath(cwd or os.getcwd())
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            prefix=".agentlab_exec_",
            dir=workdir,
            delete=False,
        ) as script:
            script.write(code)
            script_path = script.name

        popen_kwargs = {
            "args": [sys.executable, script_path],
            "cwd": workdir,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(**popen_kwargs)
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            proc.communicate()
            return (
                f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout "
                f"limit of {timeout} seconds. You must reduce the time complexity "
                "of your code."
            )

        if proc.returncode != 0:
            return (
                "[CODE EXECUTION ERROR]: Code execution process exited with "
                f"status {proc.returncode}.\n{output}"
            )
        return output
    finally:
        if script_path is not None:
            try:
                os.remove(script_path)
            except FileNotFoundError:
                pass
