import os
import ast
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from app.config import settings

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", 
    "pty", "platform", "multiprocessing", "threading", "asyncio"
}
BLOCKED_FUNCTIONS = {
    "eval", "exec", "compile", "globals", "locals", "getattr", "setattr", 
    "delattr", "hasattr", "__import__"
}


def is_safe_code(code: str) -> tuple[bool, Optional[str]]:
    """Проверка кода через AST перед выполнением в облегченном режиме.
    
    Предотвращает простые вредоносные импорты и опасные системные вызовы. 
    Не гарантирует абсолютную защиту от обхода анализа (динамический импорт, 
    обфускация). Для полной изоляции требуется запуск в контейнере Docker.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)}"
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split('.')[0]
                if base_module in BLOCKED_MODULES:
                    return False, f"Import of module '{base_module}' is blocked for security reasons."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split('.')[0]
                if base_module in BLOCKED_MODULES:
                    return False, f"Import from module '{base_module}' is blocked for security reasons."
        
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_FUNCTIONS:
                    return False, f"Function call '{node.func.id}' is blocked for security reasons."
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in BLOCKED_MODULES:
                        return False, f"Access to '{node.func.value.id}.{node.func.attr}' is blocked."
    
    return True, None


def check_docker_available() -> bool:
    """Проверка доступности Docker в системе."""
    try:
        # shell=True необходим на Windows для корректного поиска docker CLI
        result = subprocess.run(
            ["docker", "ps"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            shell=True if os.name == 'nt' else False
        )
        return result.returncode == 0
    except Exception:
        return False


class SandboxExecutor:
    """Окружение для безопасного выполнения сгенерированного Python-кода."""

    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or settings.SANDBOX_MODE
        self.timeout = settings.SANDBOX_TIMEOUT
        self.docker_available = check_docker_available()
        
        if self.mode == "docker" and not self.docker_available:
            print("⚠️ Docker mode requested but Docker is not available. Falling back to lightweight mode.")
            self.mode = "lightweight"

    def execute(self, code: str, input_files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Выполнение скрипта во временной директории с копированием входных файлов."""
        if self.mode == "disabled":
            return {
                "success": False,
                "stdout": "",
                "stderr": "Code Interpreter sandbox is disabled by settings.",
                "files": []
            }

        if self.mode == "lightweight":
            is_safe, error_msg = is_safe_code(code)
            if not is_safe:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"❌ Security Block: {error_msg}",
                    "files": []
                }

        with tempfile.TemporaryDirectory(prefix="sandbox_") as temp_dir:
            copied_files_map = {}
            if input_files:
                for file_info in input_files:
                    src_path = file_info.get("path")
                    file_name = file_info.get("name")
                    if src_path and os.path.exists(src_path) and file_name:
                        dest_path = os.path.join(temp_dir, file_name)
                        shutil.copy2(src_path, dest_path)
                        copied_files_map[file_name] = dest_path

            script_path = os.path.join(temp_dir, "main.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            files_before = set(os.listdir(temp_dir))

            stdout, stderr, returncode = "", "", -1
            try:
                if self.mode == "docker":
                    abs_temp_dir = os.path.abspath(temp_dir)
                    cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{abs_temp_dir}:/workspace",
                        "-w", "/workspace",
                        "python:3.11-slim",
                        "python", "main.py"
                    ]
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout,
                        shell=True if os.name == 'nt' else False
                    )
                    stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
                else:
                    import sys
                    cmd = [sys.executable, "main.py"]
                    proc = subprocess.run(
                        cmd,
                        cwd=temp_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout
                    )
                    stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": stdout,
                    "stderr": f"❌ Timeout Error: Code execution exceeded the time limit of {self.timeout} seconds.",
                    "files": []
                }
            except Exception as e:
                return {
                    "success": False,
                    "stdout": stdout,
                    "stderr": f"❌ Execution Error: {str(e)}",
                    "files": []
                }

            files_after = set(os.listdir(temp_dir))
            new_files = files_after - files_before
            generated_files = []

            artifacts_dir = os.path.join(tempfile.gettempdir(), "multiagent_artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)

            for file_name in new_files:
                if file_name == "main.py":
                    continue
                file_path = os.path.join(temp_dir, file_name)
                if os.path.isfile(file_path):
                    import uuid
                    artifact_id = str(uuid.uuid4())
                    ext = os.path.splitext(file_name)[1]
                    persistent_path = os.path.join(artifacts_dir, f"{artifact_id}{ext}")
                    shutil.copy2(file_path, persistent_path)
                    
                    generated_files.append({
                        "name": file_name,
                        "temp_path": persistent_path,
                        "size": os.path.getsize(file_path),
                        "mime_type": "image/png" if ext == ".png" else "application/octet-stream"
                    })

            return {
                "success": returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "files": generated_files
            }
