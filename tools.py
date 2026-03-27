import requests
import subprocess
import json
from typing import Dict, Any, List
import os

class ToolExecutor:
    """Executes various tools for the agent"""
    
    @staticmethod
    def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search the web for information"""
        try:
            # Using DuckDuckGo API (free, no key required)
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = {
                'query': query,
                'abstract': data.get('AbstractText', 'No summary available'),
                'source': data.get('AbstractSource', 'Unknown'),
                'results': []
            }
            
            # Parse related topics as search results
            for topic in data.get('RelatedTopics', [])[:num_results]:
                if 'Text' in topic:
                    results['results'].append(topic['Text'])
            
            return {'success': True, 'data': results}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def execute_python(code: str) -> Dict[str, Any]:
        """Safely execute Python code and capture printed output"""
        import io
        import contextlib
        try:
            exec_globals: Dict[str, Any] = {}
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code, exec_globals)  # noqa: S102
            output = stdout_capture.getvalue()
            err    = stderr_capture.getvalue()
            return {
                'success': True,
                'output': output or '(no output)',
                'stderr': err if err else None,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def execute_shell(command: str) -> Dict[str, Any]:
        """Execute shell commands safely"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Command timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def file_read(filepath: str) -> Dict[str, Any]:
        """Read file contents"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            return {'success': True, 'content': content, 'filepath': filepath}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def file_write(filepath: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
            return {'success': True, 'message': f'File written to {filepath}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def api_call(url: str, method: str = 'GET', headers: Dict = None, data: Dict = None) -> Dict[str, Any]:
        """Make API calls"""
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=10)
            else:
                return {'success': False, 'error': 'Unsupported method'}
            
            return {
                'success': True,
                'status_code': response.status_code,
                'response': response.json() if response.text else {}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}