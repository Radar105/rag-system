"""
LLM Runner module for RAG System.
API-agnostic interface supporting multiple LLM providers.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import Config


class LLMRunner:
    """
    Manages LLM API calls with RAG context.
    Supports: Anthropic (Claude Code CLI), OpenAI, or any subprocess-based CLI.
    """

    # Safe read-only tools for Claude Code
    BASE_TOOLS = ["Read"]

    def __init__(self, allow_tools: bool, provider: Optional[str] = None):
        """
        Initialize LLM runner.

        Args:
            allow_tools: Enable Read tool for file access (Claude only)
            provider: LLM provider - 'claude', 'openai', or custom command
                     Defaults to ANTHROPIC_API_KEY presence
        """
        self.tools = [] if not allow_tools else self.BASE_TOOLS
        self.provider = provider or self._detect_provider()

    def _detect_provider(self) -> str:
        """Auto-detect available LLM provider from environment."""
        if os.getenv("ANTHROPIC_API_KEY"):
            return "claude"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai"
        else:
            return "claude"  # Default fallback

    def call(self, prompt: str, depth: int) -> str:
        """
        Execute LLM query with RAG context.

        Args:
            prompt: Complete prompt with RAG context
            depth: Current recursion depth

        Returns:
            LLM response
        """
        if depth >= Config.RECURSION_LIMIT:
            return "⚠️  Recursion limit hit – refusing nested RAG."

        if self.provider == "claude":
            return self._call_claude(prompt, depth)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            return self._call_custom(prompt)

    def _call_claude(self, prompt: str, depth: int) -> str:
        """Call Claude via Claude Code CLI."""
        cmd = ["claude", "-p", "--strict-mcp-config"]

        # System prompt for RAG constraints
        sys_lines = [
            "You are a RAG Assistant.",
            "Answer using ONLY the RAG context in the user's message.",
            (
                "If tools are enabled, you MAY use the Read tool to open files ONLY "
                "within the whitelisted directories provided by the host. Do NOT access other files."
            ),
            "Prefer the provided context when sufficient. Keep answers concise and cite sources when present.",
            "If uncertain, ask a brief clarifying question before proceeding.",
        ]
        cmd += ["--append-system-prompt", "\n".join(sys_lines)]

        # Whitelist directories for Read tool
        if self.tools:
            allowed_dirs = self._get_allowed_directories()
            for d in allowed_dirs:
                cmd += ["--add-dir", d]

        # Tool restrictions
        cmd += ["--allowedTools", ",".join(self.tools)]

        # Debug mode
        if Config.enable_debug():
            cmd += [
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--debug",
                "--verbose",
            ]

        # Environment with recursion tracking
        env = os.environ.copy()
        env["RAG_DEPTH"] = str(depth + 1)

        try:
            res = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=Config.LLM_TIMEOUT,
                env=env,
            )
            return res.stdout if res.returncode == 0 else res.stderr
        except subprocess.TimeoutExpired:
            return "⏱️  LLM call timed out."

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API directly."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a RAG Assistant. Answer using ONLY the RAG context provided. Keep answers concise and cite sources."
                    },
                    {"role": "user", "content": prompt}
                ],
                timeout=Config.LLM_TIMEOUT
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"❌ OpenAI API error: {str(e)}"

    def _call_custom(self, prompt: str) -> str:
        """Call custom LLM command from environment."""
        custom_cmd = os.getenv("RAG_LLM_COMMAND")
        if not custom_cmd:
            return "❌ No LLM provider configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or RAG_LLM_COMMAND."

        try:
            res = subprocess.run(
                custom_cmd.split(),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=Config.LLM_TIMEOUT,
            )
            return res.stdout if res.returncode == 0 else res.stderr
        except Exception as e:
            return f"❌ Custom LLM error: {str(e)}"

    def _get_allowed_directories(self) -> List[str]:
        """
        Get safe whitelisted directories for Read tool (Claude only).

        Returns:
            List of allowed directory paths
        """
        allowed_dirs = []
        for p in Config.DEFAULT_PATHS:
            try:
                pp = Path(p)
                if not pp.exists():
                    continue

                dpath = pp if pp.is_dir() else pp.parent

                # Avoid over-broad grants like HOME
                if dpath == Config.HOME:
                    continue

                allowed_dirs.append(str(dpath))
            except Exception:
                continue

        # Deduplicate while preserving order
        seen = set()
        safe_dirs = []
        for d in allowed_dirs:
            if d not in seen:
                seen.add(d)
                safe_dirs.append(d)

        return safe_dirs
