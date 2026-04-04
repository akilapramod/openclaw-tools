"""
Compatibility shim for ChromaDB on Python 3.14+.

ChromaDB's config.py uses pydantic.v1.BaseSettings which is broken
on Python 3.14+ (pydantic v1 cannot handle PEP 649 lazy annotations).
Additionally, chromadb has unannotated fields that pydantic v2 rejects,
and chromadb's Settings reads from .env files, which may contain keys
not declared in chromadb (e.g. GROQ_API_KEY), causing "extra_forbidden".

This module:
1. Injects a pydantic.v1 shim built on pydantic-settings (v2)
   with extra="ignore" so .env keys not in Settings are silently skipped.
2. Patches chromadb.config source to add missing type annotations.

Must be imported BEFORE any chromadb usage.
See: https://github.com/chroma-core/chroma/issues/5996
"""

import sys
import types


def _patch_chromadb():
    """Pre-inject a working pydantic.v1 shim + fix chromadb annotations."""
    if sys.version_info < (3, 14):
        return

    if "chromadb" in sys.modules:
        return

    # ── 1. Patch chromadb config source to add missing annotations ───────
    import importlib.util
    from pathlib import Path

    chromadb_pkg = importlib.util.find_spec("chromadb")
    if chromadb_pkg is None or chromadb_pkg.submodule_search_locations is None:
        return

    config_path = Path(list(chromadb_pkg.submodule_search_locations)[0]) / "config.py"
    if not config_path.exists():
        return

    source = config_path.read_text(encoding="utf-8")

    # Fix unannotated fields that pydantic v2 BaseSettings rejects
    replacements = {
        'chroma_coordinator_host = "localhost"': 'chroma_coordinator_host: str = "localhost"',
        'chroma_logservice_host = "localhost"': 'chroma_logservice_host: str = "localhost"',
        'chroma_logservice_port = 50052': 'chroma_logservice_port: int = 50052',
    }

    needs_patch = any(old in source for old in replacements)
    if needs_patch:
        for old, new in replacements.items():
            source = source.replace(old, new)
        config_path.write_text(source, encoding="utf-8")

    # ── 2. Build pydantic.v1 shim ────────────────────────────────────────
    from pydantic_settings import BaseSettings as _V2BaseSettings
    from pydantic import field_validator as _fv, ConfigDict

    # Subclass that allows extra env vars (GROQ_API_KEY, TAVILY_API_KEY, etc.)
    class PermissiveBaseSettings(_V2BaseSettings):
        model_config = ConfigDict(extra="ignore")

    def validator(*fields, pre=False, always=False, allow_reuse=False):
        """Translate pydantic v1 @validator() to v2 @field_validator()."""
        mode = "before" if pre else "after"
        def decorator(func):
            if not isinstance(func, classmethod):
                func = classmethod(func)
            return _fv(*fields, mode=mode)(func)
        return decorator

    fake_v1 = types.ModuleType("pydantic.v1")
    fake_v1.__package__ = "pydantic"
    fake_v1.BaseSettings = PermissiveBaseSettings
    fake_v1.validator = validator
    sys.modules["pydantic.v1"] = fake_v1


_patch_chromadb()
