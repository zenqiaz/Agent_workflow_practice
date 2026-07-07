"""One-shot runner for the phenolphthalein UV-Vis experiment.

Injects the task message as the first auto-input so the REPL processes it
without needing stdin redirection.  Unicode-safe (replaces surrogates before any API call).
"""
import sys, os

# Ensure stdout/stdin are UTF-8 safe on Windows
sys.stdin  = open(sys.stdin.fileno(),  mode="r", encoding="utf-8", errors="replace", closefd=False)
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)
sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)

MESSAGE = """\
Calculate UV-Vis absorption spectra for phenolphthalein in its two acid/base forms \
to predict the color on each side of the color-change point.

IMPORTANT — two separate molecular structures are required:

Form A (acid, colorless, H2In):
  - Load phenolphthalein from PubChem (name="phenolphthalein"), charge=0, mult=1
  - Optimize geometry: run_opt_job, B3LYP/def2-SVP, xtb_preopt=true, wall_timeout_seconds=7200
  - Add 3 explicit water molecules: run_solvator_cluster, nsolv=3
  - UV-Vis spectrum: run_tddft_job, nroots=8, on the solvated cluster

Form B (base, pink, In2-):
  - Load the open quinoid dianion from SMILES using name_to_geometry_xyz with \
name="O=C([O-])c1ccccc1/C(=C2\\C=CC(=O)C=C2)c1ccc([O-])cc1", charge=-2, mult=1
  - Optimize geometry: run_opt_job, B3LYP/def2-SVP, charge=-2, multiplicity=1, \
xtb_preopt=true, wall_timeout_seconds=10800
  - Add 3 explicit water molecules: run_solvator_cluster, nsolv=3
  - UV-Vis spectrum: run_tddft_job, nroots=8, on the solvated cluster

Do NOT use structure_add_remove_proton to generate Form B — the dianion has a \
completely different ring topology (open quinoid) that cannot be obtained by proton \
removal from the lactone form.  Load it directly from the SMILES above.

Report: wavelengths (nm) and oscillator strengths for each form. \
Form A should show no absorption in 400-700 nm (colorless); \
Form B should show a dominant peak ~530-560 nm (pink/magenta).\
"""

# Monkey-patch input() to pull from a predefined sequence.
# MESSAGE is first so the REPL treats it as the user's planning request.
import builtins
_RESPONSES = iter([
    MESSAGE,
    # compound confirmations (phenolphthalein acid form + SMILES dianion)
    "y",
    "y",
    # after plan is displayed → execute
    "run",
    # confirm "Execute this plan? (y/N):"
    "y",
    # exit
    "quit",
])

def _auto_input(prompt=""):
    try:
        ans = next(_RESPONSES)
    except StopIteration:
        ans = "quit"
    # Only echo short responses, not the full MESSAGE block
    display = ans if len(ans) < 80 else f"<message ({len(ans)} chars)>"
    sys.stdout.write(f"{prompt}{display}\n")
    sys.stdout.flush()
    return ans

builtins.input = _auto_input

# Patch json.dumps globally to strip surrogates before any API call
import json as _json
_orig_dumps = _json.dumps
def _safe_dumps(obj, **kw):
    s = _orig_dumps(obj, **kw)
    return s.encode("utf-8", errors="replace").decode("utf-8")
_json.dumps = _safe_dumps

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import glob
import json
import time

import nbo_agent_planning as _repl

# ── record state before run ──────────────────────────────────────────────────
_t0        = time.monotonic()
_t0_utc    = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
_existing  = set(glob.glob("runtime_reports/*.json"))

asyncio.run(_repl.main())

# ── collect new runtime reports written during this run ──────────────────────
_wall_s      = round(time.monotonic() - _t0, 1)
_new_reports = sorted(r for r in glob.glob("runtime_reports/*.json") if r not in _existing)

_runs = []
for _rpath in _new_reports:
    try:
        with open(_rpath, encoding="utf-8") as _f:
            _rdata = _json.load(_f)
        _runs.append({
            "runtime_report_path": _rpath,
            "plan_name":           (_rdata.get("plan") or {}).get("name"),
            "final_status":        _rdata.get("final_status"),
            "run_started_utc":     _rdata.get("run_started_utc"),
            "run_finished_utc":    _rdata.get("run_finished_utc"),
            "final_report":        _rdata.get("final_report"),
            "artifact_keys":       (_rdata.get("summary") or {}).get("artifact_keys", []),
            # token_usage is in REPL state["token_usage"] but not persisted to runtime report;
            # add nbo_agent_planning.py hook to expose it here if needed.
        })
    except Exception as _e:
        _runs.append({"runtime_report_path": _rpath, "error": str(_e)})

_log = {
    "experiment":      "phenolphthalein_uvvis",
    "timestamp_utc":   _t0_utc,
    "wall_elapsed_s":  _wall_s,
    "model":           os.getenv("LLM_MODEL", "gpt-4.1-mini"),
    "env_file":        os.getenv("ENV_FILE", ".env"),
    "n_runs":          len(_runs),
    "runs":            _runs,
}

os.makedirs("test_logs", exist_ok=True)
_log_path = f"test_logs/phph_uvvis_{_t0_utc}.json"
with open(_log_path, "w", encoding="utf-8") as _f:
    _json.dump(_log, _f, indent=2, ensure_ascii=True)
print(f"\nExperiment log saved: {_log_path}  (wall time: {_wall_s}s)")
