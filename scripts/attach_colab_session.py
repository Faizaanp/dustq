"""Dust-Q Colab Runtime Auto-Discovery & Session Attacher.

Auto-discovers active Google Colab L4 runtime assignments on the user's account,
extracts proxy credentials, binds active Jupyter kernels/sessions, and registers
the session into google-colab-cli's state store.

Usage:
    .venv/Scripts/python scripts/attach_colab_session.py [--session-name tune-session]
"""

import argparse
import sys
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from colab_cli.common import state
    from colab_cli.state import SessionState
except ImportError:
    print("google-colab-cli not installed in python environment.", file=sys.stderr)
    sys.exit(1)


def attach_session(session_name: str = "tune-session"):
    print(f"[attach] Querying active Colab assignments for session '{session_name}'...")
    assignments = state.client.list_assignments()
    if not assignments:
        print("[attach] ERROR: No active Colab assignments found! Open Colab and connect a runtime first.")
        return False

    for i, a in enumerate(assignments):
        print(f"  [{i}] endpoint={a.endpoint}, accelerator={a.accelerator.value}, variant={a.variant.name}, shape={a.machine_shape.name}")

    a = assignments[0]
    existing_s = state.store.get(session_name)
    k_id = existing_s.kernel_id if existing_s else None
    sess_id = existing_s.session_id if existing_s else None

    s = SessionState(
        name=session_name,
        token=a.runtime_proxy_info.token,
        url=a.runtime_proxy_info.url,
        endpoint=a.endpoint,
        token_expires_at=a.runtime_proxy_info.expires_at(),
        variant=a.variant.name,
        accelerator=a.accelerator.value,
        machine_shape=a.machine_shape.name,
        kernel_id=k_id,
        session_id=sess_id,
    )
    state.store.add(s)

    # Bind active kernel and session IDs via Jupyter REST API
    headers = {"X-Colab-Runtime-Proxy-Token": s.token}
    try:
        r_s = requests.get(s.url + "/api/sessions", headers=headers, timeout=10)
        r_k = requests.get(s.url + "/api/kernels", headers=headers, timeout=10)
        if r_s.status_code == 200 and r_s.json():
            first_sess = r_s.json()[0]
            s.session_id = first_sess.get("id")
            s.kernel_id = first_sess.get("kernel", {}).get("id")
            notebook_name = first_sess.get("name", "notebook")
            print(f"[attach] Linked session '{s.session_id}' and kernel '{s.kernel_id}' ({notebook_name}).")
        elif r_k.status_code == 200 and r_k.json():
            s.kernel_id = r_k.json()[0].get("id")
            print(f"[attach] Linked kernel '{s.kernel_id}'.")
    except Exception as e:
        print(f"[attach] Warning: Failed to query Jupyter REST endpoints: {e}")

    state.store.add(s)
    print(f"[attach] Successfully registered '{session_name}' in colab-cli store.")
    print(f"         Endpoint: {s.endpoint} | Accelerator: {s.accelerator} | URL: {s.url[:45]}...")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach Colab CLI to active runtime assignment")
    parser.add_argument("--session-name", default="tune-session", help="Session identifier in state store")
    args = parser.parse_args()
    success = attach_session(args.session_name)
    sys.exit(0 if success else 1)
