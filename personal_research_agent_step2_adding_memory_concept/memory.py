import json
import re
from datetime import datetime, timezone
from pathlib import Path

class ConversationMemory:
    def __init__(self, conversation_id="default"):
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", conversation_id) or "default"
        self.directory = Path(__file__).resolve().parent / "conversations"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{safe_id}.json"
        self.data = {"conversation_id": safe_id, "history": [], "state": self.initial_state(), "updated_at": None}
        self.load()

    @staticmethod
    def initial_state():
        return {"current_task": None, "search_count": 0, "sources_found": 0, "status": "idle"}

    def load(self):
        if not self.path.exists():
            self.save()
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                history = saved.get("history", [])
                self.data["history"] = history if isinstance(history, list) else []
                state = saved.get("state", {})
                if isinstance(state, dict):
                    self.data["state"].update(state)
                self.data["updated_at"] = saved.get("updated_at")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load saved memory: {exc}")

    def save(self):
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp = self.path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        temp.replace(self.path)

    def add_message(self, role, content):
        if role not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        self.data["history"].append({"role": role, "content": str(content)})
        self.save()

    def get_history(self, limit=None):
        history = self.data["history"]
        return history[-limit:] if limit else history

    def update_state(self, **updates):
        self.data["state"].update(updates)
        self.save()

    def get_state(self):
        return dict(self.data["state"])

    def clear(self):
        self.data["history"] = []
        self.data["state"] = self.initial_state()
        self.save()
