"""
Обращение к языковой модели через Ollama.

Всё локально: ни один текст не покидает машину, платить не нужно. Модель по умолчанию —
qwen3:8b (Apache-2.0). Ответ приходит по JSON-схеме (structured outputs), поэтому разбор
не зависит от того, как модель решит оформить список.
"""
import json
import os
import time
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")

SCHEMA = {"type": "object", "properties": {"texts": {"type": "array", "items": {"type": "string"}}},
          "required": ["texts"]}

def ask(model, prompt, timeout=900):
    body = {"model": model, "prompt": prompt, "stream": False, "format": SCHEMA, "think": False,
            "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 900}}
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return json.loads(d["response"]).get("texts", []), {
        "tokens": d.get("eval_count"), "seconds": round(time.time() - t0, 1)}
