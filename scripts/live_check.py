"""Реальная проверка интеграции с LLM-провайдером (по умолчанию GigaChat).

    python scripts/live_check.py --models            # список моделей, доступных ключу
    python scripts/live_check.py --probe-structured  # принуждает ли API JSON (response_format)
    python scripts/live_check.py [файл.json ...]     # прогон примеров, метрики, результаты

Без списка файлов прогоняются ВСЕ examples/inputs (сейчас 8 примеров: до 16 обращений к модели с
учётом корректирующих вызовов, плюс повторы SDK) — для экономии токенов указывайте файлы.
Нужен GIGACHAT_CREDENTIALS в .env. Скрипт не печатает и не сохраняет секреты: перед записью
каждый файл проверяется на наличие ключа.
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # .env и относительные пути из него разрешаются от корня проекта
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import LLMError  # noqa: E402
from app.providers import create_client  # noqa: E402
from app.schemas import ProductInput  # noqa: E402
from app.service import generate_card  # noqa: E402
from app.validator import DESCRIPTION_MAX, DESCRIPTION_MIN, QUERIES_MAX, QUERIES_MIN, TITLE_MAX  # noqa: E402

OUT_DIR = ROOT / "examples" / "outputs" / "gigachat-live"


def safe_write(path: Path, text: str, secret: str) -> None:
    if secret and secret in text:
        sys.exit(f"ОТКАЗ: в {path.name} обнаружен ключ авторизации — файл не записан.")
    path.write_text(text, encoding="utf-8")


def list_models(client) -> None:
    models = (
        client._giga.get_models()
    )  # специфично для GigaChat: это скрипт проверки, не бизнес-логика
    print("Модели, доступные ключу:")
    for m in models.data:
        print("  ", getattr(m, "id_", None) or getattr(m, "id", None))


def probe_structured(client) -> None:
    """Воспроизводит эксперимент, на котором основано решение не использовать response_format."""
    from gigachat.models import Chat, Messages, MessagesRole

    schema = {
        "type": "object",
        "properties": {"date": {"type": "string"}, "event": {"type": "string"}},
        "required": ["date", "event"],
    }
    text = "27 октября 2023 года у меня родился сын"
    instruction = 'Верни ТОЛЬКО JSON-объект вида {"date": "dd.mm", "event": "..."} без пояснений.'
    for label, with_format, with_instruction in [
        ("только response_format", True, False),
        ("только инструкция в промпте", False, True),
    ]:
        messages = [Messages(role=MessagesRole.USER, content=text)]
        if with_instruction:
            messages.insert(0, Messages(role=MessagesRole.SYSTEM, content=instruction))
        extra = {"response_format": {"type": "json_schema", "schema": schema, "strict": True}}
        chat = Chat(
            model=client.model,
            temperature=0.1,
            max_tokens=80,
            messages=messages,
            **(extra if with_format else {}),
        )
        content = client._giga.chat(chat).choices[0].message.content
        try:
            json.loads(content)
            verdict = "JSON"
        except ValueError:
            verdict = "НЕ JSON"
        print(f"{label:30s} -> {verdict:8s} {content[:60]!r}")


def run_examples(client, files: list[Path], secret: str, out_dir: Path, repeat: int = 1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    for f, attempt in ((f, i) for f in files for i in range(1, repeat + 1)):
        product = ProductInput.model_validate(json.loads(f.read_text(encoding="utf-8")))
        started = time.perf_counter()
        try:
            card = generate_card(product, client)
        except LLMError as e:
            print(f"{f.name}: ОШИБКА {e.code}: {e.message}")
            continue
        seconds = time.perf_counter() - started
        stem = f.stem if repeat == 1 else f"{f.stem}.run{attempt}"
        safe_write(out_dir / f"{stem}.result.json", card.model_dump_json(indent=2) + "\n", secret)
        safe_write(out_dir / f"{stem}.result.md", card.markdown, secret)

        title_ok = card.seo_title is not None and len(card.seo_title) <= TITLE_MAX
        desc_len = len(card.seo_description) if card.seo_description else 0
        desc_ok = DESCRIPTION_MIN <= desc_len <= DESCRIPTION_MAX
        queries_ok = QUERIES_MIN <= len(card.search_queries) <= QUERIES_MAX
        kinds = sorted({w.code.value for w in card.warnings})
        rows.append((f.name, card.status.value, title_ok, desc_len, queries_ok, card.generation))
        runs.append(
            {
                "input": f.name,
                "status": card.status.value,
                "model": card.generation.model,
                "llm_calls": card.generation.llm_calls,
                "repaired": card.generation.repaired,
                "prompt_tokens": card.generation.prompt_tokens,
                "completion_tokens": card.generation.completion_tokens,
                "seconds": round(seconds, 1),
                "title_length": len(card.seo_title or ""),
                "title_within_limit": title_ok,
                "description_length": desc_len,
                "description_within_range": desc_ok,
                "queries_count": len(card.search_queries),
                "queries_within_range": queries_ok,
                "warning_codes": kinds,
            }
        )
        print(
            f"{f.name:38s} {card.status.value:13s} заголовок {len(card.seo_title or ''):>2}"
            f"{'' if title_ok else '!'} | описание {desc_len:>4}{'' if desc_ok else '!'}"
            f" | запросов {len(card.search_queries)} | вызовов {card.generation.llm_calls}"
            f" | {seconds:.1f} с | замечания: {kinds or '—'}"
        )

    if repeat > 1:  # разброс между запусками: одна цифра на прогон ничего не доказывает
        print("\nИтог по повторам (статусы / длины описания, симв. / среднее время):")
        for f in files:
            mine = [r for r in runs if r["input"] == f.name]
            if not mine:
                continue
            statuses = {s: sum(r["status"] == s for r in mine) for s in {r["status"] for r in mine}}
            lengths = [r["description_length"] for r in mine]
            mean_s = sum(r["seconds"] for r in mine) / len(mine)
            print(f"  {f.name}: {statuses} | {lengths} | {mean_s:.1f} с")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": get_settings().llm_provider,
        "configured_model": client.model,
        "runs": runs,
    }
    safe_write(
        out_dir / "run-summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        secret,
    )
    print(f"\nРезультаты сохранены в {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path, help="JSON с данными товара")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--probe-structured", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="сколько раз прогнать каждый пример")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="куда сохранить результаты")
    args = parser.parse_args()

    settings = get_settings()
    try:
        client = create_client(settings)
        if not hasattr(client, "_giga") and (args.models or args.probe_structured):
            sys.exit("Эти режимы специфичны для GigaChat.")
    except LLMError as e:
        sys.exit(f"Ошибка конфигурации: {e.message}")
    secret = (
        settings.gigachat_credentials.get_secret_value() if settings.gigachat_credentials else ""
    )

    try:
        if args.models:
            list_models(client)
        elif args.probe_structured:
            probe_structured(client)
        else:
            files = args.files or sorted((ROOT / "examples" / "inputs").glob("*.json"))
            run_examples(client, files, secret, args.out, args.repeat)
    except LLMError as e:
        sys.exit(f"Ошибка провайдера ({e.code}): {e.message}")


if __name__ == "__main__":
    main()
