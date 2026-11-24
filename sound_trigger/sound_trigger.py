#!/usr/bin/env python3
"""
Нажимает клавишу (по умолчанию G), когда микрофон слышит резкий звук:
хлопок, щелчок, всплеск воды и т.п.

Команды:
  python sound_trigger.py devices                 # список микрофонов
  python sound_trigger.py levels                  # живой индикатор громкости (для настройки порога)
  python sound_trigger.py record splash -n 5      # записать 5 примеров звука "splash"
  python sound_trigger.py run                     # жать G на ЛЮБОЙ резкий звук
  python sound_trigger.py run --match splash      # жать G только на звук, похожий на "splash"
"""
import argparse
import json
import math
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

SR = 44100            # частота дискретизации
BLOCK = 1024          # размер блока (~23 мс)
N_BANDS = 24          # число частотных полос в "отпечатке" звука
SNIPPET_SEC = 0.35    # сколько звука после начала анализируем при сравнении
PRE_SEC = 0.05        # сколько звука ДО начала захватываем
SOUNDS_DIR = Path(__file__).with_name("sounds")


# ---------------------------------------------------------------- анализ звука

def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-10)


def _band_bins(n_fft=BLOCK):
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    edges = np.geomspace(80, 16000, N_BANDS + 1)
    bands = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in zip(edges[:-1], edges[1:])]
    return [b for b in bands if len(b)]  # узкие нижние полосы без единого бина FFT выкидываем


_BANDS = _band_bins()
_WINDOW = np.hanning(BLOCK)


def _centered(v):
    return v - v.mean()  # убираем общую громкость, оставляем форму


def fingerprint(snippet):
    """Отпечаток звука: форма спектра + огибающая громкости во времени."""
    frames = [snippet[i:i + BLOCK] for i in range(0, len(snippet) - BLOCK + 1, BLOCK // 4)]
    levels = [rms_db(f) for f in frames]
    loud = max(levels) - 12  # берём только громкие кадры, без фонового шума
    frames = [f for f, lv in zip(frames, levels) if lv >= loud]
    spec = np.zeros(len(_BANDS))
    for f in frames:
        p = np.abs(np.fft.rfft(f * _WINDOW)) ** 2
        spec += np.array([p[b].mean() for b in _BANDS])
    spec = 10 * np.log10(spec / max(len(frames), 1) + 1e-12)
    env = np.array([rms_db(c) for c in np.array_split(snippet, 8)])
    return {"spec": _centered(spec).tolist(), "env": _centered(env).tolist()}


def similarity(a, b):
    """0..1 — насколько два отпечатка похожи (1 = одинаковые)."""
    d_spec = np.sqrt(np.mean((np.array(a["spec"]) - b["spec"]) ** 2))  # разница в дБ
    d_env = np.sqrt(np.mean((np.array(a["env"]) - b["env"]) ** 2))
    return float(np.exp(-(0.75 * d_spec + 0.25 * d_env) / 8))


# ---------------------------------------------------------------- микрофон

def mic_blocks(device=None):
    import sounddevice as sd  # импорт здесь, чтобы анализ работал и без PortAudio
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, blocksize=BLOCK, channels=1,
                        dtype="float32", device=device, callback=cb):
        while True:
            yield q.get()


def detect_events(blocks, threshold_db, margin_db, cooldown, snippet_sec, on_level=None):
    """Ищет резкие скачки громкости. Отдаёт (громкость_дБ, кусок_звука)."""
    pre_n = math.ceil(PRE_SEC * SR / BLOCK)
    post_n = math.ceil(snippet_sec * SR / BLOCK)
    history = deque(maxlen=pre_n)
    noise = None          # уровень фонового шума, подстраивается сам
    last = -1e9
    pending, onset_level = None, 0.0
    for i, b in enumerate(blocks):
        level = rms_db(b)
        now = i * BLOCK / SR
        if noise is None:
            noise = level
        if pending is not None:
            pending.append(b)
            if len(pending) >= pre_n + 1 + post_n:
                yield onset_level, np.concatenate(pending)
                pending = None
        elif level > threshold_db and level > noise + margin_db and now - last > cooldown:
            last = now
            onset_level = level
            pending = list(history) + [b]
            if post_n == 0:
                yield onset_level, np.concatenate(pending)
                pending = None
        if pending is None and level < noise + margin_db:
            noise = 0.95 * noise + 0.05 * level
        history.append(b)
        if on_level:
            on_level(level, noise)


# ---------------------------------------------------------------- клавиатура

def make_presser(key, directinput):
    if directinput:
        # Для игр, которые не видят обычные нажатия (Windows): pip install pydirectinput
        import pydirectinput
        return lambda: pydirectinput.press(key)
    from pynput.keyboard import Controller
    kb = Controller()

    def press():
        kb.press(key)
        time.sleep(0.05)
        kb.release(key)
    return press


# ---------------------------------------------------------------- команды

def load_templates(names):
    out = []
    for name in names:
        path = SOUNDS_DIR / f"{name}.json"
        if not path.exists():
            sys.exit(f"Нет файла {path}. Сначала: python sound_trigger.py record {name}")
        out += [(name, t) for t in json.loads(path.read_text(encoding="utf-8"))]
    return out


def cmd_devices(a):
    import sounddevice as sd
    print(sd.query_devices())


def cmd_levels(a):
    print("Пошумите/похлопайте. Ctrl+C — выход.\n")
    shown = [0]

    def show(level, noise):
        shown[0] += 1
        if shown[0] % 4:
            return
        bar = "#" * int(max(0, level + 80) / 2)
        print(f"\rгромкость {level:6.1f} дБ | фон {noise:6.1f} дБ | {bar:<40}", end="", flush=True)

    for level, _ in detect_events(mic_blocks(a.device), 999, 0, 0, 0, show):
        pass


def cmd_record(a):
    SOUNDS_DIR.mkdir(exist_ok=True)
    print(f"Издайте звук «{a.name}» {a.count} раз(а), с паузами ~1 сек.")
    prints = []
    events = detect_events(mic_blocks(a.device), a.threshold, a.margin, 1.0, SNIPPET_SEC)
    for level, snippet in events:
        prints.append(fingerprint(snippet))
        print(f"  пример {len(prints)}/{a.count} записан ({level:.1f} дБ)")
        if len(prints) >= a.count:
            break
    path = SOUNDS_DIR / f"{a.name}.json"
    path.write_text(json.dumps(prints), encoding="utf-8")
    print(f"Сохранено: {path}")


def cmd_run(a):
    templates = load_templates(a.match) if a.match else []
    press = make_presser(a.key, a.directinput)
    snippet = SNIPPET_SEC if templates else 0  # без сравнения жмём мгновенно
    mode = f"только на: {', '.join(a.match)}" if templates else "на любой резкий звук"
    print(f"Слушаю ({mode}). Клавиша: {a.key}. Ctrl+C — выход.")
    events = detect_events(mic_blocks(a.device), a.threshold, a.margin, a.cooldown, snippet)
    for level, snip in events:
        if templates:
            fp = fingerprint(snip)
            name, score = max(((n, similarity(fp, t)) for n, t in templates), key=lambda x: x[1])
            if score < a.similarity:
                print(f"  звук {level:.1f} дБ — не похож (лучшее: {name} {score:.2f})")
                continue
            print(f"  {name} ({score:.2f}) -> нажимаю {a.key}")
        else:
            print(f"  звук {level:.1f} дБ -> нажимаю {a.key}")
        press()


def main():
    p = argparse.ArgumentParser(description="Нажатие клавиши по звуку с микрофона")
    p.add_argument("--device", default=None, help="номер или имя микрофона (см. devices)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def detection_args(sp):
        sp.add_argument("--threshold", type=float, default=-35,
                        help="минимальная громкость звука, дБ (по умолчанию -35)")
        sp.add_argument("--margin", type=float, default=15,
                        help="на сколько дБ звук должен быть громче фона (по умолчанию 15)")

    sub.add_parser("devices").set_defaults(fn=cmd_devices)
    sub.add_parser("levels").set_defaults(fn=cmd_levels)

    r = sub.add_parser("record")
    r.add_argument("name")
    r.add_argument("-n", "--count", type=int, default=5)
    detection_args(r)
    r.set_defaults(fn=cmd_record)

    run = sub.add_parser("run")
    run.add_argument("--key", default="g")
    run.add_argument("--match", nargs="+", help="имена записанных звуков")
    run.add_argument("--similarity", type=float, default=0.7,
                     help="порог похожести 0..1 (по умолчанию 0.7)")
    run.add_argument("--cooldown", type=float, default=0.5,
                     help="пауза между нажатиями, сек")
    run.add_argument("--directinput", action="store_true",
                     help="режим для игр (Windows, нужен pydirectinput)")
    detection_args(run)
    run.set_defaults(fn=cmd_run)

    a = p.parse_args()
    if isinstance(a.device, str) and a.device.isdigit():
        a.device = int(a.device)
    try:
        a.fn(a)
    except KeyboardInterrupt:
        print("\nВыход.")


if __name__ == "__main__":
    main()
