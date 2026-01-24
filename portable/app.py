# coding=utf-8
"""
Qwen3-TTS Portable PRO - Русскоязычная версия со стримингом
Синтез речи с поддержкой: Дизайн голоса, Клонирование голоса, Пресеты голосов
Multi-speaker режим, профили голосов, загрузка из облака

Авторы:
@nerual_dreming - база, основной код, основатель ArtGeneration.me
"""

import os
import sys
import time
import json
import threading
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
import pickle
import hashlib

import gradio as gr
import numpy as np
import torch
import soundfile as sf
from huggingface_hub import snapshot_download, hf_hub_download

# Добавляем родительскую директорию для импорта qwen_tts
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# =====================================================
# Константы и конфигурация
# =====================================================

APP_VERSION = "2.0.0"
APP_NAME = "Qwen3-TTS Portable PRO"

# Директории
SCRIPT_DIR = Path(__file__).parent
VOICES_DIR = SCRIPT_DIR / "voices"
PROFILES_DIR = SCRIPT_DIR / "profiles"
OUTPUT_DIR = SCRIPT_DIR / "output"
CONFIG_FILE = SCRIPT_DIR / "config.json"

# Создаем директории
VOICES_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# Глобальные переменные
# =====================================================

# Загруженные модели (кэш)
loaded_models: Dict[tuple, Qwen3TTSModel] = {}

# Кэш профилей голосов
voice_profiles_cache: Dict[str, VoiceClonePromptItem] = {}

# Флаги для стриминга
is_generating = False
stop_generation = False

# Размеры моделей
MODEL_SIZES = ["0.6B", "1.7B"]

# Типы моделей
MODEL_TYPES = {
    "Base": "Клонирование голоса",
    "CustomVoice": "Пресеты голосов",
    "VoiceDesign": "Дизайн голоса"
}

# Спикеры для CustomVoice
SPEAKERS = {
    "Aiden": "Эйден (мужской, английский)",
    "Dylan": "Дилан (мужской, английский)",
    "Eric": "Эрик (мужской, английский)",
    "Ono_anna": "Анна (женский, японский)",
    "Ryan": "Райан (мужской, английский)",
    "Serena": "Серена (женский, английский)",
    "Sohee": "Сохи (женский, корейский)",
    "Uncle_fu": "Дядя Фу (мужской, китайский)",
    "Vivian": "Вивиан (женский, английский)"
}

# Языки
LANGUAGES = {
    "Auto": "Авто (определить автоматически)",
    "Russian": "Русский",
    "English": "Английский",
    "Chinese": "Китайский",
    "Japanese": "Японский",
    "Korean": "Корейский",
    "French": "Французский",
    "German": "Немецкий",
    "Spanish": "Испанский",
    "Portuguese": "Португальский",
    "Italian": "Итальянский"
}

# =====================================================
# Конфигурация приложения
# =====================================================

@dataclass
class AppConfig:
    """Конфигурация приложения."""
    default_model_size: str = "1.7B"
    default_language: str = "Russian"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    auto_save_audio: bool = True
    theme: str = "soft"

    @classmethod
    def load(cls) -> "AppConfig":
        """Загрузка конфигурации из файла."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
        return cls()

    def save(self):
        """Сохранение конфигурации в файл."""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")


# Глобальная конфигурация
app_config = AppConfig.load()

# =====================================================
# Профили голосов
# =====================================================

@dataclass
class VoiceProfile:
    """Профиль голоса для сохранения и загрузки."""
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False
    audio_hash: str = ""  # хэш референсного аудио

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(**data)


def get_audio_hash(audio_data: np.ndarray) -> str:
    """Получение хэша аудио данных."""
    return hashlib.md5(audio_data.tobytes()).hexdigest()[:16]


def save_voice_profile(
    name: str,
    description: str,
    ref_audio: Tuple[np.ndarray, int],
    ref_text: Optional[str],
    x_vector_only: bool,
    model_size: str
) -> str:
    """Сохранение профиля голоса."""
    try:
        wav, sr = ref_audio
        audio_hash = get_audio_hash(wav)

        # Создаем профиль
        profile = VoiceProfile(
            name=name,
            description=description,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            audio_hash=audio_hash
        )

        # Получаем модель для создания voice clone prompt
        tts = get_model("Base", model_size)

        # Создаем VoiceClonePromptItem
        voice_prompt = tts.create_voice_clone_prompt(
            ref_audio=(wav, sr),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only
        )

        # Сохраняем данные
        profile_dir = PROFILES_DIR / name
        profile_dir.mkdir(exist_ok=True)

        # Сохраняем метаданные
        with open(profile_dir / "profile.json", "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

        # Сохраняем аудио
        sf.write(profile_dir / "reference.wav", wav, sr)

        # Сохраняем voice prompt (тензоры)
        torch.save({
            "ref_code": voice_prompt.ref_code,
            "ref_spk_embedding": voice_prompt.ref_spk_embedding,
            "x_vector_only_mode": voice_prompt.x_vector_only_mode,
            "icl_mode": voice_prompt.icl_mode,
            "ref_text": voice_prompt.ref_text
        }, profile_dir / "voice_prompt.pt")

        # Обновляем кэш
        voice_profiles_cache[name] = voice_prompt

        return f"Профиль '{name}' успешно сохранён!"

    except Exception as e:
        return f"Ошибка сохранения профиля: {e}"


def load_voice_profile(name: str) -> Tuple[Optional[VoiceClonePromptItem], str]:
    """Загрузка профиля голоса."""
    try:
        # Проверяем кэш
        if name in voice_profiles_cache:
            return voice_profiles_cache[name], f"Профиль '{name}' загружен из кэша."

        profile_dir = PROFILES_DIR / name
        if not profile_dir.exists():
            return None, f"Профиль '{name}' не найден."

        # Загружаем voice prompt
        data = torch.load(profile_dir / "voice_prompt.pt", map_location="cpu")

        voice_prompt = VoiceClonePromptItem(
            ref_code=data["ref_code"],
            ref_spk_embedding=data["ref_spk_embedding"],
            x_vector_only_mode=data["x_vector_only_mode"],
            icl_mode=data["icl_mode"],
            ref_text=data.get("ref_text")
        )

        # Сохраняем в кэш
        voice_profiles_cache[name] = voice_prompt

        return voice_prompt, f"Профиль '{name}' успешно загружен!"

    except Exception as e:
        return None, f"Ошибка загрузки профиля: {e}"


def list_voice_profiles() -> List[str]:
    """Получение списка сохранённых профилей."""
    profiles = []
    for path in PROFILES_DIR.iterdir():
        if path.is_dir() and (path / "profile.json").exists():
            profiles.append(path.name)
    return sorted(profiles)


def delete_voice_profile(name: str) -> str:
    """Удаление профиля голоса."""
    try:
        import shutil
        profile_dir = PROFILES_DIR / name
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
            if name in voice_profiles_cache:
                del voice_profiles_cache[name]
            return f"Профиль '{name}' удалён."
        return f"Профиль '{name}' не найден."
    except Exception as e:
        return f"Ошибка удаления профиля: {e}"


# =====================================================
# Загрузка голосов из облака
# =====================================================

CLOUD_VOICES_REPO = "Slait/russia_voices"
CLOUD_VOICES_BASE_URL = "https://huggingface.co/datasets/Slait/russia_voices/resolve/main"

# Список всех доступных голосов (обновляется при загрузке)
CLOUD_VOICES_CACHE: List[str] = []

def get_cloud_voices_list() -> Tuple[List[str], str]:
    """Получение полного списка голосов из облака."""
    global CLOUD_VOICES_CACHE
    from huggingface_hub import list_repo_files

    try:
        files = list(list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset"))
        voices = [f[:-4] for f in files if f.endswith(".mp3")]
        voice_list = sorted(voices)
        CLOUD_VOICES_CACHE = voice_list
        return voice_list, f"Найдено {len(voice_list)} голосов. Репозиторий: {CLOUD_VOICES_REPO}"
    except Exception as e:
        return [], f"Ошибка загрузки списка: {e}"


def download_cloud_voice(voice_name: str) -> str:
    """Загрузка голоса из облака."""
    import requests

    try:
        # Скачиваем MP3 файл
        mp3_url = f"{CLOUD_VOICES_BASE_URL}/{voice_name}.mp3?download=true"
        txt_url = f"{CLOUD_VOICES_BASE_URL}/{voice_name}.txt?download=true"

        # Скачиваем аудио
        response = requests.get(mp3_url, timeout=60, stream=True)
        response.raise_for_status()

        mp3_path = VOICES_DIR / f"{voice_name}.mp3"
        with open(mp3_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Пробуем скачать текст
        try:
            txt_response = requests.get(txt_url, timeout=30)
            if txt_response.status_code == 200:
                txt_path = VOICES_DIR / f"{voice_name}.txt"
                txt_path.write_text(txt_response.text, encoding="utf-8")
        except:
            pass

        return f"Голос '{voice_name}' успешно загружен!"

    except Exception as e:
        return f"Ошибка загрузки голоса '{voice_name}': {e}"


# =====================================================
# Вспомогательные функции
# =====================================================

def get_device():
    """Определение устройства для вычислений."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_path(model_type: str, model_size: str) -> str:
    """Получение пути к модели."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


def get_model(model_type: str, model_size: str) -> Qwen3TTSModel:
    """Получение или загрузка модели."""
    global loaded_models
    key = (model_type, model_size)

    if key not in loaded_models:
        model_path = get_model_path(model_type, model_size)
        device = get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Пробуем использовать Flash Attention, если доступен
        attn_impl = None
        if device == "cuda":
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print(f"Flash Attention 2 активирован для {model_type} {model_size}")
            except ImportError:
                attn_impl = "sdpa"
                print(f"Используется SDPA для {model_type} {model_size}")

        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        print(f"Модель {model_type} {model_size} загружена успешно!")

    return loaded_models[key]


def normalize_audio(wav, eps=1e-12, clip=True):
    """Нормализация аудио в диапазон [-1, 1]."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Неподдерживаемый тип данных: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def audio_to_tuple(audio) -> Optional[Tuple[np.ndarray, int]]:
    """Конвертация аудио Gradio в кортеж (wav, sr)."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = normalize_audio(audio["data"])
        return wav, sr

    return None


def save_audio_file(audio_data: np.ndarray, sample_rate: int, output_dir: str = None) -> str:
    """Сохранение аудио в файл."""
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"qwen3_tts_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, audio_data, sample_rate)
    return filepath


# =====================================================
# Multi-speaker парсер
# =====================================================

def parse_multi_speaker_script(script: str) -> List[Tuple[int, str]]:
    """
    Парсинг скрипта с несколькими дикторами.
    Формат: "Speaker N: текст" или "Диктор N: текст"

    Возвращает список кортежей (speaker_id, text)
    """
    lines = script.strip().split('\n')
    result = []

    # Паттерны для парсинга
    patterns = [
        r'^Speaker\s*(\d+)\s*:\s*(.+)$',
        r'^Диктор\s*(\d+)\s*:\s*(.+)$',
        r'^Голос\s*(\d+)\s*:\s*(.+)$',
        r'^\[(\d+)\]\s*(.+)$',
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                text = match.group(2).strip()
                result.append((speaker_id, text))
                matched = True
                break

        if not matched:
            # Если формат не распознан, добавляем к Speaker 0
            result.append((0, line))

    return result


# =====================================================
# Функции генерации
# =====================================================

def generate_voice_design(
    text: str,
    language: str,
    voice_description: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация речи с дизайном голоса (стриминг)."""
    global is_generating, stop_generation

    if not text or not text.strip():
        yield None, "Ошибка: Введите текст для синтеза."
        return

    if not voice_description or not voice_description.strip():
        yield None, "Ошибка: Введите описание голоса."
        return

    # Только 1.7B поддерживает VoiceDesign
    if model_size != "1.7B":
        yield None, "Ошибка: Дизайн голоса доступен только для модели 1.7B."
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели VoiceDesign..."
        tts = get_model("VoiceDesign", model_size)

        yield None, f"Генерация речи...\nТекст: {text[:50]}...\nОписание: {voice_description[:50]}..."

        # Получаем реальный код языка
        lang_code = language.split()[0] if language != "Auto" else "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        start_time = time.time()

        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=lang_code,
            instruct=voice_description.strip(),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        generation_time = time.time() - start_time
        audio_duration = len(wavs[0]) / sr

        # Сохраняем файл
        saved_path = save_audio_file(wavs[0], sr)

        status = f"Генерация завершена!\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sr, wavs[0]), status

    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def generate_voice_clone(
    ref_audio,
    ref_text: str,
    target_text: str,
    language: str,
    use_xvector_only: bool,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Клонирование голоса (стриминг)."""
    global is_generating, stop_generation

    if not target_text or not target_text.strip():
        yield None, "Ошибка: Введите текст для синтеза."
        return

    audio_tuple = audio_to_tuple(ref_audio)
    if audio_tuple is None:
        yield None, "Ошибка: Загрузите референсное аудио."
        return

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        yield None, "Ошибка: Введите текст референсного аудио или включите режим 'Только x-vector'."
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели Base..."
        tts = get_model("Base", model_size)

        yield None, f"Клонирование голоса...\nТекст: {target_text[:50]}..."

        # Получаем реальный код языка
        lang_code = "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        start_time = time.time()

        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=lang_code,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        generation_time = time.time() - start_time
        audio_duration = len(wavs[0]) / sr

        # Сохраняем файл
        saved_path = save_audio_file(wavs[0], sr)

        status = f"Клонирование завершено!\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sr, wavs[0]), status

    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def generate_with_profile(
    profile_name: str,
    target_text: str,
    language: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация с использованием сохранённого профиля голоса."""
    global is_generating, stop_generation

    if not target_text or not target_text.strip():
        yield None, "Ошибка: Введите текст для синтеза."
        return

    if not profile_name:
        yield None, "Ошибка: Выберите профиль голоса."
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, f"Загрузка профиля '{profile_name}'..."
        voice_prompt, load_msg = load_voice_profile(profile_name)

        if voice_prompt is None:
            yield None, load_msg
            return

        yield None, "Загрузка модели Base..."
        tts = get_model("Base", model_size)

        yield None, f"Генерация с профилем '{profile_name}'...\nТекст: {target_text[:50]}..."

        # Получаем реальный код языка
        lang_code = "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        start_time = time.time()

        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=lang_code,
            voice_clone_prompt=voice_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        generation_time = time.time() - start_time
        audio_duration = len(wavs[0]) / sr

        # Сохраняем файл
        saved_path = save_audio_file(wavs[0], sr)

        status = f"Генерация завершена!\n"
        status += f"Профиль: {profile_name}\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sr, wavs[0]), status

    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def generate_custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация с пресетами голосов (стриминг)."""
    global is_generating, stop_generation

    if not text or not text.strip():
        yield None, "Ошибка: Введите текст для синтеза."
        return

    if not speaker:
        yield None, "Ошибка: Выберите голос."
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели CustomVoice..."
        tts = get_model("CustomVoice", model_size)

        # Получаем реальное имя спикера
        speaker_id = speaker.split()[0] if speaker else "Vivian"
        for sid, sname in SPEAKERS.items():
            if sname == speaker:
                speaker_id = sid
                break

        yield None, f"Генерация речи...\nТекст: {text[:50]}...\nГолос: {speaker}"

        # Получаем реальный код языка
        lang_code = "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        start_time = time.time()

        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=lang_code,
            speaker=speaker_id.lower(),
            instruct=instruct.strip() if instruct else None,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        generation_time = time.time() - start_time
        audio_duration = len(wavs[0]) / sr

        # Сохраняем файл
        saved_path = save_audio_file(wavs[0], sr)

        status = f"Генерация завершена!\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sr, wavs[0]), status

    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def generate_multi_speaker(
    script: str,
    num_speakers: int,
    speaker_audios: List,
    speaker_texts: List[str],
    language: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация диалога с несколькими дикторами."""
    global is_generating, stop_generation

    if not script or not script.strip():
        yield None, "Ошибка: Введите сценарий диалога."
        return

    # Парсим скрипт
    parsed_lines = parse_multi_speaker_script(script)
    if not parsed_lines:
        yield None, "Ошибка: Не удалось распознать формат сценария."
        return

    # Проверяем, что все дикторы имеют аудио
    used_speakers = set(sp for sp, _ in parsed_lines)
    for sp in used_speakers:
        if sp >= num_speakers:
            yield None, f"Ошибка: В сценарии используется Диктор {sp}, но настроено только {num_speakers} дикторов."
            return
        audio = speaker_audios[sp] if sp < len(speaker_audios) else None
        if audio_to_tuple(audio) is None:
            yield None, f"Ошибка: Не загружено аудио для Диктора {sp}."
            return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели Base..."
        tts = get_model("Base", model_size)

        # Получаем реальный код языка
        lang_code = "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        # Создаём voice prompts для каждого диктора
        yield None, "Создание профилей голосов для дикторов..."
        voice_prompts = {}
        for sp in used_speakers:
            audio_tuple = audio_to_tuple(speaker_audios[sp])
            ref_text = speaker_texts[sp] if sp < len(speaker_texts) else None

            voice_prompts[sp] = tts.create_voice_clone_prompt(
                ref_audio=audio_tuple,
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=not bool(ref_text)
            )

        # Генерируем аудио для каждой реплики
        all_audio_chunks = []
        total_lines = len(parsed_lines)
        sample_rate = None

        start_time = time.time()

        for i, (speaker_id, text) in enumerate(parsed_lines):
            if stop_generation:
                is_generating = False
                yield None, "Генерация остановлена пользователем."
                return

            yield None, f"Генерация реплики {i+1}/{total_lines}...\nДиктор {speaker_id}: {text[:30]}..."

            wavs, sr = tts.generate_voice_clone(
                text=text,
                language=lang_code,
                voice_clone_prompt=voice_prompts[speaker_id],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            if sample_rate is None:
                sample_rate = sr

            all_audio_chunks.append(wavs[0])

            # Добавляем паузу между репликами
            pause_samples = int(0.3 * sr)  # 300ms пауза
            all_audio_chunks.append(np.zeros(pause_samples, dtype=np.float32))

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        # Объединяем все аудио
        final_audio = np.concatenate(all_audio_chunks)

        generation_time = time.time() - start_time
        audio_duration = len(final_audio) / sample_rate

        # Сохраняем файл
        saved_path = save_audio_file(final_audio, sample_rate)

        status = f"Multi-speaker генерация завершена!\n"
        status += f"Дикторов: {len(used_speakers)}\n"
        status += f"Реплик: {total_lines}\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sample_rate, final_audio), status

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def stop_generation_fn():
    """Остановка генерации."""
    global stop_generation
    stop_generation = True
    return "Остановка генерации..."


# =====================================================
# Локальные голоса
# =====================================================

def get_local_voices() -> Dict[str, str]:
    """Получение списка локальных голосов (включая подпапки)."""
    voices = {}
    supported_ext = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    # Рекурсивный поиск во всех подпапках
    for path in VOICES_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_ext:
            voices[path.stem] = str(path)

    return dict(sorted(voices.items()))


def get_voice_text(voice_name: str) -> Optional[str]:
    """Получение текста для голоса (если есть)."""
    # Ищем в основной папке
    txt_path = VOICES_DIR / f"{voice_name}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    # Ищем в подпапках
    for txt_path in VOICES_DIR.rglob(f"{voice_name}.txt"):
        return txt_path.read_text(encoding="utf-8").strip()
    return None


# =====================================================
# Построение интерфейса
# =====================================================

def build_ui():
    """Построение интерфейса Gradio."""

    # CSS стили
    css = """
    .gradio-container {max-width: none !important;}

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
    }

    .settings-card {
        background: rgba(30, 41, 59, 0.95) !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .generation-card {
        background: rgba(30, 41, 59, 0.95) !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .speaker-block {
        background: #1e293b !important;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }

    .tab-nav button {
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
    }

    /* Исправление белых рамок */
    .examples-table, .examples-table tbody, .examples-table tr, .examples-table td {
        background: transparent !important;
        border: none !important;
    }

    .prose {
        color: #e2e8f0 !important;
    }
    """

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="purple",
    ).dark()

    with gr.Blocks(theme=theme, css=css, title=APP_NAME) as demo:
        # Заголовок
        gr.HTML(f"""
        <div class="main-header">
            <h1>{APP_NAME} v{APP_VERSION}</h1>
            <p>Синтез речи с Multi-speaker режимом и профилями голосов</p>
            <p style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                Собрал <a href="https://t.me/nerual_dreming" target="_blank" style="color: white;">Nerual Dreaming</a> -
                основатель <a href="https://artgeneration.me/" target="_blank" style="color: white;">ArtGeneration.me</a>,
                техноблогер и нейро-евангелист.
            </p>
        </div>
        """)

        with gr.Tabs() as tabs:
            # =====================================================
            # Вкладка 1: Пресеты голосов (CustomVoice)
            # =====================================================
            with gr.Tab("Пресеты голосов", id="custom"):
                gr.Markdown("### Синтез речи с предустановленными голосами")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        cv_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить...",
                            value="Привет! Это демонстрация системы синтеза речи Qwen3-TTS. Она поддерживает русский язык и множество других языков."
                        )

                        with gr.Row():
                            cv_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Russian"],
                                interactive=True,
                            )
                            cv_speaker = gr.Dropdown(
                                label="Голос",
                                choices=list(SPEAKERS.values()),
                                value=SPEAKERS["Vivian"],
                                interactive=True,
                            )

                        cv_instruct = gr.Textbox(
                            label="Стиль (опционально)",
                            lines=2,
                            placeholder="Например: Говорить радостно и энергично",
                        )

                        with gr.Row():
                            cv_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            cv_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            cv_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1
                            )
                            cv_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            cv_generate_btn = gr.Button("Сгенерировать", variant="primary", scale=2)
                            cv_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        cv_audio_out = gr.Audio(
                            label="Результат",
                            type="numpy",
                            interactive=False,
                        )
                        cv_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                cv_generate_btn.click(
                    generate_custom_voice,
                    inputs=[cv_text, cv_language, cv_speaker, cv_instruct, cv_model_size, cv_max_tokens, cv_temperature, cv_top_p],
                    outputs=[cv_audio_out, cv_status],
                )
                cv_stop_btn.click(stop_generation_fn, outputs=[cv_status])

            # =====================================================
            # Вкладка 2: Клонирование голоса (Base)
            # =====================================================
            with gr.Tab("Клонирование голоса", id="clone"):
                gr.Markdown("### Клонирование голоса из референсного аудио")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        # Выбор голоса из библиотеки
                        local_voices = get_local_voices()
                        vc_voice_preset = gr.Dropdown(
                            label="Выбрать голос из библиотеки",
                            choices=["-- Загрузить свой --"] + list(local_voices.keys()),
                            value="-- Загрузить свой --",
                            interactive=True,
                        )
                        vc_refresh_voices = gr.Button("Обновить список", size="sm")

                        vc_ref_audio = gr.Audio(
                            label="Референсное аудио (голос для клонирования)",
                            type="numpy",
                            sources=["upload", "microphone"],
                        )
                        vc_ref_text = gr.Textbox(
                            label="Текст референсного аудио",
                            lines=2,
                            placeholder="Введите текст, который произносится в референсном аудио...",
                        )

                        def load_voice_preset(voice_name):
                            if voice_name == "-- Загрузить свой --" or not voice_name:
                                return None, ""
                            voices = get_local_voices()
                            path = voices.get(voice_name)
                            if path:
                                import soundfile as sf
                                wav, sr = sf.read(path)
                                ref_text = get_voice_text(voice_name) or ""
                                return (sr, wav), ref_text
                            return None, ""

                        def refresh_voice_list():
                            voices = get_local_voices()
                            return gr.update(choices=["-- Загрузить свой --"] + list(voices.keys()))

                        vc_voice_preset.change(
                            load_voice_preset,
                            inputs=[vc_voice_preset],
                            outputs=[vc_ref_audio, vc_ref_text],
                        )
                        vc_refresh_voices.click(refresh_voice_list, outputs=[vc_voice_preset])

                        vc_xvector_only = gr.Checkbox(
                            label="Только x-vector (без текста референса, качество ниже)",
                            value=False,
                        )

                        gr.Markdown("---")

                        vc_target_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить клонированным голосом...",
                        )

                        with gr.Row():
                            vc_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Auto"],
                                interactive=True,
                            )
                            vc_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            vc_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            vc_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1
                            )
                            vc_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            vc_generate_btn = gr.Button("Клонировать и озвучить", variant="primary", scale=2)
                            vc_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        vc_audio_out = gr.Audio(
                            label="Результат",
                            type="numpy",
                            interactive=False,
                        )
                        vc_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                        # Загрузка голосов из облака
                        with gr.Accordion("Загрузить голоса из облака", open=False):
                            gr.Markdown(f"*Репозиторий: `{CLOUD_VOICES_REPO}`*")

                            vc_cloud_status = gr.Textbox(
                                label="Статус",
                                interactive=False,
                                value="Нажмите 'Загрузить список' для получения доступных голосов",
                            )

                            vc_load_cloud_btn = gr.Button("Загрузить список", variant="secondary")

                            vc_cloud_voices = gr.CheckboxGroup(
                                label="Доступные голоса",
                                choices=[],
                                interactive=True,
                            )

                            vc_download_btn = gr.Button("Скачать выбранные", variant="primary")
                            vc_download_status = gr.Textbox(
                                label="Результат загрузки",
                                interactive=False,
                            )

                            def load_cloud_list_vc():
                                voices, status = get_cloud_voices_list()
                                if voices:
                                    return status, gr.update(choices=voices, value=[])
                                return status, gr.update(choices=[], value=[])

                            vc_load_cloud_btn.click(
                                load_cloud_list_vc,
                                outputs=[vc_cloud_status, vc_cloud_voices],
                            )

                            def download_selected_voices_vc(selected):
                                if not selected:
                                    return "Выберите голоса для загрузки."
                                results = []
                                for voice in selected:
                                    result = download_cloud_voice(voice)
                                    results.append(result)
                                return "\n".join(results)

                            vc_download_btn.click(
                                download_selected_voices_vc,
                                inputs=[vc_cloud_voices],
                                outputs=[vc_download_status],
                            )

                vc_generate_btn.click(
                    generate_voice_clone,
                    inputs=[vc_ref_audio, vc_ref_text, vc_target_text, vc_language, vc_xvector_only, vc_model_size, vc_max_tokens, vc_temperature, vc_top_p],
                    outputs=[vc_audio_out, vc_status],
                )
                vc_stop_btn.click(stop_generation_fn, outputs=[vc_status])

            # =====================================================
            # Вкладка 3: Multi-speaker
            # =====================================================
            with gr.Tab("Multi-speaker", id="multi"):
                gr.Markdown("### Генерация диалога с несколькими дикторами")
                gr.Markdown("""
                **Формат сценария:**
                ```
                Speaker 0: Привет, как дела?
                Speaker 1: Отлично, спасибо! А у тебя?
                Speaker 0: Тоже хорошо!
                ```
                Также поддерживаются форматы: `Диктор N:`, `Голос N:`, `[N]`
                """)

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        ms_num_speakers = gr.Slider(
                            label="Количество дикторов",
                            minimum=2, maximum=4, value=2, step=1,
                        )

                        # Блоки дикторов
                        local_voices = get_local_voices()
                        voice_choices = ["-- Загрузить свой --"] + list(local_voices.keys())

                        speaker_blocks = []
                        speaker_audios = []
                        speaker_texts = []
                        speaker_presets = []

                        for i in range(4):
                            with gr.Column(visible=(i < 2), elem_classes="speaker-block") as block:
                                gr.Markdown(f"**Диктор {i}**")
                                preset = gr.Dropdown(
                                    label="Пресет голоса",
                                    choices=voice_choices,
                                    value=voice_choices[0] if len(voice_choices) > 0 else None,
                                )
                                audio = gr.Audio(
                                    label="Аудио референса",
                                    type="numpy",
                                    sources=["upload", "microphone"],
                                )
                                text = gr.Textbox(
                                    label="Текст референса (опционально)",
                                    lines=1,
                                    placeholder="Текст произносимый в аудио...",
                                )

                                # Обработчик выбора пресета
                                def update_from_preset(preset_name, idx=i):
                                    if preset_name == "-- Загрузить свой --":
                                        return gr.update(value=None), gr.update(value="")
                                    path = local_voices.get(preset_name)
                                    if path:
                                        import soundfile as sf
                                        wav, sr = sf.read(path)
                                        ref_text = get_voice_text(preset_name) or ""
                                        return gr.update(value=(sr, wav)), gr.update(value=ref_text)
                                    return gr.update(value=None), gr.update(value="")

                                preset.change(
                                    update_from_preset,
                                    inputs=[preset],
                                    outputs=[audio, text],
                                )

                                speaker_blocks.append(block)
                                speaker_audios.append(audio)
                                speaker_texts.append(text)
                                speaker_presets.append(preset)

                        # Обновление видимости блоков
                        def update_speaker_visibility(num):
                            return [gr.update(visible=(i < num)) for i in range(4)]

                        ms_num_speakers.change(
                            update_speaker_visibility,
                            inputs=[ms_num_speakers],
                            outputs=speaker_blocks,
                        )

                        gr.Markdown("---")

                        with gr.Row():
                            ms_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Auto"],
                                interactive=True,
                            )
                            ms_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            ms_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            ms_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1
                            )
                            ms_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        ms_script = gr.Textbox(
                            label="Сценарий диалога",
                            lines=10,
                            placeholder="Speaker 0: Привет!\nSpeaker 1: Привет, как дела?",
                            value="Speaker 0: Привет! Ты уже попробовал новую модель Qwen3-TTS?\nSpeaker 1: Да, она отлично работает! Особенно впечатляет качество клонирования голоса.\nSpeaker 0: Согласен, результаты просто потрясающие!",
                        )

                        with gr.Row():
                            ms_generate_btn = gr.Button("Сгенерировать диалог", variant="primary", scale=2)
                            ms_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                        ms_audio_out = gr.Audio(
                            label="Результат",
                            type="numpy",
                            interactive=False,
                        )
                        ms_status = gr.Textbox(
                            label="Статус",
                            lines=6,
                            interactive=False,
                        )

                # Wrapper для передачи аудио дикторов
                def multi_speaker_wrapper(script, num_speakers, audio0, audio1, audio2, audio3, text0, text1, text2, text3, language, model_size, max_tokens, temperature, top_p):
                    audios = [audio0, audio1, audio2, audio3]
                    texts = [text0, text1, text2, text3]
                    yield from generate_multi_speaker(script, num_speakers, audios, texts, language, model_size, max_tokens, temperature, top_p)

                ms_generate_btn.click(
                    multi_speaker_wrapper,
                    inputs=[ms_script, ms_num_speakers,
                            speaker_audios[0], speaker_audios[1], speaker_audios[2], speaker_audios[3],
                            speaker_texts[0], speaker_texts[1], speaker_texts[2], speaker_texts[3],
                            ms_language, ms_model_size, ms_max_tokens, ms_temperature, ms_top_p],
                    outputs=[ms_audio_out, ms_status],
                )
                ms_stop_btn.click(stop_generation_fn, outputs=[ms_status])

            # =====================================================
            # Вкладка 4: Дизайн голоса (VoiceDesign)
            # =====================================================
            with gr.Tab("Дизайн голоса", id="design"):
                gr.Markdown("### Создание голоса по текстовому описанию")
                gr.Markdown("*Доступно только для модели 1.7B*")
                gr.Markdown("**Примечание:** Описание голоса можно писать на русском, но на английском результат лучше.")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        vd_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить...",
                            value="Привет! Как твои дела? Это демонстрация синтеза речи."
                        )

                        vd_language = gr.Dropdown(
                            label="Язык",
                            choices=list(LANGUAGES.values()),
                            value=LANGUAGES["Russian"],
                            interactive=True,
                        )

                        vd_description = gr.Textbox(
                            label="Описание голоса (лучше на английском)",
                            lines=3,
                            placeholder="Young female voice, warm and friendly...",
                            value="Young female voice, warm and friendly, speaking with enthusiasm"
                        )

                        vd_model_size = gr.Dropdown(
                            label="Размер модели",
                            choices=["1.7B"],
                            value="1.7B",
                            interactive=False,
                        )

                        with gr.Accordion("Параметры генерации", open=False):
                            vd_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            vd_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1
                            )
                            vd_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            vd_generate_btn = gr.Button("Сгенерировать", variant="primary", scale=2)
                            vd_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        vd_audio_out = gr.Audio(
                            label="Результат",
                            type="numpy",
                            interactive=False,
                        )
                        vd_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                        # Примеры промптов
                        gr.Markdown("**Готовые промпты** (кликни для применения)")
                        gr.Examples(
                            examples=[
                                ["Female, 25 years old, warm soprano voice, speaking with a gentle smile and soft tone"],
                                ["Male, 35 years old, deep baritone, confident and authoritative, measured pace"],
                                ["Male, 17 years old, tenor range, gaining confidence - deeper breath support now"],
                                ["Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice"],
                                ["Elderly woman, 70 years old, soft and caring, speaking slowly with wisdom and warmth"],
                                ["Young child, 8 years old, playful and cheerful, high-pitched with innocent excitement"],
                                ["Professional news anchor, clear articulation, neutral tone, moderate pace"],
                                ["Speak with intense anger, sharp emphasis on words, aggressive tone"],
                                ["Whisper softly with mystery, secretive and intimate, barely audible"],
                                ["Exhausted and sleepy voice, slow drowsy delivery, yawning between words"],
                                ["Speak with genuine surprise and disbelief, voice rising in pitch"],
                                ["Seductive female voice, low and breathy, slow sensual pace"],
                            ],
                            inputs=[vd_description],
                            label=""
                        )

                vd_generate_btn.click(
                    generate_voice_design,
                    inputs=[vd_text, vd_language, vd_description, vd_model_size, vd_max_tokens, vd_temperature, vd_top_p],
                    outputs=[vd_audio_out, vd_status],
                )
                vd_stop_btn.click(stop_generation_fn, outputs=[vd_status])


    return demo


# =====================================================
# Точка входа
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"{APP_NAME} v{APP_VERSION}")
    print("=" * 60)
    print()

    # Определяем устройство
    device = get_device()
    print(f"Устройство: {device.upper()}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print()
    print(f"Директория голосов: {VOICES_DIR}")
    print(f"Директория профилей: {PROFILES_DIR}")
    print(f"Директория вывода: {OUTPUT_DIR}")

    local_voices = get_local_voices()
    print(f"Локальных голосов: {len(local_voices)}")

    profiles = list_voice_profiles()
    print(f"Сохранённых профилей: {len(profiles)}")

    print()
    print("Запуск веб-интерфейса...")
    print()

    # Строим и запускаем интерфейс
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
