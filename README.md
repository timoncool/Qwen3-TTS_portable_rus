# Qwen3-TTS Portable PRO

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
</p>

**Портативная русскоязычная версия** мощной системы синтеза речи Qwen3-TTS с поддержкой:
- 🎙️ **Клонирование голоса** — создание копии голоса из короткого аудио
- 🎨 **Дизайн голоса** — генерация голоса по текстовому описанию
- 👥 **Multi-speaker режим** — создание диалогов с несколькими дикторами
- 🌍 **Мультиязычность** — поддержка 10+ языков включая русский

## Другие портативные нейросети

| Проект | Описание |
|--------|----------|
| [Foundation Music Lab](https://github.com/timoncool/RC-stable-audio-tools-portable) | Генерация музыки + таймлайн-редактор |
| [VibeVoice ASR](https://github.com/timoncool/VibeVoice_ASR_portable_ru) | Распознавание речи (ASR) |
| [LavaSR](https://github.com/timoncool/LavaSR_portable_ru) | Улучшение качества аудио |
| [SuperCaption Qwen3-VL](https://github.com/timoncool/SuperCaption_Qwen3-VL) | Генерация описаний изображений |
| [VideoSOS](https://github.com/timoncool/videosos) | AI-видеопродакшн в браузере |

## Авторы

**Собрал [Nerual Dreaming](https://t.me/nerual_dreming)** — основатель [ArtGeneration.me](https://artgeneration.me/), техноблогер и нейро-евангелист.

**[Нейро-Софт](https://t.me/neuroport)** — репаки и портативки полезных нейросетей

## Установка

1. Скачайте и распакуйте архив
2. Запустите `portable/install.bat` для установки зависимостей
3. Запустите `portable/run.bat` для запуска приложения

## Системные требования

- **ОС:** Windows 10/11
- **GPU:** NVIDIA с поддержкой CUDA (минимум 8GB VRAM)
- **RAM:** 16GB+
- **Интернет:** Требуется при первом запуске для загрузки моделей

## Возможности

### Пресеты голосов
Использование встроенных голосовых пресетов (Aiden, Dylan, Eric, Serena и др.)

### Клонирование голоса
Загрузите короткое аудио (5-30 сек) и получите синтез речи этим голосом.

### Multi-speaker
Создавайте диалоги с несколькими дикторами в формате:
```
Speaker 0: Привет! Как дела?
Speaker 1: Отлично, спасибо!
```

### Дизайн голоса
Опишите желаемый голос текстом на английском языке:
```
Young female voice, warm and friendly, speaking with enthusiasm
```

## Голосовые пакеты

При установке автоматически загружается голосовой пакет с русскими голосами.
Дополнительные голоса можно загрузить из облака прямо в приложении.

## Лицензия

Модель Qwen3-TTS распространяется под лицензией [Qwen License](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT).

## Оригинальный проект

- 🤗 [Hugging Face](https://huggingface.co/collections/Qwen/qwen3-tts)
- 📑 [Blog](https://qwen.ai/blog?id=qwen3tts-0115)
- 📑 [Paper](https://arxiv.org/abs/2601.15621)
