import gc
import os
import random
import numpy as np
from scipy.signal.windows import hann
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path
import tempfile
import argparse
import librosa
from audiosr import build_model, super_resolution
from scipy import signal
import pyloudnorm as pyln


import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    """
    Согласовывает размеры двух массивов NumPy, обрезая или дополняя нулями.
    Работает как с одномерными, так и с двумерными массивами (моно и стерео).
    """
    if (len(array_1.shape) == 1) & (len(array_2.shape) == 1):  # Оба моно
        if array_1.shape[0] > array_2.shape[0]:
            array_1 = array_1[:array_2.shape[0]]
        elif array_1.shape[0] < array_2.shape[0]:
            array_1 = np.pad(array_1, ((array_2.shape[0] - array_1.shape[0], 0)), 'constant', constant_values=0)
    elif (len(array_1.shape) == 2) & (len(array_2.shape) == 2):  # Оба стерео
        if array_1.shape[1] > array_2.shape[1]:
            array_1 = array_1[:,:array_2.shape[1]]
        elif array_1.shape[1] < array_2.shape[1]:
            padding = array_2.shape[1] - array_1.shape[1]
            array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    elif (len(array_1.shape) == 1) & (len(array_2.shape) == 2): # array_1 моно, array_2 стерео
        if array_1.shape[0] > array_2.shape[1]:
            array_1 = array_1[:array_2.shape[1]]
        elif array_1.shape[0] < array_2.shape[1]:
            padding = array_2.shape[1] - array_1.shape[0]
            array_1 = np.pad(array_1, (0, padding), 'constant', constant_values=0)
            array_1 = np.expand_dims(array_1, axis=0)  # Преобразование в (1, N)
    elif (len(array_1.shape) == 2) & (len(array_2.shape) == 1):  # array_1 стерео, array_2 моно
        if array_1.shape[1] > array_2.shape[0]:
             array_1 = array_1[:, :array_2.shape[0]]
        elif array_1.shape[1] < array_2.shape[0]:
            padding = array_2.shape[0] - array_1.shape[1]
            array_1 = np.pad(array_1, ((0, 0), (0, padding)), 'constant')

    return array_1


def lr_filter(audio, cutoff, filter_type, order=12, sr=48000):
    """
    Применяет фильтр Баттерворта (lowpass или highpass) к аудио.

    Args:
        audio: Аудиоданные (NumPy array).
        cutoff: Частота среза.
        filter_type: 'lowpass' или 'highpass'.
        order: Порядок фильтра.
        sr: Частота дискретизации.

    Returns:
        Отфильтрованные аудиоданные.
    """
    audio = audio.T  # Транспонируем для корректной работы с sosfiltfilt
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T  # Возвращаем транспонированный результат


class Predictor(BasePredictor):
    def setup(self, model_name="basic", device="auto"):
        """
        Инициализирует модель AudioSR.

        Args:
            model_name: Имя модели ('basic', 'speech', 'music', 'general', 'multiband').
            device: Устройство ('auto', 'cuda', 'cpu').
        """
        self.model_name = model_name
        self.device = device
        self.sr = 48000  # Фиксированная частота дискретизации модели
        print(f"Loading {self.model_name} Model...")
        try:
            self.audiosr = build_model(model_name=self.model_name, device=self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        print("Model loaded!")

    def process_audio(self, input_file, chunk_size=5.12, overlap=0.1, seed=None, guidance_scale=3.5, ddim_steps=50, multiband_ensemble=False, input_cutoff=12000):
        """
        Обрабатывает аудиофайл по частям (chunks), применяя суперразрешение.

        Args:
            input_file: Путь к входному аудиофайлу.
            chunk_size: Размер фрагмента в секундах.
            overlap: Перекрытие между фрагментами в долях от chunk_size.
            seed:  Seed для генератора случайных чисел.
            guidance_scale: Параметр classifier-free guidance.
            ddim_steps: Количество шагов DDIM.
            multiband_ensemble: Включить многополосную обработку.
            input_cutoff: Частота среза для многополосной обработки.

        Returns:
            Обработанное аудио (NumPy array).
        """
        try:
            audio, sr = librosa.load(input_file, sr=None, mono=False)  # Загружаем с исходной частотой
        except Exception as e:
            print(f"Error loading audio file: {e}")
            raise

        # Если входное аудио имеет частоту дискретизации выше, чем input_cutoff * 2,
        # выполняем downsampling до input_cutoff * 2.
        if sr > input_cutoff * 2:
            print(f"Downsampling input audio from {sr} Hz to {input_cutoff * 2} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=input_cutoff * 2)
            sr = input_cutoff * 2
        elif sr < input_cutoff * 2 and sr != self.sr :
            print(f"Upsampling input audio from {sr} Hz to {input_cutoff * 2} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=input_cutoff * 2)
            sr = input_cutoff * 2

        audio = audio.T  # Транспонируем, чтобы первая размерность была временем
        print(f"audio.shape = {audio.shape}")
        print(f"input cutoff = {input_cutoff}")

        is_stereo = len(audio.shape) == 2
        audio_channels = [audio] if not is_stereo else [audio[:, 0], audio[:, 1]]  # Разделяем на каналы
        print("audio is stereo" if is_stereo else "Audio is mono")

        chunk_samples = int(chunk_size * sr)  # Размер фрагмента в сэмплах (исходная частота)
        overlap_samples = int(overlap * chunk_samples)  # Перекрытие в сэмплах
        output_chunk_samples = int(chunk_size * self.sr)  # Размер выходного фрагмента в сэмплах (48000 Гц)
        output_overlap_samples = int(overlap * output_chunk_samples)  # Перекрытие выходного фрагмента
        enable_overlap = overlap > 0
        print(f"enable_overlap = {enable_overlap}")
        
        def process_chunks(audio):
            """Разбивает аудио на фрагменты с перекрытием."""
            chunks = []
            original_lengths = []  # Список для хранения исходных длин фрагментов
            start = 0
            while start < len(audio):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                # Если последний фрагмент короче chunk_samples, дополняем его нулями
                if len(chunk) < chunk_samples:
                    original_lengths.append(len(chunk))  # Сохраняем исходную длину
                    chunk = np.concatenate([chunk, np.zeros(chunk_samples - len(chunk))])
                else:
                    original_lengths.append(chunk_samples) # Сохраняем полную длину
                chunks.append(chunk)
                # Сдвигаем начало следующего фрагмента с учетом перекрытия
                start += chunk_samples - overlap_samples if enable_overlap else chunk_samples
            return chunks, original_lengths

        # Разбиваем каждый канал на фрагменты
        chunks_per_channel = [process_chunks(channel) for channel in audio_channels]

        # Вычисляем общую длину выходного аудио (в сэмплах)
        sample_rate_ratio = self.sr / sr
        total_length = int(len(chunks_per_channel[0][0]) * output_chunk_samples - (len(chunks_per_channel[0][0]) - 1) * (output_overlap_samples if enable_overlap else 0))
        reconstructed_channels = [np.zeros((1, total_length)) for _ in audio_channels]


        meter_before = pyln.Meter(sr)  # Измеритель громкости для входного аудио
        meter_after = pyln.Meter(self.sr)  # Измеритель громкости для выходного аудио (48000 Гц)

        # Обрабатываем фрагменты для каждого канала
        for ch_idx, (chunks, original_lengths) in enumerate(chunks_per_channel):
            for i, chunk in enumerate(chunks):
                loudness_before = meter_before.integrated_loudness(chunk)  # Громкость до обработки
                print(f"Processing chunk {i+1} of {len(chunks)} for {'Left/Mono' if ch_idx == 0 else 'Right'} channel")

                # Временный файл для передачи фрагмента в AudioSR
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                    sf.write(temp_wav.name, chunk, sr)  # Сохраняем фрагмент во временный файл

                    # Применяем суперразрешение с помощью AudioSR
                    out_chunk = super_resolution(
                        self.audiosr,
                        temp_wav.name,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        ddim_steps=ddim_steps,
                        latent_t_per_second=12.8  # Важный параметр!
                    )

                    out_chunk = out_chunk[0] # убираем лишнюю размерность батча
                    # Обрезаем выходной фрагмент до нужной длины, основываясь на *исходной* длине
                    # входного фрагмента и соотношении частот дискретизации.
                    num_samples_to_keep = int(original_lengths[i] * sample_rate_ratio)
                    out_chunk = out_chunk[:, :num_samples_to_keep].squeeze()

                    # Проверяем длину out_chunk
                    min_length = 2048 # Примерное значение для block_size = 0.2 при sr=48000
                    if len(out_chunk) > min_length:
                        # Нормализуем громкость с помощью pyloudnorm
                        loudness_after = meter_after.integrated_loudness(out_chunk)
                        out_chunk = pyln.normalize.loudness(out_chunk, loudness_after, loudness_before)
                    else:
                        print(f"Chunk {i+1} is too short for loudness normalization, skipping.")
                        # Если фрагмент слишком короткий, можно просто пропустить нормализацию
                        # Или, как вариант, дополнить его нулями (см. вариант 3 выше)
                        out_chunk = np.pad(out_chunk, (0, min_length - len(out_chunk)), 'constant')

                    # Применяем плавное затухание/нарастание (fade-in/fade-out) на краях фрагментов
                    if enable_overlap:
                        # Фактическое количество сэмплов перекрытия (может быть меньше, чем output_overlap_samples)
                        actual_overlap_samples = min(output_overlap_samples, num_samples_to_keep)
                        fade_out = np.linspace(1., 0., actual_overlap_samples)  # Линейное затухание
                        fade_in = np.linspace(0., 1., actual_overlap_samples)   # Линейное нарастание

                        if i == 0:  # Первый фрагмент: только затухание в конце
                            out_chunk[-actual_overlap_samples:] *= fade_out
                        elif i < len(chunks) - 1:  # Средние фрагменты: нарастание в начале, затухание в конце
                            out_chunk[:actual_overlap_samples] *= fade_in
                            out_chunk[-actual_overlap_samples:] *= fade_out
                        else:  # Последний фрагмент: только нарастание в начале
                            out_chunk[:actual_overlap_samples] *= fade_in

                # Добавляем обработанный фрагмент в соответствующий канал восстановленного аудио
                start = int(i * (output_chunk_samples - output_overlap_samples if enable_overlap else output_chunk_samples))
                end = start + out_chunk.shape[0]
                reconstructed_channels[ch_idx][0, start:end] += out_chunk.flatten()

        # Объединяем каналы, если стерео
        reconstructed_audio = np.stack(reconstructed_channels, axis=-1) if is_stereo else reconstructed_channels[0]

        # Многополосная обработка (multiband ensemble)
        if multiband_ensemble:
            try:
                low, _ = librosa.load(input_file, sr=self.sr, mono=False) # Загружаем с частотой дискретизации модели
            except Exception as e:
                print(f"Error loading audio for multiband processing: {e}")
                raise

            # Применяем фильтры к reconstructed_audio ДО согласования размеров
            high = lr_filter(reconstructed_audio.T, input_cutoff - 1000, 'highpass', order=10)
            high = lr_filter(high, 23000, 'lowpass', order=2)  # Дополнительный ФНЧ
            low = lr_filter(low.T, input_cutoff - 1000, 'lowpass', order=10)

            # Согласовываем размеры ПОСЛЕ фильтрации
            if len(low.shape) == 1 and len(high.shape) == 2:
                # Если low - моно, а high - стерео, преобразуем low в стерео
                low = np.stack([low, low], axis=0)
            elif len(high.shape) == 1 and len(low.shape) == 2:
                high = np.stack([high, high], axis=0)

            output = match_array_shapes(high, low)
            output = low + output  # Складываем low и high (уже согласованные)
        else:
            output = reconstructed_audio

        return output


    def predict(self,
        input_file: Path = Input(description="Audio to upsample"),
        ddim_steps: int = Input(description="Number of inference steps", default=50, ge=10, le=500),
        guidance_scale: float = Input(description="Scale for classifier free guidance", default=3.5, ge=1.0, le=20.0),
        overlap: float = Input(description="overlap size", default=0.04),
        chunk_size: float = Input(description="chunksize", default=10.24),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
        model_name: str = Input(description="AudioSR model name", default="basic", choices=["basic", "speech"]),
        multiband_ensemble: bool = Input(description="Enable multiband ensemble", default=False),
        input_cutoff: int = Input(description="Input cutoff frequency for multiband", default=12000)
    ) -> Path:      
        """
        Запускает процесс суперразрешения на входном аудиофайле.

        Args:
            input_file: Путь к входному аудиофайлу.
            ddim_steps: Количество шагов DDIM.
            guidance_scale: Параметр classifier-free guidance.
            overlap: Перекрытие между фрагментами (в долях от chunk_size).
            chunk_size: Размер фрагмента в секундах.
            seed:  Seed для генератора случайных чисел.
            model_name: Имя модели AudioSR.
            multiband_ensemble: Включить многополосную обработку.
            input_cutoff: Частота среза для многополосной обработки.

        Returns:
            Путь к выходному аудиофайлу.
        """

        # Обновляем модель, если указана другая
        if self.model_name != model_name:
            del self.audiosr
            gc.collect()
            torch.cuda.empty_cache()
            self.setup(model_name=model_name)

        if seed is None:  # Если seed не указан, генерируем случайный
            seed = random.randint(0, 2**32 - 1)
        print(f"Setting seed to: {seed}")
        print(f"overlap = {overlap}")
        print(f"guidance_scale = {guidance_scale}")
        print(f"ddim_steps = {ddim_steps}")
        print(f"chunk_size = {chunk_size}")
        print(f"multiband_ensemble = {multiband_ensemble}")
        print(f"input file = {os.path.basename(input_file)}")

        # Создаем временный файл для вывода
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = output_file.name

        # Обрабатываем аудио
        waveform = self.process_audio(
            input_file,
            chunk_size=chunk_size,
            overlap=overlap,
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            multiband_ensemble=multiband_ensemble,
            input_cutoff=input_cutoff
        )
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)  # Убираем лишнюю размерность, если она есть

        # Сохраняем обработанное аудио во временный файл
        sf.write(output_path, data=waveform.T, samplerate=self.sr)  # Используем self.sr
        print(f"file created: {output_path}")

        del waveform  # Освобождаем память
        gc.collect()
        torch.cuda.empty_cache()

        return Path(output_path)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AudioSR Upsampling Script")
    parser.add_argument("--input", help="Path to input audio file", required=True)
    parser.add_argument("--output", help="Output folder", required=True)  # Изменено на папку
    parser.add_argument("--ddim_steps", help="Number of ddim steps", type=int, default=50)
    parser.add_argument("--chunk_size", help="chunk size", type=float, default=10.24)
    parser.add_argument("--guidance_scale", help="Guidance scale value", type=float, default=3.5)
    parser.add_argument("--seed", help="Seed value, 0 = random seed", type=int, default=0)
    parser.add_argument("--overlap", help="overlap value", type=float, default=0.04)
    parser.add_argument("--multiband_ensemble", type=bool, help="Use multiband ensemble with input", default=False) # Добавлено default
    parser.add_argument("--input_cutoff", help="Define the crossover of audio input in the multiband ensemble", type=int, default=12000)
    parser.add_argument("--model_name", help="AudioSR model name", type=str, default="basic", choices=["basic", "speech"])

    args = parser.parse_args()

    input_file_path = args.input
    output_folder = args.output # Теперь это папка
    ddim_steps = args.ddim_steps
    chunk_size = args.chunk_size
    guidance_scale = args.guidance_scale
    seed = args.seed
    overlap = args.overlap
    input_cutoff = args.input_cutoff
    multiband_ensemble = args.multiband_ensemble
    model_name = args.model_name


    # Создаем папку вывода, если она не существует
    os.makedirs(output_folder, exist_ok=True)

    # Создаем экземпляр Predictor и настраиваем его
    p = Predictor()
    p.setup(model_name=model_name)

    # Формируем имя выходного файла на основе имени входного
    base_name = os.path.basename(input_file_path)
    name_without_ext, _ = os.path.splitext(base_name)
    output_file_path = os.path.join(output_folder, f"SR_{name_without_ext}.wav")


    # Запускаем предсказание (обработку)
    out = p.predict(
        input_file_path,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        multiband_ensemble=multiband_ensemble,
        input_cutoff=input_cutoff
    )

    # Перемещаем выходной файл из временного в нужное место
    os.replace(str(out), output_file_path)
    print(f"Final output file saved to: {output_file_path}")

    del p  # Освобождаем ресурсы
    gc.collect()
    torch.cuda.empty_cache()
