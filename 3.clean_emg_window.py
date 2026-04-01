import numpy as np
import pandas as pd
import os
import gc
import re
from collections import deque
from tqdm import tqdm
import time
import ast

# --- Signal Processing Libraries ---
from scipy.signal import butter, filtfilt, iirnotch
import pywt # Make sure PyWavelets is installed: pip install PyWavelets

# --- Configuration (Adjust for Sliding Window) ---
sampling_rate = 2000  # Hz
window_size_ms = 1000  # ms (Window duration: 1 second)
overlap_ms = 0  # ms (No overlap)
window_size_samples = int(window_size_ms * sampling_rate / 1000)  # Calculate samples: 1000ms * 2000Hz / 1000 = 2000 samples
# Removed step_size_samples as per request. Deque will advance by window_size_samples for non-overlapping windows.

chunksize = 1000 # Chunk size for reading processed_*.csv files
folder_path = r'C:\Users\พีค\OneDrive\เดสก์ท็อป\Data sci (NSTDA)\DataPrep'  # Folder containing processed_*.csv files
output_folder = r'C:\Users\พีค\OneDrive\เดสก์ท็อป\Data sci (NSTDA)\train&test_window_2000_stepss_0'  # Output folder for cleaned single-grasp files
num_emg_channels = 12

# --- Signal Processing Parameters (Defined here for clarity) ---
# Bandpass Filter
lowcut_freq = 20  # Hz, Minimum frequency to pass
highcut_freq = 450 # Hz, Maximum frequency to pass
filter_order = 4   # Order of Butterworth filter

# Notch Filter (for electrical noise, e.g., 50Hz or 60Hz)
notch_freq = 50    # Hz (For Thailand, use 50Hz)
notch_Q = 30       # Quality factor for Notch filter (High Q = narrow notch)

# DWT Denoising (reducing noise with Wavelet Transform)
dwt_wavelet = 'db4' # Type of Wavelet (e.g., 'db4', 'sym8')
dwt_level = 2       # Level of Decomposition for DWT (Adjusted from 3 to 2 to avoid boundary effects)
epsilon = 1e-10     # Small value to prevent division by zero in DWT thresholding

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def safe_parse_emg(emg_string):
    """
    Converts EMG signal from String to NumPy array, handling potential errors.
    """
    emg_string = emg_string.strip()
    if not emg_string:
        return np.array([])
    # Remove surrounding brackets if present, otherwise ast.literal_eval expects a single value
    if emg_string.startswith("[") and emg_string.endswith("]"):
        emg_string = emg_string[1:-1]
    try:
        # Attempt to evaluate as a list first
        emg_list = ast.literal_eval(f"[{emg_string}]")
        return np.array(emg_list, dtype=np.float32)
    except (ValueError, SyntaxError) as e:
        # Fallback to np.fromstring for comma-separated values if ast.literal_eval fails
        try:
            # Replace multiple spaces with single commas, then strip leading/trailing commas
            cleaned_string = re.sub(r'\s+', ',', emg_string).strip(',')
            return np.fromstring(cleaned_string, sep=',', dtype=np.float32)
        except Exception as e:
            # print(f"⚠️ Error parsing EMG: {e} | Data: '{emg_string[:100]}...'") # Comment out to reduce console output
            return np.array([])
    except Exception as e:
        # print(f"⚠️ Error parsing EMG (initial): {e} | Data: '{emg_string[:100]}...'") # Comment out to reduce console output
        return np.array([])

def safe_parse_grasp_ids(grasp_id):
    """
    Converts Grasp ID from String or Value to a List of Integer IDs.
    """
    if isinstance(grasp_id, str):
        grasp_id = grasp_id.strip()
        if not grasp_id:
            return []
        if grasp_id.startswith("[") and grasp_id.endswith("]"):
            try:
                return [int(x) for x in ast.literal_eval(grasp_id)]
            except (ValueError, SyntaxError) as e:
                # print(f"⚠️ Error parsing Grasp IDs (string): {e} | Data: '{grasp_id[:100]}...'") # Comment out
                return []
        else:
            try:
                return [int(grasp_id)]
            except ValueError:
                # print(f"⚠️ Error parsing single Grasp ID (string): '{grasp_id}'") # Comment out
                return []
    elif isinstance(grasp_id, (int, np.int64)):
        return [int(grasp_id)]
    elif isinstance(grasp_id, list):
        try:
            return [int(x) for x in grasp_id]
        except ValueError:
            # print(f"⚠️ Error parsing Grasp IDs (list): '{grasp_id}'") # Comment out
            return []
    else:
        # print(f"⚠️ Error: Unknown type for Grasp ID: {type(grasp_id)} | Value: '{grasp_id}'") # Comment out
        return []

def safe_parse_grasp_repetition(repetition_id):
    """
    Converts Grasp Repetition ID. Returns None if 0 or problematic.
    """
    if isinstance(repetition_id, (int, np.int64)):
        if repetition_id == 0:
            return None
        return int(repetition_id)
    elif isinstance(repetition_id, str):
        repetition_id = repetition_id.strip()
        if not repetition_id:
            return None
        try:
            rep_id = int(repetition_id)
            if rep_id == 0:
                return None
            return rep_id
        except ValueError:
            # print(f"⚠️ Error parsing Grasp Repetition ID (string): '{repetition_id}'") # Comment out
            return None
    else:
        # print(f"⚠️ Error: Unknown type for Grasp Repetition ID: {type(repetition_id)}, Value: '{repetition_id}'") # Comment out
        return None

# --- New function for Signal Preprocessing ---
def apply_preprocessing(emg_data_flat, sampling_rate, num_channels,
                        lowcut, highcut, filter_order,
                        notch_freq, notch_Q,
                        dwt_wavelet, dwt_level, epsilon):
    """
    Applies Bandpass filtering, Notch filtering, and DWT-based denoising to EMG data.
    
    Args:
        emg_data_flat (np.array): Flat EMG data for one Time Segment
                                  (e.g., one row from CSV), expected shape (num_time_points * num_channels,)
        sampling_rate (int): Sampling rate in Hz
        num_channels (int): Number of EMG channels
        lowcut (float): Lower cutoff frequency for Bandpass filter
        highcut (float): Upper cutoff frequency for Bandpass filter
        filter_order (int): Order of Butterworth filter
        notch_freq (float): Frequency to notch out (e.g., 50 or 60 Hz)
        notch_Q (float): Quality factor for Notch filter
        dwt_wavelet (str): Wavelet name for DWT (e.g., 'db4')
        dwt_level (int): Decomposition level for DWT
        epsilon (float): Small value to prevent division by zero

    Returns:
        np.array: Processed EMG data, flattened back to (num_time_points * num_channels,)
    """
    if emg_data_flat.size == 0:
        return emg_data_flat

    # Reshape data to (number of time points in segment, number of channels) for per-channel processing
    num_time_points_in_segment = emg_data_flat.size // num_channels
    if num_time_points_in_segment == 0:
        return emg_data_flat
    
    emg_data_reshaped = emg_data_flat.reshape(num_time_points_in_segment, num_channels)
    processed_emg_data = np.zeros_like(emg_data_reshaped, dtype=np.float32)

    nyquist = 0.5 * sampling_rate

    # Calculate Bandpass filter coefficients
    normalized_lowcut = lowcut / nyquist
    normalized_highcut = highcut / nyquist
    
    # Ensure cutoff frequencies are valid (0 to 1, where 1 is Nyquist)
    if normalized_highcut >= 1.0: 
        normalized_highcut = 0.99
    if normalized_lowcut >= normalized_highcut: 
        normalized_lowcut = normalized_highcut * 0.1 

    b_band, a_band = butter(filter_order, [normalized_lowcut, normalized_highcut], btype='band')
    
    # Calculate Notch filter coefficients
    normalized_notch_freq = notch_freq / nyquist
    if normalized_notch_freq >= 1.0: # Ensure notch_freq is below Nyquist
        normalized_notch_freq = 0.99
    b_notch, a_notch = iirnotch(normalized_notch_freq, notch_Q)

    for ch_idx in range(num_channels):
        channel_signal = emg_data_reshaped[:, ch_idx]

        # 1. Bandpass Filtering
        # filtfilt needs signal length > filter order to work correctly
        if len(channel_signal) > filter_order: 
            filtered_signal = filtfilt(b_band, a_band, channel_signal)
        else:
            filtered_signal = channel_signal.copy()

        # 2. Notch Filtering
        if len(filtered_signal) > filter_order: 
            filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)
        else:
            pass # If too short, skip Notch filter

        # 3. DWT Denoising (reducing noise with Wavelet Transform)
        # Check if signal has sufficient data and is not nearly zero before DWT
        if len(filtered_signal) > 0 and np.sum(np.abs(filtered_signal)) > 1e-6:
            try:
                # Perform Discrete Wavelet Transform
                # Check minimum length for DWT level
                min_len_for_dwt = pywt.dwt_coeff_len(len(filtered_signal), dwt_wavelet, mode='symmetric')
                if len(filtered_signal) < min_len_for_dwt or dwt_level >= pywt.dwt_max_level(len(filtered_signal), dwt_wavelet):
                    # If signal is too short for chosen DWT level, use original filtered signal
                    denoised_signal = filtered_signal
                else:
                    coeffs = pywt.wavedec(filtered_signal, dwt_wavelet, level=dwt_level)
                    
                    # Estimate Noise Standard Deviation (Universal threshold)
                    # Add epsilon to prevent division by zero
                    sigma = np.median(np.abs(coeffs[-1])) / (0.6745 + epsilon)
                    threshold = sigma * np.sqrt(2 * np.log(len(filtered_signal)))

                    # Denoising using Soft Thresholding on Detail Coefficients
                    denoised_coeffs = [coeffs[0]] # Approximation coefficient is kept as is
                    for i in range(1, len(coeffs)):
                        denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
                    
                    # Reconstruct signal from denoised coefficients
                    denoised_signal = pywt.waverec(denoised_coeffs, dwt_wavelet)

                    # Ensure reconstructed signal length matches original segment length
                    if len(denoised_signal) != len(channel_signal):
                        denoised_signal = denoised_signal[:len(channel_signal)]
            except ValueError as ve:
                # print(f"Warning: DWT ValueError for channel {ch_idx}: {ve}. Using filtered signal.")
                denoised_signal = filtered_signal
            except Exception as e:
                # print(f"Warning: DWT Denoising failed for channel {ch_idx}: {e}. Using filtered signal.")
                denoised_signal = filtered_signal
        else:
            denoised_signal = filtered_signal # If signal is empty or zero, skip DWT

        processed_emg_data[:, ch_idx] = denoised_signal

    return processed_emg_data.flatten()

start_time_all = time.time()

processed_files = [
    f for f in os.listdir(folder_path) if f.startswith("processed_") and f.endswith(".csv")
]

if not processed_files:
    print(f"⚠️ ไม่พบไฟล์ที่ขึ้นต้นด้วย 'processed_' ในโฟลเดอร์: {folder_path}")
else:
    for filename in processed_files:
        file_path = os.path.join(folder_path, filename)
        patient_id = filename.split("_")[-1].split(".")[0]
        output_cleaned_file = os.path.join(
            output_folder, f"cleaned_single_grasp_{patient_id}.csv"
        )

        print(f"\n--- กำลังประมวลผลไฟล์: {os.path.basename(file_path)} ---")
        start_time_file = time.time()
        total_processing_time_file = 0
        total_emg_points_file = 0 # To track total EMG points processed in this file
        total_windows_file = 0 # To track total windows generated and saved for this file
        multi_grasp_windows_count = 0
        error_count_file = 0
        files_without_repetition_id_flag = False
        problematic_files_flag = False

        first_write_to_output_file = True

        try:
            df_chunks = pd.read_csv(
                file_path, iterator=True, chunksize=chunksize, on_bad_lines="warn"
            )
            
            # Deques should have maxlen as the required samples in a window
            current_emg_data = deque(maxlen=window_size_samples * num_emg_channels)
            current_grasp_ids = deque(maxlen=window_size_samples)
            current_repetition_ids = deque(maxlen=window_size_samples)

            for chunk in tqdm(
                df_chunks, desc=f"อ่านและประมวลผล {os.path.basename(file_path)}"
            ):
                # Filter out 'Grasp ID' that is 0
                chunk_filtered = chunk[chunk["Grasp ID"] != 0].copy()

                if (
                    "EMG Signal" in chunk_filtered.columns
                    and "Grasp ID" in chunk_filtered.columns
                    and "Grasp Repetition" in chunk_filtered.columns
                ):
                    emg_signal_strings = chunk_filtered["EMG Signal"].dropna().values
                    grasp_ids_chunk = chunk_filtered["Grasp ID"].dropna().values
                    repetition_ids_chunk = (
                        chunk_filtered["Grasp Repetition"].dropna().values
                    )

                    for (
                        emg_str,
                        grasp_id,
                        repetition_id,
                    ) in zip(
                        emg_signal_strings, grasp_ids_chunk, repetition_ids_chunk
                    ):
                        emg_array = safe_parse_emg(emg_str)
                        
                        # --- Call Signal Preprocessing function here ---
                        if emg_array.size > 0:
                            emg_array = apply_preprocessing(
                                emg_array,
                                sampling_rate,
                                num_emg_channels,
                                lowcut_freq,
                                highcut_freq,
                                filter_order,
                                notch_freq,
                                notch_Q,
                                dwt_wavelet,
                                dwt_level,
                                epsilon 
                            )
                        # --- End Signal Preprocessing call ---

                        parsed_grasp_ids = safe_parse_grasp_ids(grasp_id)
                        parsed_repetition_id = safe_parse_grasp_repetition(
                            repetition_id
                        )

                        if (
                            emg_array.size > 0
                            and parsed_grasp_ids
                            and parsed_repetition_id is not None
                        ):
                            if emg_array.size % num_emg_channels != 0:
                                error_count_file += 1
                                continue

                            emg_list = emg_array.tolist()
                            total_emg_points_file += emg_array.size
                            current_emg_data.extend(emg_list)
                            
                            num_time_points_in_emg_list = len(emg_list) // num_emg_channels

                            if len(parsed_grasp_ids) == 1:
                                current_grasp_ids.extend([parsed_grasp_ids[0]] * num_time_points_in_emg_list)
                                current_repetition_ids.extend(
                                    [parsed_repetition_id] * num_time_points_in_emg_list
                                )
                            elif len(parsed_grasp_ids) == num_time_points_in_emg_list:
                                current_grasp_ids.extend(parsed_grasp_ids)
                                current_repetition_ids.extend(
                                    [parsed_repetition_id] * num_time_points_in_emg_list
                                )
                            else:
                                error_count_file += 1
                                continue

                        else:
                            error_count_file += 1
                            if parsed_repetition_id is None:
                                files_without_repetition_id_flag = True

                    # --- Non-overlapping Window Processing and Direct Writing ---
                    # Advance deque by window_size_samples because overlap_ms is 0
                    while len(current_emg_data) >= window_size_samples * num_emg_channels:
                        window_emg_flat = list(current_emg_data)[:window_size_samples * num_emg_channels]
                        window_emg_reshaped = np.array(window_emg_flat, dtype=np.float32).reshape(window_size_samples, num_emg_channels)
                        
                        window_grasp = list(current_grasp_ids)[:window_size_samples]
                        window_repetition = list(current_repetition_ids)[:window_size_samples]
                        
                        total_windows_file += 1

                        unique_grasps = set(window_grasp)
                        unique_repetition_ids = set(window_repetition)

                        if (
                            len(unique_grasps) == 1
                            and len(unique_repetition_ids) == 1
                            and list(unique_repetition_ids)[0] is not None
                        ):
                            window_data_dict = {}
                            for ch_idx in range(num_emg_channels):
                                for s_idx in range(window_size_samples):
                                    window_data_dict[f'EMG_Ch{ch_idx+1}_Smp{s_idx+1}'] = window_emg_reshaped[s_idx, ch_idx]
                            
                            window_data_dict['grasp_id'] = list(unique_grasps)[0]
                            window_data_dict['Grasp Repetition'] = list(unique_repetition_ids)[0]
                            
                            df_single_window = pd.DataFrame([window_data_dict])
                            
                            if first_write_to_output_file:
                                df_single_window.to_csv(output_cleaned_file, mode="w", header=True, index=False)
                                first_write_to_output_file = False
                            else:
                                df_single_window.to_csv(output_cleaned_file, mode="a", header=False, index=False)
                            
                            del df_single_window
                            gc.collect()

                        else:
                            multi_grasp_windows_count += 1
                            if (len(unique_grasps) == 1 and
                                (len(unique_repetition_ids) > 1 or list(unique_repetition_ids)[0] is None)):
                                files_without_repetition_id_flag = True

                        # Advance the deques by window_size_samples for non-overlapping windows
                        current_emg_data = deque(
                            list(current_emg_data)[window_size_samples * num_emg_channels:], # Use window_size_samples directly
                            maxlen=window_size_samples * num_emg_channels,
                        )
                        current_grasp_ids = deque(
                            list(current_grasp_ids)[window_size_samples:], # Use window_size_samples directly
                            maxlen=window_size_samples,
                        )
                        current_repetition_ids = deque(
                            list(current_repetition_ids)[window_size_samples:], # Use window_size_samples directly
                            maxlen=window_size_samples,
                        )

                else:
                    print(
                        f"⚠️ Warning: ไม่พบคอลัมน์ 'EMG Signal' หรือ 'Grasp ID' หรือ 'Grasp Repetition' ในไฟล์ {os.path.basename(file_path)}. ข้ามการประมวลผล EMG."
                    )
                    error_count_file += 1
                    problematic_files_flag = True

                del chunk
                del chunk_filtered
                gc.collect()

        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
            problematic_files_flag = True
            continue

        end_time_file = time.time()
        total_processing_time_file = end_time_file - start_time_file
        minutes_file = int(total_processing_time_file // 60)
        seconds_file = int(total_processing_time_file % 60)

        window_duration_seconds = window_size_samples / sampling_rate
        total_duration_seconds = total_windows_file * window_duration_seconds
        total_duration_minutes = int(total_duration_seconds // 60)
        total_duration_remaining_seconds = int(total_duration_seconds % 60)
        total_emg_duration_seconds = total_emg_points_file / sampling_rate
        total_emg_duration_minutes = int(total_emg_duration_seconds // 60)
        total_emg_duration_remaining_seconds = int(total_emg_duration_seconds % 60)

        print(f"\n--- สรุปการประมวลผลไฟล์: {os.path.basename(file_path)} ---")
        print(f"ใช้เวลา: {minutes_file} นาที {seconds_file} วินาที")
        print(
            f"จำนวนข้อมูล EMG ทั้งหมดที่ประมวลผล: {total_emg_points_file} จุด ({total_emg_duration_minutes} นาที {total_emg_duration_remaining_seconds} วินาที)"
        )
        print(f"จำนวน Windows ที่สร้างและบันทึกได้ทั้งหมด: {total_windows_file} windows")
        print(f"จำนวน Windows ที่ถูกกรองออก (Grasp ID มากกว่า 1 หรือ Repetition ID ไม่สอดคล้องกัน/ไม่มี): {multi_grasp_windows_count} windows")
        print(f"จำนวน Error ในการประมวลผล (ภายในไฟล์): {error_count_file}")
        if files_without_repetition_id_flag:
            print(f"⚠️ พบ Windows ที่ไม่มี Grasp Repetition ID ในไฟล์นี้")
        if problematic_files_flag:
            print(f"⚠️ พบข้อผิดพลาดทั่วไปในการประมวลผลไฟล์นี้")
        print(f"ขนาด Window: {window_size_samples} ช่องสัญญาณ ({window_duration_seconds:.3f} วินาที)")
        print(f"จำนวนช่องสัญญาณ EMG: {num_emg_channels}")
        # Updated overlap description for clarity
        print(f"Overlap: 0% (เป็น Non-overlapping Window)") 
        print(
            f"ระยะเวลาทั้งหมดของ window ทั้งหมด: {total_duration_minutes} นาที {total_duration_remaining_seconds} วินาที"
        )

        # Clear deques and explicitly call garbage collection after each file is processed
        current_emg_data.clear()
        current_grasp_ids.clear()
        current_repetition_ids.clear()
        gc.collect()

    end_time_all = time.time()
    total_processing_time_all = end_time_all - start_time_all
    minutes_all = int(total_processing_time_all // 60)
    seconds_all = int(total_processing_time_all % 60)

    print(f"\n--- สิ้นสุดการประมวลผลไฟล์ทั้งหมด ---")
    print(f"ใช้เวลาประมวลผลทั้งหมด: {minutes_all} นาที {seconds_all} วินาที")

    print("การประมวลผลเสร็จสิ้น (อาจมี Warning บางส่วนถูกปิดเพื่อลด Output)")

    gc.collect()
aa
