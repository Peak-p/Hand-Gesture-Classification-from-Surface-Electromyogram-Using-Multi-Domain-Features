import numpy as np
import pandas as pd
import os
import gc
import re
from collections import deque
from tqdm import tqdm
import time
import ast

# --- ค่ากำหนด (ปรับตาม Sliding Window) ---
sampling_rate = 2000  # Hz
window_size_samples = 400  # ขนาด Window 400 samples
overlap_percent = 0.0
step_size_samples = window_size_samples  # Step เท่ากับขนาด Window
chunksize = 1000
folder_path = r'C:\Users\พีค\OneDrive\เดสก์ท็อป\Data sci (NSTDA)\DataPrep'  # โฟลเดอร์ที่เก็บไฟล์ processed_*.csv
output_folder = r'C:\Users\พีค\OneDrive\เดสก์ท็อป\Data sci (NSTDA)\train&test_window_size_samples _400'  # โฟลเดอร์สำหรับเก็บไฟล์ cleaned ที่มี grasp เดี่ยว

os.makedirs(output_folder, exist_ok=True)

def safe_parse_emg(emg_string):
    """
    แปลง EMG signal ที่เป็น String ให้เป็น NumPy array พร้อมจัดการ Error ที่อาจเกิดขึ้น

    Args:
        emg_string (str): String ที่แทน EMG signal

    Returns:
        numpy.ndarray: EMG signal ในรูปแบบ NumPy array หรือ Array ว่างเปล่าหากมี Error
    """
    emg_string = emg_string.strip()  # ลบช่องว่างหน้าหลัง
    if not emg_string:  # ตรวจสอบว่า String ว่างหรือไม่
        return np.array([])
    if emg_string.startswith("[") and emg_string.endswith("]"):
        emg_string = emg_string[1:-1]
    try:
  
        emg_list = ast.literal_eval(f"[{emg_string}]")
        return np.array(emg_list, dtype=np.float32)  # ระบุ dtype
    except (ValueError, SyntaxError) as e:
        try:
        
            cleaned_string = re.sub(r'\s+', ',', emg_string).strip(',')
            return np.fromstring(cleaned_string, sep=',', dtype=np.float32)  # ระบุ dtype
        except Exception as e:
            print(f"⚠️ Error parsing EMG: {e} | Data: '{emg_string[:100]}...'")
            return np.array([])
    except Exception as e:
        print(f"⚠️ Error parsing EMG (initial): {e} | Data: '{emg_string[:100]}...'")
        return np.array([])



def safe_parse_grasp_ids(grasp_id):
    """
    แปลง Grasp ID ที่เป็น String หรือ Value ให้เป็น List ของ Integer IDs

    Args:
        grasp_id (str, int, numpy.int64): Grasp ID(s) ที่ต้องการแปลง

    Returns:
        list: List ของ Integer Grasp IDs หรือ List ว่างเปล่าหากมี Error
    """
    if isinstance(grasp_id, str):
        grasp_id = grasp_id.strip()
        if not grasp_id:
            return []
        if grasp_id.startswith("[") and grasp_id.endswith("]"):
            try:
                
                return [int(x) for x in ast.literal_eval(grasp_id)]
            except (ValueError, SyntaxError) as e:
                print(f"⚠️ Error parsing Grasp IDs (string): {e} | Data: '{grasp_id[:100]}...'")
                return []
        else:
            try:
                return [int(grasp_id)]
            except ValueError:
                print(f"⚠️ Error parsing single Grasp ID (string): '{grasp_id}'")
                return []
    elif isinstance(grasp_id, (int, np.int64)):
        return [int(grasp_id)]
    elif isinstance(grasp_id, list):
        try:
            return [int(x) for x in grasp_id]
        except ValueError:
            print(f"⚠️ Error parsing Grasp IDs (list): '{grasp_id}'")
            return []
    else:
        print(f"⚠️ Error: Unknown type for Grasp ID: {type(grasp_id)} | Value: '{grasp_id}'")
        return []



def safe_parse_grasp_repetition(repetition_id):
    """
    แปลง Grasp Repetition ID.  ส่งคืน None ถ้าเป็น 0 หรือมีปัญหา

    Args:
        repetition_id: Grasp Repetition ID ที่ต้องการแปลง

    Returns:
        int: ค่า Repetition ID ที่แปลงแล้ว, None หากมี Error หรือเป็น 0.
    """
    if isinstance(repetition_id, (int, np.int64)):
        if repetition_id == 0:
            return None  # Return None for 0
        return int(repetition_id)
    elif isinstance(repetition_id, str):
        repetition_id = repetition_id.strip()  # remove white spaces
        if not repetition_id:
            return None  # Return None for empty string
        try:
            rep_id = int(repetition_id)
            if rep_id == 0:
                return None # Return None if converted value is 0
            return rep_id
        except ValueError:
            print(f"⚠️ Error parsing Grasp Repetition ID (string): '{repetition_id}'")
            return None
    else:
        print(
            f"⚠️ Error: Unknown type for Grasp Repetition ID: {type(repetition_id)}, Value: '{repetition_id}'"
        )
        return None



start_time_all = time.time()


processed_files = [
    f for f in os.listdir(folder_path) if f.startswith("processed_") and f.endswith(".csv")
]

if not processed_files:
    print(f"⚠️ ไม่พบไฟล์ที่ขึ้นต้นด้วย 'processed_' ในโฟลเดอร์: {folder_path}")
else:
    for filename in processed_files:
        file_path = os.path.join(folder_path, filename)
        patient_id = filename.split("_")[-1].split(".")[
            0
        ]  # ดึง 'S010' จาก 'processed_S010.csv'
        output_cleaned_file = os.path.join(
            output_folder, f"cleaned_single_grasp_{patient_id}.csv"
        )

        print(f"\n--- กำลังประมวลผลไฟล์: {os.path.basename(file_path)} ---")
        start_time_file = time.time()
        total_processing_time_file = 0
        total_emg_points_file = 0
        all_windows_data_single = []
        all_windows_grasp_ids_single = []
        all_windows_repetition_ids_single = (
            []
        )  # To store repetition ids, now will hold None if no rep_id
        total_windows_file = 0
        multi_grasp_windows_count = 0
        error_count_file = 0  # Count errors in a file
        files_without_repetition_id = (
            []
        )  # Keep track of files without repetition ID
        problematic_files = []

        try:
            df_chunks = pd.read_csv(
                file_path, iterator=True, chunksize=chunksize, on_bad_lines="warn"
            )
            current_emg_data = deque(maxlen=window_size_samples * 2)
            current_grasp_ids = deque(maxlen=window_size_samples * 2)
            current_repetition_ids = deque(
                maxlen=window_size_samples * 2
            )  # track repetition ids

            for chunk in tqdm(
                df_chunks, desc=f"อ่านและประมวลผล {os.path.basename(file_path)}"
            ):
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
                    )  # Get repetition ids
                    total_emg_points_file += sum(
                        len(safe_parse_emg(s)) for s in emg_signal_strings
                    )

                    for (
                        emg_str,
                        grasp_id,
                        repetition_id,
                    ) in zip(
                        emg_signal_strings, grasp_ids_chunk, repetition_ids_chunk
                    ):  # add repetition_id
                        emg_array = safe_parse_emg(emg_str)
                        parsed_grasp_ids = safe_parse_grasp_ids(grasp_id)
                        parsed_repetition_id = safe_parse_grasp_repetition(
                            repetition_id
                        )  # Parse repetition id

                        if (
                            emg_array.size > 0
                            and parsed_grasp_ids
                            and parsed_repetition_id is not None
                        ):  # Check for valid repetition id
                            emg_list = emg_array.tolist()
                            current_emg_data.extend(emg_list)
                            # ทำซ้ำ Grasp ID ตามจำนวนจุด EMG ที่สอดคล้องกัน
                            if len(parsed_grasp_ids) == 1:
                                current_grasp_ids.extend([parsed_grasp_ids[0]] * len(emg_list))
                                current_repetition_ids.extend(
                                    [parsed_repetition_id] * len(emg_list)
                                )  # Propagate repetition id
                            elif len(parsed_grasp_ids) == len(emg_list):
                                current_grasp_ids.extend(parsed_grasp_ids)
                                current_repetition_ids.extend(
                                    [parsed_repetition_id] * len(emg_list)
                                )  # Propagate repetition ids
                            else:
                                print(
                                    f"⚠️ Warning: ขนาดของ Grasp IDs ไม่ตรงกับ EMG Signal ในแถว ในไฟล์: {os.path.basename(file_path)}"
                                )
                                error_count_file += 1
                                continue  # ข้าม iteration นี้หากขนาดไม่ตรงกัน
                        else:
                            error_count_file += 1
                            if parsed_repetition_id is None:
                                files_without_repetition_id.append(
                                    filename
                                )  # Track file

                    while len(current_emg_data) >= window_size_samples:
                        window_emg = list(current_emg_data)[:window_size_samples]
                        window_grasp = list(current_grasp_ids)[:window_size_samples]
                        window_repetition = list(
                            current_repetition_ids
                        )[:window_size_samples]  # Get window repetition ids
                        total_windows_file += 1

                        # ตรวจสอบว่ามี grasp id มากกว่า 1 ใน window หรือไม่
                        unique_grasps = set(window_grasp)
                        if (
                            len(unique_grasps) == 1
                            and window_repetition[0] is not None
                        ):  # Only process if single grasp and has repetition
                            all_windows_data_single.append(window_emg)
                            all_windows_grasp_ids_single.append(list(unique_grasps)[0])
                            all_windows_repetition_ids_single.append(
                                window_repetition[0]
                            )  # store repetition id
                        else:
                            multi_grasp_windows_count += 1
                            if (
                                len(unique_grasps) == 1
                            ):  # if single grasp, but no repetition id.
                                files_without_repetition_id.append(
                                    filename
                                )  # Track the file
                            # Discard the window.  We only want windows with single grasp and repetition id.

                        current_emg_data = deque(
                            list(current_emg_data)[step_size_samples:],
                            maxlen=window_size_samples * 2,
                        )
                        current_grasp_ids = deque(
                            list(current_grasp_ids)[step_size_samples:],
                            maxlen=window_size_samples * 2,
                        )
                        current_repetition_ids = deque(
                            list(current_repetition_ids)[step_size_samples:],
                            maxlen=window_size_samples * 2,
                        )

                else:
                    print(
                        f"⚠️ Warning: ไม่พบคอลัมน์ 'EMG Signal' หรือ 'Grasp ID' หรือ 'Grasp Repetition' ในไฟล์ {os.path.basename(file_path)}. ข้ามการประมวลผล EMG."
                    )
                    error_count_file += 1

            del chunk
            del chunk_filtered
            gc.collect()

        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
            problematic_files.append(filename)  # Keep track of problem files.
            continue  # Continue to the next file

        end_time_file = time.time()
        total_processing_time_file = end_time_file - start_time_file
        minutes_file = int(total_processing_time_file // 60)
        seconds_file = int(total_processing_time_file % 60)

        # บันทึกข้อมูลที่ถูกแยกให้เหลือ grasp id เดี่ยว และมี repetition id
        if all_windows_data_single:
            print(
                f"กำลังบันทึก Windows ที่มี Grasp ID เดี่ยว และ Grasp Repetition ID ลงไฟล์: {os.path.basename(output_cleaned_file)}"
            )
            df_cleaned_windows_emg_single = pd.DataFrame(all_windows_data_single)
            df_cleaned_windows_emg_single["grasp_id"] = all_windows_grasp_ids_single  # Add grasp_id column
            df_cleaned_windows_emg_single["Grasp Repetition"] = all_windows_repetition_ids_single  # Add grasp_repetition_id column, changed name
            df_cleaned_windows_emg_single.to_csv(
                output_cleaned_file, mode="w", header=True, index=False
            )
        else:
            print(
                f"⚠️ Warning: ไม่พบ Windows ที่มี Grasp ID เดี่ยว และ Grasp Repetition ID ในไฟล์ {os.path.basename(file_path)}. ไม่มีการบันทึกไฟล์."
            )

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
            f"จำนวนข้อมูล EMG ทั้งหมด: {total_emg_points_file} จุด ({total_emg_duration_minutes} นาที {total_emg_duration_remaining_seconds} วินาที)"
        )
        print(f"จำนวน Windows ที่สร้างได้ทั้งหมด: {total_windows_file} windows")
        print(f"จำนวน Windows ที่มี Grasp ID มากกว่า 1: {multi_grasp_windows_count} windows")
        print(f"จำนวน Error ในการประมวลผล: {error_count_file}")
        if files_without_repetition_id:
            print(
                f"จำนวนไฟล์ที่ไม่มี Grasp Repetition ID: {len(files_without_repetition_id)}"
            )
        print(f"ขนาด Window: {window_size_samples} ช่องสัญญาณ ({window_duration_seconds:.3f} วินาที)")
        print(f"Overlap: {overlap_percent * 100:.0f}% (Step size: {step_size_samples} ช่องสัญญาณ)")
        print(
            f"ระยะเวลาทั้งหมดของ window ทั้งหมด: {total_duration_minutes} นาที {total_duration_remaining_seconds} วินาที"
        )

        gc.collect()

    end_time_all = time.time()
    total_processing_time_all = end_time_all - start_time_all
    minutes_all = int(total_processing_time_all // 60)
    seconds_all = int(total_processing_time_all % 60)

    print(f"\n--- สิ้นสุดการประมวลผลไฟล์ทั้งหมด ---")
    print(f"ใช้เวลาประมวลผลทั้งหมด: {minutes_all} นาที {seconds_all} วินาที")

    if problematic_files:
        print("⚠️ ไฟล์ที่มีปัญหาในการประมวลผล:")
        for filename in problematic_files:
            print(filename)
    if files_without_repetition_id:
        print("⚠️ ไฟล์ที่ไม่มี Grasp Repetition ID:")
        for filename in files_without_repetition_id:
            print(filename)

    gc.collect()
