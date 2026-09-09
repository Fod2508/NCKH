"""
EEG Explorer — UI khám phá bộ dữ liệu CHB-MIT
Chạy: python -m streamlit run eeg_explorer.py
"""

import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import mne
mne.set_log_level("WARNING")

# ── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Explorer — CHB-MIT",
    page_icon="🧠",
    layout="wide",
)

PHYSIONET_URL   = "https://physionet.org/files/chbmit/1.0.0"
LOCAL_ROOT      = r"D:\NCKH\data\chbmit"
SFREQ           = 256

# Folder ID của Google Drive (set trong Streamlit secrets hoặc hardcode)
# Cấu trúc Drive: chbmit/ -> chb01/ -> chb01_01.edf, chb01-summary.txt, ...
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")

# Chế độ: "local" / "gdrive" / "physionet"
if os.path.exists(LOCAL_ROOT):
    DATA_MODE = "local"
    DATA_ROOT = LOCAL_ROOT
elif GDRIVE_FOLDER_ID:
    DATA_MODE = "gdrive"
    DATA_ROOT = None
else:
    DATA_MODE = "physionet"
    DATA_ROOT = None

# ── Thông tin cơn động kinh (từ summary.txt) ────────────────────────────────
SEIZURE_INFO = {
    "chb01": {
        "chb01_03.edf": [(2996, 3036)],
        "chb01_04.edf": [(1467, 1494)],
        "chb01_15.edf": [(1732, 1772)],
        "chb01_16.edf": [(1015, 1066)],
        "chb01_18.edf": [(1720, 1810)],
        "chb01_21.edf": [(327,  420)],
        "chb01_26.edf": [(1862, 1963)],
    }
}

# ════════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Đang đọc file EDF...")
def load_edf(subject, filename):
    """
    Load EDF theo thứ tự ưu tiên:
    1. Local  (máy đã tải về)
    2. Google Drive  (Streamlit Cloud)
    3. PhysioNet  (fallback)
    """
    import tempfile

    # -- LOCAL --
    if DATA_MODE == "local":
        filepath = os.path.join(DATA_ROOT, subject, filename)

    # -- GOOGLE DRIVE --
    elif DATA_MODE == "gdrive":
        import gdown, json

        # Lấy file ID từ cache đã index trước
        index = _get_gdrive_index()
        file_id = index.get(subject, {}).get(filename)

        if not file_id:
            st.error(f"Không tìm thấy {subject}/{filename} trên Drive. Kiểm tra lại folder ID.")
            st.stop()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        tmp.close()
        gdown.download(id=file_id, output=tmp.name, quiet=True)
        filepath = tmp.name

    # -- PHYSIONET fallback --
    else:
        import requests as req
        url = f"{PHYSIONET_URL}/{subject}/{filename}"
        with st.spinner(f"Tải từ PhysioNet: {filename}..."):
            r = req.get(url, timeout=120)
            r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        tmp.write(r.content)
        tmp.flush()
        filepath = tmp.name

    raw  = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    data = raw.get_data() * 1e6
    return data, raw.ch_names, int(raw.info["sfreq"]), raw.times[-1]


@st.cache_data(ttl=3600, show_spinner="Đang lấy danh sách file từ Drive...")
def _get_gdrive_index():
    """
    Trả về dict: {subject: {filename: file_id}}
    Đọc từ file gdrive_index.json (tạo bởi build_gdrive_index.py)
    """
    index_path = "gdrive_index.json"
    if os.path.exists(index_path):
        import json
        with open(index_path) as f:
            return json.load(f)
    return {}


def band_power(signal_1ch, sfreq, lo, hi):
    freqs = np.fft.rfftfreq(len(signal_1ch), d=1/sfreq)
    fft   = np.abs(np.fft.rfft(signal_1ch)) ** 2
    idx   = np.where((freqs >= lo) & (freqs < hi))[0]
    return float(np.sum(fft[idx]))


def compute_features(segment):
    """segment: (n_channels, n_samples) µV"""
    var    = float(np.var(segment))
    amp    = float(np.mean(np.abs(segment)))
    p2p    = float(segment.max() - segment.min())
    ll     = float(np.mean(np.sum(np.abs(np.diff(segment, axis=1)), axis=1)))
    zcr    = float(np.mean(
        np.sum(np.diff(np.sign(segment), axis=1) != 0, axis=1) / segment.shape[1]
    ))

    avg_ch = np.mean(segment, axis=0)
    d  = band_power(avg_ch, SFREQ, 0.5,  4)
    th = band_power(avg_ch, SFREQ, 4,    8)
    al = band_power(avg_ch, SFREQ, 8,   13)
    be = band_power(avg_ch, SFREQ, 13,  30)
    ga = band_power(avg_ch, SFREQ, 30,  40)
    total = d + th + al + be + ga + 1e-10

    return {
        "Variance (µV²)":       var,
        "Mean Amplitude (µV)":  amp,
        "Peak-to-Peak (µV)":    p2p,
        "Line Length":          ll,
        "Zero Crossing Rate":   zcr,
        "Delta % (0.5–4Hz)":    d  / total * 100,
        "Theta % (4–8Hz)":      th / total * 100,
        "Alpha % (8–13Hz)":     al / total * 100,
        "Beta % (13–30Hz)":     be / total * 100,
        "Gamma % (30–40Hz)":    ga / total * 100,
    }


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — chọn bệnh nhân & file
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.title("EEG Explorer")

# Danh sách bệnh nhân
if DATA_ROOT and os.path.exists(DATA_ROOT):
    # Local: lấy từ thư mục đã tải
    available = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("chb")
    ])
else:
    # Cloud: dùng danh sách cố định
    available = [f"chb{i:02d}" for i in range(1, 25)]

if not available:
    st.error("Chưa tìm thấy dữ liệu.")
    st.stop()

subject = st.sidebar.selectbox("Chọn bệnh nhân", available)

# Danh sách file EDF
if DATA_ROOT and os.path.exists(os.path.join(DATA_ROOT, subject)):
    edf_files = sorted([f for f in os.listdir(os.path.join(DATA_ROOT, subject)) if f.endswith(".edf")])
else:
    # Cloud: lấy từ summary info cứng
    edf_files = [f"{subject}_{i:02d}.edf" for i in range(1, 47)]

edf_file  = st.sidebar.selectbox("Chọn file EDF", edf_files)

# Thông tin cơn của file được chọn
seizures = SEIZURE_INFO.get(subject, {}).get(edf_file, [])
has_seizure = len(seizures) > 0

if has_seizure:
    st.sidebar.success(f"⚡ File này có {len(seizures)} cơn động kinh")
    for i, (s, e) in enumerate(seizures, 1):
        st.sidebar.info(f"Cơn {i}: giây {s} → {e} ({e-s}s)")
else:
    st.sidebar.info("✅ File này không có cơn động kinh")

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
data, ch_names, sfreq, duration = load_edf(subject, edf_file)
n_ch, n_samples = data.shape

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "Thuộc tính file",
    "Xem sóng EEG",
    "So sánh giai đoạn",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1: Thuộc tính file
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Thuộc tính file")

    st.info("""
    **Bước này làm gì?**
    Đọc phần header của file EDF để xem các thông số kỹ thuật cơ bản:
    tần số lấy mẫu, số kênh điện cực, thời lượng, và dữ liệu thực sự trông như thế nào.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tần số lấy mẫu", f"{sfreq} Hz",
                help="Mỗi giây ghi 256 điểm dữ liệu cho mỗi kênh")
    col2.metric("Số kênh", f"{n_ch}",
                help="23 điện cực đặt trên da đầu theo chuẩn 10-20")
    col3.metric("Thời lượng", f"{duration/60:.0f} phút",
                help=f"{duration:.0f} giây")
    col4.metric("Tổng số mẫu", f"{n_samples:,}",
                help=f"{sfreq} Hz × {duration:.0f}s = {n_samples:,}")

    st.subheader("Danh sách kênh điện cực")
    st.caption("Tên kênh = cặp điện cực đo hiệu điện thế giữa 2 điểm trên da đầu (ví dụ FP1-F7 = trán trái)")
    cols = st.columns(5)
    for i, ch in enumerate(ch_names):
        cols[i % 5].write(f"`{i+1:2d}. {ch}`")

    st.subheader("Dữ liệu thô — 5 mẫu đầu của kênh FP1-F7")
    st.caption("Mỗi số = điện thế đo được tại 1 thời điểm (đơn vị µV, mỗi mẫu cách nhau 1/256 ≈ 3.9ms)")
    sample_vals = data[0, :5]
    for i, v in enumerate(sample_vals):
        st.write(f"  Mẫu {i} (t = {i/sfreq*1000:.1f}ms): **{v:.3f} µV**")

    st.subheader("Thống kê toàn file")
    import pandas as pd
    stats_data = []
    for i, ch in enumerate(ch_names):
        stats_data.append({
            "Kênh": ch,
            "Min (µV)": round(float(data[i].min()), 1),
            "Max (µV)": round(float(data[i].max()), 1),
            "Mean (µV)": round(float(data[i].mean()), 3),
            "Std (µV)": round(float(data[i].std()), 1),
        })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2: Xem sóng EEG
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Xem sóng EEG")

    st.info("""
    **Bước này làm gì?**
    Vẽ đồ thị tín hiệu EEG theo thời gian. Bạn chọn kênh và khoảng thời gian muốn xem.
    Mục tiêu: thấy sóng EEG trông như thế nào và nhận biết được khi nào có cơn bằng mắt.
    """)

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        selected_ch = st.multiselect(
            "Chọn kênh xem",
            ch_names,
            default=ch_names[:3],
            max_selections=6,
        )
    with col_b:
        t_start = st.slider("Thời điểm bắt đầu (giây)", 0, int(duration) - 10, 0)
        t_len   = st.slider("Độ dài đoạn xem (giây)", 5, 60, 10)
    with col_c:
        amp_scale = st.slider("Biên độ hiển thị (µV)", 50, 500, 150)

    if selected_ch:
        s_idx = t_start * sfreq
        e_idx = min(s_idx + t_len * sfreq, n_samples)
        t_arr = np.arange(e_idx - s_idx) / sfreq + t_start

        fig, axes = plt.subplots(len(selected_ch), 1,
                                 figsize=(14, 2.2 * len(selected_ch)),
                                 sharex=True)
        if len(selected_ch) == 1:
            axes = [axes]

        for ax, ch in zip(axes, selected_ch):
            ch_idx = ch_names.index(ch)
            seg    = data[ch_idx, s_idx:e_idx]
            ax.plot(t_arr, seg, lw=0.6, color="steelblue")
            ax.set_ylabel(ch, fontsize=8)
            ax.set_ylim(-amp_scale, amp_scale)
            ax.axhline(0, color="gray", lw=0.3, ls="--")

            # Đánh dấu vùng cơn nếu có
            for sz_s, sz_e in seizures:
                if sz_s <= t_start + t_len and sz_e >= t_start:
                    ax.axvspan(max(sz_s, t_start), min(sz_e, t_start + t_len),
                               alpha=0.25, color="red", label="Cơn động kinh")

        # Chú thích cơn
        if seizures:
            for sz_s, sz_e in seizures:
                if sz_s <= t_start + t_len and sz_e >= t_start:
                    axes[0].legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Thời gian (giây)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if has_seizure:
            st.caption("🔴 Vùng đỏ = cơn động kinh theo chú thích trong summary.txt")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3: So sánh giai đoạn
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("So sánh giai đoạn")

    st.info("""
    **Bước này làm gì?**
    Cắt tín hiệu EEG thành 3 đoạn rồi đặt cạnh nhau để thấy sự khác biệt rõ ràng.
    Đây là bước để hiểu **tại sao** các đặc trưng như Variance hay Amplitude lại hữu ích.
    """)

    if not has_seizure:
        st.warning("File này không có cơn. Hãy chọn file có cơn ở thanh bên (ví dụ chb01_03.edf).")
    else:
        sz_s, sz_e = seizures[0]
        pre_len  = st.slider("Độ dài đoạn trước/sau cơn (giây)", 30, 300, 120)
        compare_ch = st.selectbox("Kênh xem", ch_names, index=0)
        ch_idx = ch_names.index(compare_ch)

        # Cắt 3 đoạn
        pre_s  = max(0, sz_s - pre_len)
        post_e = min(n_samples // sfreq, sz_e + pre_len)

        seg_pre   = data[ch_idx, pre_s*sfreq  : sz_s*sfreq]
        seg_ictal = data[ch_idx, sz_s*sfreq   : sz_e*sfreq]
        seg_post  = data[ch_idx, sz_e*sfreq   : post_e*sfreq]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        fig.suptitle(f"So sánh 3 giai đoạn — kênh {compare_ch}", fontsize=13)

        for ax, (sig, title, color, offset) in zip(axes, [
            (seg_pre,   f"Trước cơn ({pre_s}s → {sz_s}s)", "steelblue", pre_s),
            (seg_ictal, f"TRONG CƠN ({sz_s}s → {sz_e}s) ← cơn thật",   "red",       sz_s),
            (seg_post,  f"Sau cơn   ({sz_e}s → {post_e}s)", "green",      sz_e),
        ]):
            t = np.arange(len(sig)) / sfreq + offset
            ax.plot(t, sig, lw=0.5, color=color)
            ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
            ax.set_ylabel("µV")
            ax.axhline(0, color="gray", lw=0.3, ls="--")
            ax.annotate(f"std={sig.std():.1f}µV  max|x|={np.abs(sig).max():.1f}µV",
                        xy=(0.98, 0.92), xycoords="axes fraction",
                        ha="right", fontsize=8, color=color)

        axes[-1].set_xlabel("Thời gian (giây trong file)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Phổ tần số so sánh
        st.subheader("Phổ tần số (FFT) — 3 giai đoạn")
        st.caption("Trục X = tần số (Hz), trục Y = năng lượng. Vùng đỏ = trong cơn thường có năng lượng phân bố khác hẳn.")

        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        for ax, (sig, title, color) in zip(axes2, [
            (seg_pre,   "Trước cơn",  "steelblue"),
            (seg_ictal, "Trong cơn",  "red"),
            (seg_post,  "Sau cơn",    "green"),
        ]):
            freqs_f = np.fft.rfftfreq(len(sig), d=1/sfreq)
            fft_mag = np.abs(np.fft.rfft(sig))
            mask    = freqs_f <= 50
            ax.fill_between(freqs_f[mask], fft_mag[mask], alpha=0.4, color=color)
            ax.plot(freqs_f[mask], fft_mag[mask], lw=0.8, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Tần số (Hz)")
            ax.set_ylabel("Biên độ FFT")
            for f in [4, 8, 13, 30]:
                ax.axvline(f, color="gray", ls="--", lw=0.6)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ────────────────────────────────────────────────────────────────────────────
# TAB 4: Đặc trưng & Decision Tree
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🌳 Bước 4 — Tính đặc trưng & hiểu Decision Tree")

    st.info("""
    **Bước này làm gì?**
    Tính các số đặc trưng (Variance, Amplitude, Band Power...) cho từng giai đoạn.
    Sau đó giải thích Decision Tree dùng các số này để phân loại như thế nào.
    """)

    if not has_seizure:
        st.warning("Chọn file có cơn để so sánh đầy đủ (ví dụ chb01_03.edf).")
    else:
        sz_s, sz_e = seizures[0]

        # Tính đặc trưng 4 đoạn
        win = 60   # giây
        seg_normal_all = data[:, max(0, (sz_s-600))*sfreq : max(0, (sz_s-540))*sfreq]
        seg_pre_all    = data[:, (sz_s-300)*sfreq : sz_s*sfreq]
        seg_ictal_all  = data[:, sz_s*sfreq : sz_e*sfreq]
        seg_post_all   = data[:, sz_e*sfreq : (sz_e+300)*sfreq]

        import pandas as pd

        rows = {}
        for name, seg in [
            ("Bình thường", seg_normal_all),
            ("Trước cơn",   seg_pre_all),
            ("Trong cơn",   seg_ictal_all),
            ("Sau cơn",     seg_post_all),
        ]:
            if seg.shape[1] > 0:
                rows[name] = compute_features(seg)

        df = pd.DataFrame(rows).T.round(2)

        st.subheader("Bảng đặc trưng — 4 giai đoạn")
        st.caption("Mỗi hàng = 1 giai đoạn. Số nào thay đổi nhiều nhất giữa 'Bình thường' và 'Trong cơn' là đặc trưng tốt nhất.")

        # Highlight giá trị lớn nhất mỗi cột
        def highlight_max(s):
            is_max = s == s.max()
            return ["background-color: #ffcccc" if v else "" for v in is_max]

        st.dataframe(df.style.apply(highlight_max), use_container_width=True)
        st.caption("🔴 Ô đỏ = giá trị lớn nhất trong cột đó")

        # ── Decision Tree minh họa ────────────────────────────────────────
        st.subheader("Decision Tree đơn giản — minh họa")
        st.caption("Chọn 2 đặc trưng để xem ngưỡng phân loại Decision Tree sẽ dùng.")

        feat_names = list(df.columns)
        col_x, col_y = st.columns(2)
        feat1 = col_x.selectbox("Đặc trưng 1 (nhánh gốc)", feat_names, index=0)
        feat2 = col_y.selectbox("Đặc trưng 2 (nhánh con)", feat_names, index=4)

        v_normal = df.loc["Bình thường", feat1] if "Bình thường" in df.index else 0
        v_pre    = df.loc["Trước cơn",   feat1] if "Trước cơn"   in df.index else 0
        v_ictal  = df.loc["Trong cơn",   feat1] if "Trong cơn"   in df.index else 0
        threshold1 = (v_normal + v_pre) / 2

        v2_normal = df.loc["Bình thường", feat2] if "Bình thường" in df.index else 0
        v2_pre    = df.loc["Trước cơn",   feat2] if "Trước cơn"   in df.index else 0
        threshold2 = (v2_normal + v2_pre) / 2

        st.markdown(f"""
```
                    [BẮT ĐẦU — cửa sổ EEG 10 giây]
                               |
               {feat1} > {threshold1:.1f} ?
              /                         \\
           CÓ                          KHÔNG
           |                              |
  {feat2} > {threshold2:.1f} ?       → BÌNH THƯỜNG
       /          \\
     CÓ           KHÔNG
     |               |
 TRƯỚC CƠN      BÌNH THƯỜNG

Giá trị thực tế:
  Bình thường : {feat1} = {v_normal:.1f}  |  {feat2} = {v2_normal:.1f}
  Trước cơn   : {feat1} = {v_pre:.1f}  |  {feat2} = {v2_pre:.1f}
  Trong cơn   : {feat1} = {v_ictal:.1f}
```
        """)

        st.info(f"""
        **Đọc cây như thế nào?**
        - Mỗi node hỏi 1 câu hỏi dạng "đặc trưng X > ngưỡng Y không?"
        - Đi theo nhánh CÓ hoặc KHÔNG tùy giá trị thực tế
        - Đến lá → ra kết quả phân loại
        - Ngưỡng {threshold1:.1f} được tính tự động = trung bình giữa bình thường ({v_normal:.1f}) và trước cơn ({v_pre:.1f})
        - **Random Forest** = 500 cây như trên, mỗi cây dùng tập con dữ liệu khác nhau → bỏ phiếu đa số
        """)
