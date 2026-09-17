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

@st.cache_data(show_spinner=False)
def load_edf(subject, filename):
    """
    Load EDF theo thứ tự ưu tiên:
    1. Local  (máy đã tải về)
    2. Google Drive — tải về /tmp rồi đọc
    3. PhysioNet  (fallback)
    """
    import tempfile

    # -- LOCAL --
    if DATA_MODE == "local":
        filepath = os.path.join(DATA_ROOT, subject, filename)

    # -- GOOGLE DRIVE --
    elif DATA_MODE == "gdrive":
        import gdown
        import threading

        index    = _get_gdrive_index()
        file_id  = index.get(subject, {}).get(filename)

        if not file_id:
            st.error(f"Khong tim thay {subject}/{filename} trong gdrive_index.json")
            st.stop()

        tmp_path = f"/tmp/{subject}_{filename}"
        if not os.path.exists(tmp_path):
            bar = st.progress(0, text=f"Đang tải {filename} từ Drive (40MB)...")
            # Tải trong thread riêng, cập nhật progress
            done = {"v": False, "err": None}

            def _dl():
                try:
                    gdown.download(id=file_id, output=tmp_path, quiet=True)
                except Exception as e:
                    done["err"] = str(e)
                done["v"] = True

            t = threading.Thread(target=_dl, daemon=True)
            t.start()
            import time
            elapsed = 0
            while not done["v"]:
                time.sleep(1)
                elapsed += 1
                pct = min(int(elapsed / 45 * 100), 95)
                bar.progress(pct, text=f"Đang tải {filename}... ({elapsed}s)")
            bar.progress(100, text="Tải xong!")
            if done["err"]:
                st.error(f"Lỗi tải file: {done['err']}")
                st.stop()
        filepath = tmp_path

    # -- PHYSIONET fallback --
    else:
        import requests as req
        url = f"{PHYSIONET_URL}/{subject}/{filename}"
        tmp_path = f"/tmp/{subject}_{filename}"
        if not os.path.exists(tmp_path):
            with st.spinner(f"Tải từ PhysioNet: {filename}..."):
                r = req.get(url, timeout=120)
                r.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(r.content)
        filepath = tmp_path

    raw  = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    ch_names = raw.ch_names
    sfreq    = int(raw.info["sfreq"])
    duration = raw.times[-1]
    # Load tối đa 60 phút nhưng đủ để bao phủ cơn nếu có
    if DATA_MODE != "local":
        # Lấy thời điểm cơn xa nhất từ SEIZURE_INFO nếu có
        sz_info = SEIZURE_INFO.get(subject, {}).get(filename, [])
        max_s = min(duration, 3600)
        if sz_info:
            last_sz_end = max(e for _, e in sz_info)
            max_s = min(duration, last_sz_end + 300)  # lấy thêm 5 phút sau cơn
        raw.crop(tmax=max_s).load_data(verbose=False)
        raw.pick(raw.ch_names[:8])
    else:
        raw.load_data(verbose=False)
    data = raw.get_data() * 1e6
    return data, ch_names, sfreq, duration


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
_loading_placeholder = st.empty()
_loading_placeholder.info(f"Đang tải {edf_file} (~40MB từ Drive, lần đầu mất ~45 giây)...")
data, ch_names, sfreq, duration = load_edf(subject, edf_file)
_loading_placeholder.empty()   # xóa thông báo sau khi tải xong
n_ch, n_samples = data.shape

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Thuộc tính file",
    "Xem sóng EEG",
    "So sánh giai đoạn",
    "Decision Tree",
    "Tính đặc trưng thủ công",
    "Mô hình",
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
    for i in range(n_ch):
        stats_data.append({
            "Kênh": ch_names[i],
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

    # Gợi ý ngữ cảnh cho file đang xem
    if has_seizure:
        sz_hints = " | ".join([f"Cơn {i+1}: giây {s}–{e}" for i, (s,e) in enumerate(seizures)])
        st.success(f"**File này có cơn động kinh** — {sz_hints}. "
                   f"Kéo slider 'Thời điểm bắt đầu' đến gần giây đó để thấy sóng thay đổi.")
    else:
        st.info("File này không có cơn — đây là tín hiệu bình thường. "
                "Chọn file có cơn ở sidebar (ví dụ chb01_03.edf) để so sánh.")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        available_ch = ch_names[:n_ch]
        # Gợi ý kênh tốt nhất để xem cơn
        recommended = [c for c in available_ch if any(x in c for x in ["T7","FP1","F7","T8"])]
        default_ch  = recommended[:3] if recommended else available_ch[:3]
        selected_ch = st.multiselect(
            "Chọn kênh xem",
            available_ch,
            default=default_ch,
            max_selections=6,
            help="Kênh vùng thái dương (T7, F7, FP1) thường cho thấy cơn rõ nhất vì hay là vùng khởi phát."
        )
    with col_b:
        # Gợi ý mốc thời gian tốt để nhảy đến
        if has_seizure:
            sz_s_hint = seizures[0][0]
            default_t = max(0, sz_s_hint - 30)   # nhảy đến 30 giây trước cơn
            hint_txt  = f"💡 Giây {sz_s_hint} = bắt đầu cơn. Kéo đến ~{default_t}s để thấy trước + trong cơn."
        else:
            default_t = 0
            hint_txt  = "Kéo slider để di chuyển dọc theo bản ghi."
        t_start = st.slider("Thời điểm bắt đầu (giây)", 0, int(duration) - 10, default_t,
                            help=hint_txt)
        st.caption(hint_txt)
        t_len = st.slider("Độ dài đoạn xem (giây)", 5, 60, 30,
                          help="60 giây để thấy tổng quan. 10–20 giây để thấy hình dạng sóng chi tiết.")
    with col_c:
        amp_scale = st.slider("Biên độ hiển thị (µV)", 50, 500, 150,
                              help="Tăng lên 300–500 µV nếu đang xem đoạn trong cơn vì sóng to hơn bình thường.")

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
        st.info(f"**Cơn trong file này**: giây **{sz_s}** → **{sz_e}** (kéo dài {sz_e-sz_s} giây). "
                f"Kênh **FP1-F7** hoặc **F7-T7** thường cho thấy sự khác biệt rõ nhất.")
        pre_len  = st.slider("Độ dài đoạn trước/sau cơn (giây)", 30, 300, 120,
                             help="120 giây = 2 phút trước và sau cơn. Tăng lên 300s để thấy não dần thay đổi từ xa.")
        compare_ch = st.selectbox("Kênh xem", ch_names[:n_ch], index=0,
                                  help="Thử đổi kênh để thấy cơn rõ hay mờ khác nhau — phản ánh vùng khởi phát.")
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
            (seg_ictal, f"TRONG CƠN ({sz_s}s → {sz_e}s) ← cơn thật",   "red",   sz_s),
            (seg_post,  f"Sau cơn   ({sz_e}s → {post_e}s)", "green",    sz_e),
        ]):
            if len(sig) == 0:
                ax.set_title(f"{title} — không có dữ liệu", fontsize=9)
                continue
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
        st.caption("Trục X = tần số (Hz), trục Y = năng lượng. Chú ý trục Y chung — trong cơn năng lượng cao hơn hẳn.")

        # Tính trước để dùng chung trục Y
        fft_data = []
        for sig, title, color in [
            (seg_pre,   "Trước cơn",  "steelblue"),
            (seg_ictal, "Trong cơn",  "red"),
            (seg_post,  "Sau cơn",    "green"),
        ]:
            freqs_f = np.fft.rfftfreq(len(sig), d=1/sfreq)
            fft_mag = np.abs(np.fft.rfft(sig))
            mask    = freqs_f <= 50
            fft_data.append((freqs_f[mask], fft_mag[mask], title, color))

        y_max = max(d[1].max() for d in fft_data) * 1.05

        col_scale = st.columns([1, 3])
        use_log = col_scale[0].checkbox("Log scale", value=False,
                                         help="Log scale giúp thấy rõ vùng tần số cao hơn")

        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        fig2.suptitle("Phổ tần số — trục Y chung (dễ so sánh hơn)", fontsize=11)
        for ax, (freqs_f, fft_mag, title, color) in zip(axes2, fft_data):
            ax.fill_between(freqs_f, fft_mag, alpha=0.4, color=color)
            ax.plot(freqs_f, fft_mag, lw=0.8, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Tần số (Hz)")
            ax.set_ylabel("Biên độ FFT")
            if use_log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, y_max)
            for f in [4, 8, 13, 30]:
                ax.axvline(f, color="gray", ls="--", lw=0.6)
            # Label các dải
            for f, lbl in [(2,"δ"), (6,"θ"), (10,"α"), (20,"β"), (35,"γ")]:
                ax.text(f, y_max*0.95 if not use_log else fft_mag.max()*2,
                        lbl, ha="center", fontsize=8, color="gray")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Bảng % năng lượng để so sánh rõ hơn
        st.subheader("% năng lượng theo dải tần số")
        import pandas as pd

        def pct_bands(sig):
            freqs_b = np.fft.rfftfreq(len(sig), d=1/sfreq)
            fft_b   = np.abs(np.fft.rfft(sig)) ** 2
            def bp(lo, hi):
                return float(np.sum(fft_b[(freqs_b>=lo)&(freqs_b<hi)]))
            d=bp(0.5,4); th=bp(4,8); al=bp(8,13); be=bp(13,30); ga=bp(30,40)
            tot = d+th+al+be+ga
            return [round(d/tot*100,1), round(th/tot*100,1),
                    round(al/tot*100,1), round(be/tot*100,1), round(ga/tot*100,1)]

        df_pct = pd.DataFrame(
            [pct_bands(seg_pre), pct_bands(seg_ictal), pct_bands(seg_post)],
            index=["Trước cơn", "Trong cơn", "Sau cơn"],
            columns=["Delta 0.5-4Hz", "Theta 4-8Hz", "Alpha 8-13Hz",
                     "Beta 13-30Hz", "Gamma 30-40Hz"]
        )
        st.dataframe(df_pct.style.highlight_max(axis=0, color="#ffcccc")
                                  .highlight_min(axis=0, color="#ccffcc"),
                     use_container_width=True)
        st.caption("🔴 Đỏ = cao nhất  |  🟢 Xanh = thấp nhất  |  So sánh từng cột để thấy sự dịch chuyển năng lượng")


# ────────────────────────────────────────────────────────────────────────────
# TAB 4: Decision Tree — hướng dẫn từng bước
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Decision Tree — Học từng bước với dữ liệu thực")

    if not has_seizure:
        st.warning("Chọn file có cơn (ví dụ chb01_03.edf) để thấy dữ liệu thực tế.")
        st.stop()

    import pandas as pd
    sz_s, sz_e = seizures[0]

    # ── Tính đặc trưng 4 giai đoạn ────────────────────────────────────────
    seg_normal_data = data[:, max(0, (sz_s-600))*sfreq : max(0, (sz_s-540))*sfreq]
    seg_pre_data    = data[:, (sz_s-300)*sfreq : sz_s*sfreq]
    seg_ictal_data  = data[:, sz_s*sfreq : sz_e*sfreq]
    seg_post_data   = data[:, sz_e*sfreq : min(n_samples, (sz_e+300)*sfreq)]

    rows = {}
    for name, seg in [
        ("Bình thường", seg_normal_data),
        ("Trước cơn",   seg_pre_data),
        ("Trong cơn",   seg_ictal_data),
        ("Sau cơn",     seg_post_data),
    ]:
        if seg.shape[1] > 0:
            rows[name] = compute_features(seg)

    df_feat = pd.DataFrame(rows).T.round(2)
    feat_names = list(df_feat.columns)

    # ── BƯỚC 1: Xem bảng đặc trưng ───────────────────────────────────────
    st.subheader("Bước 1 — Xem bảng đặc trưng của 4 giai đoạn")
    st.caption("""
    Mỗi hàng = 1 giai đoạn. Mỗi cột = 1 đặc trưng.
    Decision Tree sẽ dùng các số này để học cách phân loại.
    Câu hỏi cần trả lời: **cột nào phân biệt "Bình thường" vs "Trong cơn" tốt nhất?**
    """)

    def highlight_max(s):
        return ["background-color: #ffcccc" if v == s.max() else "" for v in s]
    st.dataframe(df_feat.style.apply(highlight_max), use_container_width=True)
    st.caption("🔴 Ô đỏ = giá trị lớn nhất trong cột — đặc trưng nào tăng nhiều nhất khi có cơn?")

    # ── BƯỚC 2: Chọn đặc trưng ───────────────────────────────────────────
    st.subheader("Bước 2 — Chọn đặc trưng để xây dựng cây")
    st.markdown("""
    **Tiêu chí chọn đặc trưng tốt:**
    - Giá trị khác biệt rõ giữa "Bình thường" và "Trong cơn"
    - Ngưỡng phân chia dễ xác định
    - Variance và Line Length thường là lựa chọn tốt nhất vì thay đổi lớn nhất
    """)

    col_f1, col_f2 = st.columns(2)
    feat1 = col_f1.selectbox("Đặc trưng gốc (node 1)", feat_names,
                              index=0,
                              help="Chọn đặc trưng thay đổi nhiều nhất — thường là Variance")
    feat2 = col_f2.selectbox("Đặc trưng nhánh con (node 2)", feat_names,
                              index=min(4, len(feat_names)-1),
                              help="Chọn đặc trưng thứ 2 để phân biệt thêm")

    # Lấy giá trị thực tế
    v_normal = df_feat.loc["Bình thường", feat1] if "Bình thường" in df_feat.index else 0
    v_pre    = df_feat.loc["Trước cơn",   feat1] if "Trước cơn"   in df_feat.index else 0
    v_ictal  = df_feat.loc["Trong cơn",   feat1] if "Trong cơn"   in df_feat.index else 0
    v2_normal = df_feat.loc["Bình thường", feat2] if "Bình thường" in df_feat.index else 0
    v2_pre    = df_feat.loc["Trước cơn",   feat2] if "Trước cơn"   in df_feat.index else 0

    thresh1 = round((v_normal + v_ictal) / 2, 1)
    thresh2 = round((v2_normal + v2_pre) / 2, 4)

    # ── BƯỚC 3: Tính ngưỡng ──────────────────────────────────────────────
    st.subheader("Bước 3 — Tính ngưỡng phân chia")
    st.markdown(f"""
    Decision Tree chọn ngưỡng = **trung điểm giữa 2 lớp** (đơn giản nhất):

    **Node 1 — {feat1}:**
    - Bình thường = {v_normal:.1f}
    - Trong cơn   = {v_ictal:.1f}
    - Ngưỡng      = ({v_normal:.1f} + {v_ictal:.1f}) / 2 = **{thresh1}**

    **Node 2 — {feat2}:**
    - Bình thường = {v2_normal:.1f}
    - Trước cơn   = {v2_pre:.1f}
    - Ngưỡng      = ({v2_normal:.1f} + {v2_pre:.1f}) / 2 = **{thresh2}**
    """)

    # ── BƯỚC 4: Vẽ cây ───────────────────────────────────────────────────
    st.subheader("Bước 4 — Cây quyết định")

    st.code(f"""
                [CỬA SỔ EEG 10 GIÂY]
                        |
            {feat1} > {thresh1} ?
           /                      \\
         CÓ                      KHÔNG
          |                         |
 {feat2} > {thresh2} ?        → BÌNH THƯỜNG ✓
      /          \\
    CÓ           KHÔNG
     |               |
TRƯỚC CƠN ⚠️    BÌNH THƯỜNG ✓

Kiểm tra với dữ liệu thực:
  Bình thường : {feat1}={v_normal:.1f}  → {thresh1} ? {'CÓ' if v_normal > thresh1 else 'KHÔNG'} → BÌNH THƯỜNG {'✓' if v_normal <= thresh1 else '✗ (sai)'}
  Trước cơn   : {feat1}={v_pre:.1f}    → {thresh1} ? {'CÓ' if v_pre > thresh1 else 'KHÔNG'}
  Trong cơn   : {feat1}={v_ictal:.1f}  → {thresh1} ? {'CÓ' if v_ictal > thresh1 else 'KHÔNG'}
    """, language=None)

    # ── BƯỚC 5: So sánh cây tốt vs xấu ──────────────────────────────────
    st.subheader("Bước 5 — Cây nào tốt hơn cây nào?")

    # Tính tỷ lệ thay đổi để đánh giá đặc trưng
    change_ratios = {}
    for feat in feat_names:
        if "Bình thường" in df_feat.index and "Trong cơn" in df_feat.index:
            v_n = abs(df_feat.loc["Bình thường", feat]) + 1e-10
            v_i = abs(df_feat.loc["Trong cơn",   feat]) + 1e-10
            change_ratios[feat] = round(max(v_n, v_i) / min(v_n, v_i), 2)

    df_change = pd.DataFrame.from_dict(
        change_ratios, orient="index", columns=["Tỷ lệ thay đổi (Trong cơn / Bình thường)"]
    ).sort_values("Tỷ lệ thay đổi (Trong cơn / Bình thường)", ascending=False)

    col_good, col_bad = st.columns(2)

    with col_good:
        st.success("**Cây TỐT — dùng đặc trưng thay đổi nhiều**")
        best_feat = df_change.index[0]
        best_ratio = df_change.iloc[0, 0]
        st.markdown(f"""
        Ví dụ dùng **{best_feat}**:
        - Bình thường: {df_feat.loc['Bình thường', best_feat]:.1f}
        - Trong cơn:   {df_feat.loc['Trong cơn', best_feat]:.1f}
        - Tỷ lệ thay đổi: **{best_ratio}x** ← rất dễ phân biệt
        - Ngưỡng chia rõ ràng → cây ít sai
        """)

    with col_bad:
        st.error("**Cây XẤU — dùng đặc trưng thay đổi ít**")
        worst_feat = df_change.index[-1]
        worst_ratio = df_change.iloc[-1, 0]
        st.markdown(f"""
        Ví dụ dùng **{worst_feat}**:
        - Bình thường: {df_feat.loc['Bình thường', worst_feat]:.1f}
        - Trong cơn:   {df_feat.loc['Trong cơn', worst_feat]:.1f}
        - Tỷ lệ thay đổi: **{worst_ratio}x** ← khó phân biệt
        - 2 lớp gần nhau → cây dễ đoán sai
        """)

    st.subheader("Bảng xếp hạng đặc trưng theo khả năng phân biệt")
    st.caption("Tỷ lệ thay đổi càng lớn = đặc trưng càng tốt để làm node gốc của cây")

    def color_ratio(val):
        if val >= 5:   return "background-color: #c6efce"
        if val >= 2:   return "background-color: #ffeb9c"
        return "background-color: #ffc7ce"

    st.dataframe(
        df_change.style.map(color_ratio),
        use_container_width=True
    )
    st.caption("🟢 Xanh ≥ 5x = rất tốt | 🟡 Vàng 2–5x = tạm được | 🔴 Đỏ < 2x = không nên dùng")

    st.info("""
    **Tóm tắt để trả lời cô:**
    - Chọn đặc trưng có tỷ lệ thay đổi lớn nhất làm node gốc
    - Decision Tree tự động tìm ngưỡng tối ưu (dùng Gini/Entropy)
    - Random Forest = 500 cây như trên, mỗi cây dùng tập con ngẫu nhiên → bỏ phiếu đa số
    - Cây đơn dễ overfit, Random Forest khắc phục bằng cách đa dạng hóa
    """)

# ────────────────────────────────────────────────────────────────────────────
# TAB 5: Tính đặc trưng thủ công
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Tính đặc trưng thủ công")
    st.caption("Kéo slider chọn đoạn thời gian → đọc 10 giá trị đặc trưng → tự vẽ cây quyết định")

    # Gợi ý mốc thời gian
    if has_seizure:
        sz_s0, sz_e0 = seizures[0]
        st.info(f"""
        **Gợi ý 4 đoạn để so sánh** (file {edf_file}, cơn tại giây {sz_s0}–{sz_e0}):
        - **Bình thường**: giây **{max(0, sz_s0-600)}** (cách cơn 10 phút)
        - **Trước cơn**: giây **{max(0, sz_s0-60)}** (1 phút trước cơn)
        - **Trong cơn**: giây **{sz_s0}** (đúng lúc cơn bắt đầu)
        - **Sau cơn**: giây **{sz_e0+30}** (30 giây sau cơn)
        """)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    t_begin = col_ctrl1.slider(
        "Thời điểm bắt đầu cửa sổ (giây)",
        min_value=0,
        max_value=max(0, n_samples // sfreq - 10),
        value=max(0, seizures[0][0] - 600) if has_seizure else 0,
        step=1,
        help="Kéo để chọn đoạn muốn tính. Thử các mốc gợi ý ở trên để so sánh."
    )
    win_len = col_ctrl2.selectbox(
        "Độ dài cửa sổ (giây)",
        [5, 10, 20, 30],
        index=1,
        help="10 giây là mặc định trong nghiên cứu = 2,560 mẫu"
    )
    ch_sel = col_ctrl3.selectbox(
        "Kênh tính",
        ch_names[:n_ch],
        index=0,
        help="Chọn kênh muốn tính. Thử FP1-F7 và T7-P7 để xem có khác nhau không."
    )

    # Cắt đoạn tín hiệu
    ch_i   = ch_names.index(ch_sel)
    s_idx  = t_begin * sfreq
    e_idx  = min(s_idx + win_len * sfreq, n_samples)
    window = data[ch_i, s_idx:e_idx]   # µV

    if len(window) < 10:
        st.warning("Đoạn quá ngắn, kéo slider lại.")
        st.stop()

    # ── Hiển thị tín hiệu thô của cửa sổ ────────────────────────────────
    st.subheader(f"Tín hiệu thô — kênh {ch_sel}, giây {t_begin}–{t_begin+win_len}")

    fig_w, ax_w = plt.subplots(figsize=(14, 2.5))
    t_arr = np.arange(len(window)) / sfreq + t_begin
    ax_w.plot(t_arr, window, lw=0.6, color="steelblue")
    ax_w.set_xlabel("Giây"); ax_w.set_ylabel("µV")
    ax_w.axhline(0, color="gray", lw=0.3, ls="--")
    # Đánh dấu nếu trong cơn
    for sz_s2, sz_e2 in seizures:
        if sz_s2 <= t_begin + win_len and sz_e2 >= t_begin:
            ax_w.axvspan(max(sz_s2, t_begin), min(sz_e2, t_begin+win_len),
                         color="red", alpha=0.3, label="Cơn")
            ax_w.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_w); plt.close()

    # ── Hiển thị giá trị số thực ─────────────────────────────────────────
    with st.expander("Xem giá trị số thô (10 mẫu đầu)"):
        import pandas as pd
        df_raw_vals = pd.DataFrame({
            "Mẫu": range(10),
            "t (s)": [round(t_begin + i/sfreq, 4) for i in range(10)],
            "x (µV)": np.round(window[:10], 4),
        })
        st.dataframe(df_raw_vals, use_container_width=True)
        st.caption(f"Tổng {len(window)} mẫu trong cửa sổ này ({win_len}s × {sfreq}Hz)")

    # ── Tính 10 đặc trưng ────────────────────────────────────────────────
    st.subheader("10 đặc trưng của cửa sổ này")

    # Thời gian
    var_val  = float(np.var(window))
    amp_val  = float(np.mean(np.abs(window)))
    p2p_val  = float(window.max() - window.min())
    ll_val   = float(np.sum(np.abs(np.diff(window))))
    zcr_val  = float(np.sum(np.diff(np.sign(window)) != 0) / len(window))

    # Tần số
    freqs_w = np.fft.rfftfreq(len(window), d=1/sfreq)
    fft_w   = np.abs(np.fft.rfft(window)) ** 2

    def bp_w(lo, hi):
        return float(np.sum(fft_w[(freqs_w >= lo) & (freqs_w < hi)]))

    d_p  = bp_w(0.5, 4);  th_p = bp_w(4, 8)
    al_p = bp_w(8, 13);   be_p = bp_w(13, 30);  ga_p = bp_w(30, 40)
    tot  = d_p + th_p + al_p + be_p + ga_p + 1e-10

    # Xác định nhãn thực tế
    label_actual = "BÌNH THƯỜNG"
    label_color  = "🟢"
    for sz_s3, sz_e3 in seizures:
        if t_begin >= sz_s3 and (t_begin + win_len) <= sz_e3:
            label_actual = "TRONG CƠN"; label_color = "🔴"; break
        if t_begin >= max(0, sz_s3-300) and t_begin < sz_s3:
            label_actual = "TRƯỚC CƠN"; label_color = "🟡"; break
        if t_begin >= sz_e3 and t_begin < sz_e3+300:
            label_actual = "SAU CƠN"; label_color = "🟠"; break

    st.markdown(f"**Nhãn thực tế của đoạn này: {label_color} {label_actual}**")

    # Bảng kết quả
    import pandas as pd
    df_result = pd.DataFrame([
        {"Nhóm": "Thời gian", "Đặc trưng": "Variance (µV²)",       "Giá trị": round(var_val, 2),         "Công thức": "mean((x - mean(x))²)"},
        {"Nhóm": "Thời gian", "Đặc trưng": "Mean Amplitude (µV)",   "Giá trị": round(amp_val, 4),         "Công thức": "mean(|x|)"},
        {"Nhóm": "Thời gian", "Đặc trưng": "Peak-to-Peak (µV)",     "Giá trị": round(p2p_val, 2),         "Công thức": "max(x) - min(x)"},
        {"Nhóm": "Thời gian", "Đặc trưng": "Line Length",           "Giá trị": round(ll_val, 1),          "Công thức": "sum(|x[i+1] - x[i]|)"},
        {"Nhóm": "Thời gian", "Đặc trưng": "Zero Crossing Rate",    "Giá trị": round(zcr_val, 4),         "Công thức": "số lần đổi dấu / N"},
        {"Nhóm": "Tần số",    "Đặc trưng": "Delta % (0.5–4 Hz)",    "Giá trị": round(d_p/tot*100, 2),     "Công thức": "BP(0.5,4) / tổng × 100"},
        {"Nhóm": "Tần số",    "Đặc trưng": "Theta % (4–8 Hz)",      "Giá trị": round(th_p/tot*100, 2),    "Công thức": "BP(4,8) / tổng × 100"},
        {"Nhóm": "Tần số",    "Đặc trưng": "Alpha % (8–13 Hz)",     "Giá trị": round(al_p/tot*100, 2),    "Công thức": "BP(8,13) / tổng × 100"},
        {"Nhóm": "Tần số",    "Đặc trưng": "Beta % (13–30 Hz)",     "Giá trị": round(be_p/tot*100, 2),    "Công thức": "BP(13,30) / tổng × 100"},
        {"Nhóm": "Tần số",    "Đặc trưng": "Gamma % (30–40 Hz)",    "Giá trị": round(ga_p/tot*100, 2),    "Công thức": "BP(30,40) / tổng × 100"},
    ])
    st.dataframe(df_result, use_container_width=True)

    # ── Ghi chú cách tự vẽ cây ───────────────────────────────────────────
    st.subheader("Cách dùng bảng này để tự vẽ cây")
    st.markdown(f"""
    **Bước 1:** Chạy lần lượt 4 đoạn gợi ý (bình thường → trước cơn → trong cơn → sau cơn)
    → Ghi lại giá trị Variance và Line Length của từng đoạn vào 1 bảng

    **Bước 2:** Chọn đặc trưng node gốc = cái có giá trị khác biệt nhiều nhất giữa
    "Bình thường" và "Trong cơn"

    **Bước 3:** Tính ngưỡng = (giá trị bình thường + giá trị trong cơn) / 2

    **Bước 4:** Vẽ cây:
    ```
    [đặc trưng] > [ngưỡng]?
        CÓ → ...
        KHÔNG → BÌNH THƯỜNG
    ```

    **Bước 5:** Kiểm tra lại — kéo slider đến từng đoạn và xem cửa sổ đó đi theo nhánh nào,
    có đúng với nhãn thực tế không?
    """)
    st.caption(f"Nhãn hiện tại: **{label_actual}** | Variance = {var_val:.1f} | Line Length = {ll_val:.0f}")

# ────────────────────────────────────────────────────────────────────────────
# TAB 6: So sanh 3 mo hinh
# ────────────────────────────────────────────────────────────────────────────
with tab6:
    st.header("Mô hình")
    st.caption("Chạy 3 mô hình trên dữ liệu chb01 — 10 đặc trưng — cửa sổ 10 giây")

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  ConfusionMatrixDisplay, roc_auc_score,
                                  recall_score, roc_curve)
    from sklearn.model_selection import train_test_split

    FEATURE_NAMES_6 = [
        "Variance", "MeanAmp", "PeakToPeak", "LineLength", "ZCR",
        "Delta%", "Theta%", "Alpha%", "Beta%", "Gamma%"
    ]
    WIN_SEC_6  = 10
    STEP_SEC_6 = 5
    SPH_SEC_6  = 300

    SEIZURE_INFO_6 = {
        "chb01_03.edf": [(2996, 3036)],
        "chb01_04.edf": [(1467, 1494)],
        "chb01_15.edf": [(1732, 1772)],
        "chb01_16.edf": [(1015, 1066)],
        "chb01_18.edf": [(1720, 1810)],
        "chb01_21.edf": [(327,  420)],
        "chb01_26.edf": [(1862, 1963)],
    }

    @st.cache_data(show_spinner="Đang trích xuất đặc trưng (lần đầu ~2 phút)...")
    def build_dataset_tab6():
        X_all, y_all = [], []
        WIN  = WIN_SEC_6  * SFREQ
        STEP = STEP_SEC_6 * SFREQ
        subj_dir_6 = os.path.join(DATA_ROOT, "chb01") if DATA_ROOT else None
        for fname, seizures_6 in SEIZURE_INFO_6.items():
            fpath = os.path.join(subj_dir_6, fname) if subj_dir_6 else None
            if not fpath or not os.path.exists(fpath):
                continue
            raw6 = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw6.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)
            d6 = raw6.get_data() * 1e6
            ns = d6.shape[1]
            for sz_s6, sz_e6 in seizures_6:
                pre_start6 = max(0, sz_s6 - SPH_SEC_6)
                for i in range(0, ns - WIN, STEP):
                    t_s6 = i / SFREQ; t_e6 = (i + WIN) / SFREQ
                    w6 = d6[:, i:i+WIN]
                    f6 = list(compute_features(w6).values())
                    if not (t_e6 < sz_s6 or t_s6 > sz_e6):
                        X_all.append(f6); y_all.append(1)
                    elif t_e6 < pre_start6:
                        X_all.append(f6); y_all.append(0)
        return np.array(X_all), np.array(y_all)

    if not (DATA_ROOT and os.path.exists(os.path.join(DATA_ROOT, "chb01", "chb01_03.edf"))):
        st.warning("Cần có dữ liệu local của chb01 để chạy tab này.")
        st.stop()

    X6, y6 = build_dataset_tab6()
    st.success(f"Dữ liệu: {np.sum(y6==0)} cửa sổ bình thường | {np.sum(y6==1)} cửa sổ trong cơn")

    X_tr6, X_te6, y_tr6, y_te6 = train_test_split(
        X6, y6, test_size=0.3, random_state=42, stratify=y6)

    sc6 = StandardScaler()
    X_tr_sc = sc6.fit_transform(X_tr6)
    X_te_sc  = sc6.transform(X_te6)

    # Chọn mô hình để xem
    mo_hinh_chon = st.radio(
        "Chọn mô hình muốn xem:",
        ["Decision Tree", "SVM", "Random Forest"],
        horizontal=True
    )

    # ── Hàm hiển thị kết quả chung ─────────────────────────────────────────
    def hien_thi_ket_qua(name, y_pred, y_prob, color):
        cm6 = confusion_matrix(y_te6, y_pred)
        TP = cm6[1,1]; FN = cm6[1,0]; FP = cm6[0,1]; TN = cm6[0,0]
        sens = TP/(TP+FN)*100 if (TP+FN)>0 else 0
        hrs  = (TN+FP)*WIN_SEC_6/3600
        fpr_h = FP/hrs if hrs>0 else 0
        auc  = roc_auc_score(y_te6, y_prob)

        # Bảng chỉ số
        st.subheader(f"Kết quả — {name}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",    f"{float(np.mean(y_pred==y_te6))*100:.1f}%",
                    help="Tỷ lệ đoán đúng tổng thể")
        col2.metric("Sensitivity", f"{sens:.1f}%",
                    help="% cơn được phát hiện — quan trọng nhất")
        col3.metric("FPR/h",       f"{fpr_h:.3f}",
                    help="Số lần báo nhầm mỗi giờ — tiêu chuẩn < 0.15")
        col4.metric("ROC-AUC",     f"{auc:.4f}",
                    help="Càng gần 1 càng tốt")

        # Giải thích TP TN FP FN
        st.markdown(f"""
        **Đọc Confusion Matrix:**
        | | Đoán: Bình thường | Đoán: Trong cơn |
        |---|---|---|
        | **Thực tế: Bình thường** | TN = {TN} ✓ | FP = {FP} ← báo nhầm |
        | **Thực tế: Trong cơn** | FN = {FN} ← bỏ sót | TP = {TP} ✓ |

        - **TP = {TP}**: cơn thật, đoán đúng là cơn ✅
        - **TN = {TN}**: bình thường thật, đoán đúng ✅
        - **FP = {FP}**: bình thường nhưng báo là cơn ❌ → báo nhầm → FPR/h = {fpr_h:.3f}
        - **FN = {FN}**: cơn thật nhưng bỏ sót ❌ → Sensitivity = {sens:.1f}%
        """)

        # Biểu đồ
        col_cm, col_roc = st.columns(2)
        with col_cm:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(cm6, display_labels=["Binh thuong","Trong con"]
                                   ).plot(ax=ax_cm, cmap=color, values_format="d", colorbar=False)
            ax_cm.set_title(f"Confusion Matrix\n{name}"); ax_cm.grid(False)
            st.pyplot(fig_cm); plt.close()

        with col_roc:
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            fpr_c, tpr_c, _ = roc_curve(y_te6, y_prob)
            ax_roc.plot(fpr_c, tpr_c, lw=2, label=f"AUC={auc:.3f}")
            ax_roc.plot([0,1],[0,1], "gray", ls="--", lw=1)
            ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR (Sensitivity)")
            ax_roc.set_title(f"ROC Curve\n{name}")
            ax_roc.legend(); ax_roc.grid(True, ls="--", alpha=0.4)
            st.pyplot(fig_roc); plt.close()

        st.caption(classification_report(y_te6, y_pred, zero_division=0,
                                          target_names=["Binh thuong","Trong con"]))

    # ══════════════════════════════════════════════════════════════════════
    # MÔ HÌNH 1: DECISION TREE
    # ══════════════════════════════════════════════════════════════════════
    if mo_hinh_chon == "Decision Tree":
        st.markdown("""
        **Decision Tree là gì?**
        Cây hỏi từng câu hỏi Yes/No dựa trên một đặc trưng. Tại mỗi node,
        cây chọn đặc trưng và ngưỡng làm giảm Gini Impurity nhiều nhất.
        """)

        st.subheader("Quy trình Decision Tree từng bước")
        st.markdown(f"""
        **Bước 1 — Dữ liệu đầu vào (X_train):**
        Mỗi hàng = 1 cửa sổ EEG 10 giây, mỗi cột = 1 đặc trưng. Nhãn y = 0 (BT) hoặc 1 (TC).
        ```
        Cửa sổ  | Variance | LineLength | Alpha% | ...  | Nhãn
        ─────────────────────────────────────────────────────
        CW_001   |   1218   |    9950    |  14.0  | ...  |   0  ← bình thường
        CW_002   |   6278   |   19376    |   1.8  | ...  |   1  ← trong cơn
        CW_003   |   1105   |    9200    |  12.5  | ...  |   0  ← bình thường
        ...      | {X_tr6.shape[0]} hàng train
        ```

        **Bước 2 — Cây chọn node gốc:**
        Thử tất cả (đặc trưng, ngưỡng) → chọn cái giảm Gini nhiều nhất.
        Gini trước chia = 1 - ((p_BT)² + (p_TC)²).
        Ví dụ: {np.sum(y_tr6==0)} BT + {np.sum(y_tr6==1)} TC → Gini = {1 - (np.sum(y_tr6==0)/len(y_tr6))**2 - (np.sum(y_tr6==1)/len(y_tr6))**2:.3f}

        **Bước 3 — Chia nhánh đệ quy:**
        Mỗi nhánh lại hỏi câu hỏi tiếp theo trên tập con → tiếp tục đến max_depth=4.

        **Bước 4 — Lá (leaf node):**
        Khi đạt max_depth hoặc Gini=0 → gán nhãn = lớp chiếm đa số tại lá đó.

        **Bước 5 — Dự đoán cửa sổ mới:**
        Đi từ gốc xuống lá theo câu trả lời Yes/No → ra nhãn ở lá.
        """)

        with st.spinner("Huấn luyện Decision Tree..."):
            dt6 = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
            dt6.fit(X_tr6, y_tr6)
            y_pred_dt = dt6.predict(X_te6)
            y_prob_dt = dt6.predict_proba(X_te6)[:, 1]

        hien_thi_ket_qua("Decision Tree", y_pred_dt, y_prob_dt, "Blues")

        st.subheader("Cây quyết định — dạng text")
        st.caption("Đọc từ trên xuống: mỗi dòng = 1 câu hỏi. class: 0 = Bình thường, class: 1 = Trong cơn")
        st.code(export_text(dt6, feature_names=FEATURE_NAMES_6, max_depth=4), language=None)

        st.subheader("Cây quyết định — hình ảnh (2 mức đầu)")
        st.caption("Màu xanh = Bình thường, màu cam = Trong cơn. gini=0 là phân loại hoàn toàn đúng.")
        fig_tree, ax_tree = plt.subplots(figsize=(14, 6))
        plot_tree(dt6, feature_names=FEATURE_NAMES_6, class_names=["BT","TC"],
                  filled=True, max_depth=2, ax=ax_tree, fontsize=8)
        st.pyplot(fig_tree); plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # MÔ HÌNH 2: SVM
    # ══════════════════════════════════════════════════════════════════════
    elif mo_hinh_chon == "SVM":
        st.markdown("""
        **SVM là gì?**
        SVM tìm đường ranh giới (hyperplane) phân cách 2 lớp, tối đa hóa
        khoảng cách đến các điểm gần nhất (support vectors).
        Dùng kernel RBF để xử lý dữ liệu không phân tách tuyến tính.

        **Lưu ý:** SVM cần chuẩn hóa dữ liệu (StandardScaler) trước khi train.
        """)

        st.subheader("Quy trình SVM từng bước")
        st.markdown(f"""
        **Bước 1 — Chuẩn hóa dữ liệu (bắt buộc với SVM):**
        Variance có thể = 9000, còn ZCR = 0.05 → chênh lệch 180,000 lần.
        StandardScaler đưa tất cả về mean=0, std=1.
        ```
        Trước chuẩn hóa: Variance ∈ [500, 10000],  ZCR ∈ [0.02, 0.15]
        Sau chuẩn hóa  : Variance ∈ [-1.5, 2.3],   ZCR ∈ [-1.2, 1.8]
        ```

        **Bước 2 — Tìm hyperplane:**
        Trong không gian 10 chiều (10 đặc trưng), SVM tìm siêu phẳng
        phân cách BT và TC với margin lớn nhất.
        Các điểm gần ranh giới nhất = **support vectors** — chỉ chúng quyết định ranh giới.

        **Bước 3 — Kernel RBF:**
        Nếu 2 lớp không phân tách thẳng được → RBF chiếu lên không gian cao hơn
        → vẫn tìm được đường thẳng ở đó.
        Tham số C=10: cho phép một số điểm vi phạm margin (soft margin).

        **Bước 4 — Dự đoán:**
        Cửa sổ mới → chuẩn hóa → xem nằm phía nào của hyperplane → gán nhãn.

        **Khác Decision Tree:**
        SVM không cho biết đặc trưng nào quan trọng nhất.
        SVM nhạy với outlier nếu không chuẩn hóa.
        """)

        with st.spinner("Huấn luyện SVM (chậm hơn Decision Tree)..."):
            svm6 = SVC(kernel="rbf", C=10, gamma="scale",
                       class_weight="balanced", probability=True, random_state=42)
            svm6.fit(X_tr_sc, y_tr6)
            y_pred_svm = svm6.predict(X_te_sc)
            y_prob_svm = svm6.predict_proba(X_te_sc)[:, 1]

        hien_thi_ket_qua("SVM (RBF kernel)", y_pred_svm, y_prob_svm, "Oranges")

        st.info("""
        **SVM khác Decision Tree ở chỗ nào?**
        - Decision Tree: hỏi từng câu hỏi rõ ràng, dễ đọc
        - SVM: tìm đường biên tối ưu, khó giải thích nhưng thường tổng quát hóa tốt hơn
        - SVM cần chuẩn hóa dữ liệu (các đặc trưng về cùng scale)
        - SVM không cho biết đặc trưng nào quan trọng nhất
        """)

    # ══════════════════════════════════════════════════════════════════════
    # MÔ HÌNH 3: RANDOM FOREST
    # ══════════════════════════════════════════════════════════════════════
    elif mo_hinh_chon == "Random Forest":
        st.markdown("""
        **Random Forest là gì?**
        100 Decision Tree, mỗi cây học trên tập con ngẫu nhiên của dữ liệu
        và đặc trưng. Kết quả cuối = bỏ phiếu đa số từ 100 cây.
        """)

        st.subheader("Quy trình Random Forest từng bước")
        st.markdown(f"""
        **Bước 1 — Bootstrap sampling:**
        Từ {X_tr6.shape[0]} cửa sổ train, lấy ngẫu nhiên có hoàn lại {X_tr6.shape[0]} cửa sổ
        → mỗi cây nhận tập train khác nhau (~63% unique, ~37% trùng).
        ```
        Cây 1: lấy ngẫu nhiên CW_001, CW_001, CW_045, CW_203, CW_001, ...
        Cây 2: lấy ngẫu nhiên CW_089, CW_012, CW_045, CW_089, CW_203, ...
        ...100 cây, 100 tập train khác nhau
        ```

        **Bước 2 — Feature subset (random subspace):**
        Tại mỗi node, thay vì xét tất cả 10 đặc trưng → chỉ xét √10 ≈ 3 đặc trưng ngẫu nhiên.
        → Các cây đa dạng hơn, ít tương quan với nhau hơn.

        **Bước 3 — Xây 100 cây độc lập:**
        Mỗi cây = 1 Decision Tree đầy đủ (max_depth=10) trên tập train riêng.
        Không cần chuẩn hóa (khác SVM).

        **Bước 4 — Bỏ phiếu đa số:**
        Cửa sổ mới → đưa qua 100 cây → mỗi cây cho 1 nhãn:
        ```
        Cây 1  → BT (0)
        Cây 2  → TC (1)
        Cây 3  → BT (0)
        ...
        Cây 98 → BT (0)
        Cây 99 → BT (0)
        Cây 100→ BT (0)
        ─────────────────
        BT: 92 phiếu  TC: 8 phiếu → KẾT QUẢ: BT
        Xác suất TC = 8/100 = 0.08  (dùng cho ROC curve)
        ```

        **Tại sao tốt hơn 1 cây?**
        1 cây dễ overfit (học thuộc train). 100 cây đa dạng → trung bình hóa sai số
        → ổn định hơn trên dữ liệu mới.
        """)

        with st.spinner("Huấn luyện Random Forest (100 cây)..."):
            rf6 = RandomForestClassifier(n_estimators=100, max_depth=10,
                                          class_weight="balanced_subsample",
                                          random_state=42, n_jobs=-1)
            rf6.fit(X_tr6, y_tr6)
            y_pred_rf = rf6.predict(X_te6)
            y_prob_rf = rf6.predict_proba(X_te6)[:, 1]

        hien_thi_ket_qua("Random Forest", y_pred_rf, y_prob_rf, "Greens")

        st.subheader("Feature Importance — đặc trưng nào quan trọng nhất?")
        st.caption("Random Forest tự tính mức độ đóng góp của từng đặc trưng vào quyết định phân loại")
        imp6 = rf6.feature_importances_
        df_imp = pd.DataFrame({"Dac trung": FEATURE_NAMES_6, "Importance": imp6}
                              ).sort_values("Importance", ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
        ax_imp.bar(df_imp["Dac trung"], df_imp["Importance"], color="forestgreen", alpha=0.8)
        ax_imp.set_ylabel("Importance (0→1)")
        ax_imp.set_title("Đặc trưng nào Random Forest dùng nhiều nhất?")
        ax_imp.tick_params(axis="x", rotation=30)
        ax_imp.grid(axis="y", ls="--", alpha=0.4)
        for i, (_, row) in enumerate(df_imp.iterrows()):
            ax_imp.text(i, row["Importance"]+0.003, f"{row['Importance']:.3f}",
                        ha="center", fontsize=8)
        st.pyplot(fig_imp); plt.close()

        st.info("""
        **Đọc Feature Importance:**
        - LineLength và Variance cao nhất → xác nhận đúng phân tích thủ công
        - Đây là lý do chọn 2 đặc trưng này làm ví dụ Decision Tree đơn giản
        - Random Forest dùng tất cả 10 đặc trưng, không chỉ 2 cái
        """)

