"""
Tạo gdrive_index.json từ Google Drive.

CÁCH DÙNG:
1. Upload toàn bộ D:\NCKH\data\chbmit\ lên Google Drive
2. Share thư mục gốc "chbmit" -> Anyone with link -> Viewer
3. Chạy script này với Folder ID của thư mục chbmit:
      python build_gdrive_index.py <FOLDER_ID>

Ví dụ:
      python build_gdrive_index.py 1AbCdEfGhIjKlMnOpQrStUvWx

Script sẽ duyệt qua từng subfolder chbXX và lưu file_id của mỗi file EDF.
Kết quả: gdrive_index.json -> push lên GitHub cùng code.
"""

import sys, json, re, requests

def get_file_ids_in_folder(folder_id):
    """
    Lấy {filename: file_id} trong 1 folder Drive công khai.
    Dùng Google Drive API v3 (không cần auth cho folder public).
    """
    url   = "https://www.googleapis.com/drive/v3/files"
    key   = "AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY"  # public API key (read-only)
    items = {}
    page_token = None

    while True:
        params = {
            "q"       : f"'{folder_id}' in parents and trashed=false",
            "fields"  : "nextPageToken,files(id,name,mimeType)",
            "pageSize": 1000,
            "key"     : key,
        }
        if page_token:
            params["pageToken"] = page_token

        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  LỖI API: {r.status_code} — {r.text[:200]}")
            print("  -> Folder có thể chưa được share public hoặc API key không hợp lệ")
            break

        data = r.json()
        for f in data.get("files", []):
            items[f["name"]] = {"id": f["id"], "type": f["mimeType"]}

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return items


def build_index(root_folder_id):
    print(f"Bắt đầu scan folder: {root_folder_id}")
    root_items = get_file_ids_in_folder(root_folder_id)

    if not root_items:
        print("Không lấy được danh sách file. Kiểm tra:")
        print("  1. Folder ID đúng chưa?")
        print("  2. Đã share 'Anyone with the link' chưa?")
        return

    index = {}

    for name, info in sorted(root_items.items()):
        if info["type"] == "application/vnd.google-apps.folder" and name.startswith("chb"):
            print(f"  Scanning {name}...", end=" ", flush=True)
            subfiles = get_file_ids_in_folder(info["id"])
            edf_files = {k: v["id"] for k, v in subfiles.items() if k.endswith(".edf") or k.endswith(".txt")}
            index[name] = edf_files
            print(f"{len(edf_files)} files")

    output = "gdrive_index.json"
    with open(output, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nXong! Lưu tại: {output}")
    print(f"Tổng: {len(index)} bệnh nhân, {sum(len(v) for v in index.values())} files")
    print("\nBước tiếp theo: push gdrive_index.json lên GitHub cùng eeg_explorer.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python build_gdrive_index.py <FOLDER_ID>")
        print("Ví dụ   : python build_gdrive_index.py 1AbCdEfGhIjKlMnOpQ")
    else:
        build_index(sys.argv[1])
