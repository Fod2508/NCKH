import sys, json, re, requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def get_files_in_folder(folder_id):
    """
    Parse file ID tu HTML cua Google Drive folder public.
    Tra ve dict: {filename: file_id}
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        print(f"  LOI HTTP {r.status_code}")
        return {}

    html = r.text
    # Pattern 1: file entries trong JSON embed
    # Drive embed data dang: ["filename",[["file_id",...
    pattern_entry = re.compile(
        r'"([^"]+\.(?:edf|txt|seizures))"[^[]*\[\["([-\w]{25,})"'
    )
    found = {}
    for m in pattern_entry.finditer(html):
        fname, fid = m.group(1), m.group(2)
        found[fname] = fid

    # Pattern 2: folder entries
    pattern_folder = re.compile(
        r'"(chb\d{2})"[^[]*\[\["([-\w]{25,})"'
    )
    folders = {}
    for m in pattern_folder.finditer(html):
        name, fid = m.group(1), m.group(2)
        folders[name] = fid

    return found, folders


def build_index(root_id):
    print(f"Scanning folder: {root_id}")

    files, subfolders = get_files_in_folder(root_id)
    print(f"Root: {len(files)} files truc tiep, {len(subfolders)} subfolder chbXX")

    index = {}

    if subfolders:
        # Cau truc: chbmit/ -> chb01/ -> *.edf
        for subj, sfid in sorted(subfolders.items()):
            print(f"  Scanning {subj}...", end=" ", flush=True)
            sub_files, _ = get_files_in_folder(sfid)
            index[subj] = sub_files
            print(f"{len(sub_files)} files")
    elif files:
        # Cau truc flat: tat ca file trong 1 folder
        # Nhom theo prefix chbXX
        for fname, fid in files.items():
            subj = fname[:5]  # chb01
            if subj not in index:
                index[subj] = {}
            index[subj][fname] = fid
        print(f"Flat structure: {len(index)} benh nhan")
    else:
        print("\nKhong parse duoc file nao tu HTML.")
        print("Co the:")
        print("  1. Folder chua duoc share 'Anyone with link'")
        print("  2. Google thay doi cau truc HTML")
        print("\nThu cach thu cong: mo link Drive, F12 -> Network -> tim request /drive/folders/")
        return

    with open("gdrive_index.json", "w") as f:
        json.dump(index, f, indent=2)

    total = sum(len(v) for v in index.values())
    print(f"\nXong! gdrive_index.json: {len(index)} benh nhan, {total} files")
    if total == 0:
        print("CANH BAO: 0 files duoc index. Kiem tra lai cau truc Drive.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cach dung: python build_gdrive_index.py <FOLDER_ID>")
    else:
        build_index(sys.argv[1])
