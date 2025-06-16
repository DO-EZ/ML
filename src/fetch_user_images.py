import os
import zipfile
import requests

def fetch_and_extract_user_images(
    be_base_url: str,
    output_dir: str = "user_data/images",
    zip_name: str = "user_images.zip",
    timeout: int = 30
):
    """
    BE의 /images 엔드포인트를 호출해서 ZIP 받아오고,
    user_data/images 폴더에 압축 해제.
    """
    endpoint = be_base_url.rstrip("/") + "/images"
    print(f"[FetchUserImages] GET {endpoint}")
    response = requests.get(endpoint, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"BE /images 요청 실패: {response.status_code} {response.text}")

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    os.remove(zip_path)
    print(f"[FetchUserImages] 압축 해제 완료 → {output_dir}")
