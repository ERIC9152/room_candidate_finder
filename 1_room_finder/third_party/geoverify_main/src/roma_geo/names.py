"""從路徑取出房子名與影像識別碼。"""
import glob
import os

from .errors import RomaGeoError

#: 全景影像副檔名（小寫）。比對時忽略大小寫。
IMAGE_EXTENSIONS = (".jpg", ".jpeg")

#: Windows 檔名不接受的字元。本專案的房子名會當檔名用，各平台同一套規則。
_WINDOWS_BAD = '<>:"|?*'


def is_path_component(name):
    """名稱可否作為單一檔名成分（不含路徑分隔、上層參照或 Windows 非法字元）。"""
    if not isinstance(name, str) or not name:
        return False
    if name in (".", ".."):
        return False
    if os.path.sep in name or "/" in name or "\\" in name:
        return False
    if os.path.isabs(name):
        return False
    if any(ord(c) < 32 for c in name):
        return False
    if any(c in name for c in _WINDOWS_BAD):
        return False
    return os.path.basename(name) == name


def glob_in_dir(directory, pattern, recursive=False):
    """在 directory 底下套用 glob。目錄本身含 `*`、`?`、`[` 時不當萬用字元。"""
    if not directory:
        return []
    base = glob.escape(os.path.abspath(directory))
    return glob.glob(os.path.join(base, pattern), recursive=recursive)


def require_path_component(name, input_name="識別名稱"):
    """用於組出檔案路徑的名稱。不合法時丟 RomaGeoError。"""
    if is_path_component(name):
        return name
    raise RomaGeoError(
        "名稱 %r 不可用於檔案路徑" % (name,),
        input_name=input_name,
        condition="不得含路徑分隔、上層參照或檔名非法字元",
    )


def require_pair_names(house, img_a, img_b, input_name=None):
    """房子名與兩張影像識別碼都須可作為單一檔名成分。"""
    label = input_name or ("房子 %s" % house)
    require_path_component(house, input_name=label)
    require_path_component(img_a, input_name=label)
    require_path_component(img_b, input_name=label)
    return house, img_a, img_b


def dest_in_dir(directory, name, suffix=".json", input_name=None):
    """把名稱組成 directory 底下的檔案路徑。不合法或會寫出目錄之外時丟 RomaGeoError。"""
    label = input_name or ("房子 %s" % name)
    require_path_component(name, input_name=label)
    path = os.path.join(directory, "%s%s" % (name, suffix))
    parent = os.path.abspath(directory)
    dest = os.path.abspath(path)
    if dest != parent and not dest.startswith(parent + os.sep):
        raise RomaGeoError(
            "名稱 %r 會寫出目錄之外" % name,
            input_name=label,
            condition="不得含路徑分隔、上層參照或檔名非法字元",
        )
    return path


def house_of(path):
    """從影像路徑取出房子名。

    取影像的直接上層目錄名。資料集根目錄即使名稱含 DollhouseTask 或 direct_match，
    也不會蓋掉各房子自己的目錄名。路徑只剩檔名時回傳 unknown。
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def image_stem(path):
    """影像檔名去掉副檔名。"""
    return os.path.splitext(os.path.basename(str(path)))[0]


def is_image_filename(name):
    """檔名是否為本專案接受的全景影像。"""
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def dir_has_images(directory):
    """目錄當層是否含有全景影像。"""
    if not directory or not os.path.isdir(directory):
        return False
    try:
        for name in os.listdir(directory):
            if is_image_filename(name) and os.path.isfile(os.path.join(directory, name)):
                return True
    except OSError:
        return False
    return False


def find_image_file(directory, stem):
    """在目錄當層找 `stem` 加上 jpg／jpeg（大小寫皆可）。找不到回 None。"""
    if not directory or not os.path.isdir(directory):
        return None
    try:
        for name in os.listdir(directory):
            root, ext = os.path.splitext(name)
            if root == stem and ext.lower() in IMAGE_EXTENSIONS:
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    return path
    except OSError:
        return None
    return None
