
from pathlib import Path

this_file_path: Path = Path(__file__)

RC_EXPORT_BUNDLER_XML_PATH = this_file_path.parent.parent / "configs" / "rc_export_bundler.xml"
RC_EXPORT_IMAGELIST_XML_PATH = this_file_path.parent.parent / "configs" / "rc_export_imagelist.xml"
