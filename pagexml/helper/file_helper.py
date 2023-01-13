from typing import Generator, List, Literal, Tuple, Union
import os
import tarfile
from zipfile import ZipFile
from tarfile import TarFile
import zipfile

from tqdm import tqdm
import py7zr


ZIP_EXTENSIONS = {'.zip', '.7z'}


def parse_archived_filename(archived_fname: str) -> Tuple[str, str, str]:
    """Split the full pathname of an archive file into directory, file base name and extension."""
    if archived_fname.count('\\') > 0 and archived_fname.count('/') == 0:
        dir_separator = '\\'
    elif archived_fname.count('\\') == 0 and archived_fname.count('/') > 0:
        dir_separator = '/'
    elif archived_fname.count('\\') == 0 and archived_fname.count('/') == 0:
        dir_separator = None
    else:
        dir_separator = os.sep
        archived_fname = os.path.normpath(archived_fname)

    if dir_separator == os.sep:
        archived_fname_dir, archived_fname_file = os.path.split(archived_fname)
    elif dir_separator is None:
        archived_fname_dir = ''
        archived_fname_file = archived_fname
    else:
        # directory separator is different from OS default. Use string split
        *archived_fname_dirs, archived_fname_file = archived_fname.split(dir_separator)
        archived_fname_dir = '\\'.join(archived_fname_dirs)
    archived_fname_base, archived_fname_ext = os.path.splitext(archived_fname_file)
    return archived_fname_dir, archived_fname_file, archived_fname_ext


def get_archived_file_names(archive_handle: Union[TarFile, ZipFile]) -> List[str]:
    if isinstance(archive_handle, ZipFile):
        return archive_handle.namelist()
    elif isinstance(archive_handle, TarFile):
        return archive_handle.getnames()


def read_archive_handle(archive_fname: str, archive_handle: Union[TarFile, ZipFile],
                        filenames_only: bool = False) -> Generator[Tuple[dict, Union[str, None]], None, None]:
    archived_filenames = get_archived_file_names(archive_handle)
    for archived_fname in archived_filenames:
        archived_fname_dir, archived_fname_file, archived_fname_ext = parse_archived_filename(archived_fname)
        with archive_handle.open(archived_fname) as fh:
            file_info = {
                'source_file': [archive_fname],
                'archived_filename': archived_fname_file,
                'archived_filepath': archived_fname
            }
            if archived_fname_ext in ZIP_EXTENSIONS:
                if archived_fname_ext == '.zip':
                    zip_func = zipfile.ZipFile
                    read_func = read_archive_handle
                elif archived_fname_ext == '.7z':
                    zip_func = py7zr.SevenZipFile
                    read_func = read_7z_handle
                with zip_func(fh, mode='r') as inner_zip_handle:
                    for inner_file_info, file_data in read_func(archived_fname,
                                                                inner_zip_handle,
                                                                filenames_only=filenames_only):
                        inner_file_info['source_file'] = file_info['source_file'] + inner_file_info['source_file']
                        yield inner_file_info, file_data
            else:
                file_data = fh.read() if filenames_only is False else None
                yield file_info, file_data


def read_7z_handle(page_7z_file: str, page_7z_handle: py7zr.SevenZipFile,
                   filenames_only: bool = False) -> Generator[Tuple[dict, Union[str, None]], None, None]:
    for zipped_fname, zipped_file_data in page_7z_handle.readall().items():
        zipped_fname_dir, zipped_fname_file, zipped_fname_ext = parse_archived_filename(zipped_fname)
        file_info = {
            'source_file': [page_7z_file],
            'archived_filename': zipped_fname_file,
            'archived_filepath': zipped_fname
        }
        if filenames_only is True:
            yield file_info, None
        else:
            yield file_info, zipped_file_data


def read_page_7z_file(page_7z_file: str,
                      filenames_only: bool = False) -> Generator[Tuple[dict, Union[str, None]], None, None]:
    with py7zr.SevenZipFile(page_7z_file, mode='r') as zh:
        for file_info, file_data in read_7z_handle(page_7z_file, zh, filenames_only=filenames_only):
            yield file_info, file_data


class Extractor:

    def __init__(self, page_archive_file: str, filenames_only: bool = False):
        archiver, read_mode = get_archiver_mode(page_archive_file)
        self.page_archive_file = page_archive_file
        self.filenames_only = filenames_only
        self.archiver = archiver
        self.read_mode = read_mode
        if archiver == "py7zr":
            self.open_func = py7zr.SevenZipFile
            self.handle_func = read_7z_handle
        elif archiver == "tar":
            self.open_func = tarfile.TarFile
            self.handle_func = read_archive_handle
        elif archiver == "zip":
            self.open_func = zipfile.ZipFile
            self.handle_func = read_archive_handle

    def __iter__(self):
        with self.open_func(self.page_archive_file, mode=self.read_mode) as ah:
            extractor = self.handle_func(self.page_archive_file, ah,
                                         filenames_only=self.filenames_only)
            for file_info, file_data in extractor:
                yield file_info, file_data


def get_archiver_mode(page_archive_file: str) -> Tuple[Literal["tar", "zip", "py7zr"],
                                                       Literal["r", "r:", "r:gz", "r:bz2"]]:
    archived_fname_dir, archived_fname_file, archived_fname_ext = parse_archived_filename(page_archive_file)
    if archived_fname_ext in {".tar.gz", ".tgz"}:
        return "tar", "r:gz"
    elif archived_fname_ext in {".tar.bz2", ".tbz2"}:
        return "tar", "r:bz2"
    elif archived_fname_ext == ".tar":
        return "tar", "r:"
    elif archived_fname_ext == ".zip":
        return "zip", "r"
    elif archived_fname_ext == ".7z":
        return "py7zr", "r"
    else:
        raise ValueError(f"Unknown archived file extension {archived_fname_ext} for file {page_archive_file}")


def read_page_archive_file(page_archive_file: str,
                           filenames_only: bool = False,
                           show_progress: bool = False) -> Generator[Tuple[dict, Union[str, None]], None, None]:
    """Read PageXML files from an archive file (e.g. zip, tar or 7z).

    :param page_archive_file: the name of the archive file
    :type page_archive_file: str
    :param filenames_only: whether to return only the archived filenames or also the content (default is False)
    :type filenames_only: bool
    :return: a generator that yields a tuple of archived file name and content
    :param show_progress: whether a TQDM progress bar is shown (default is False)
    :type show_progress: bool
    :rtype: Generator[Tuple[str, str], None, None]
    """
    extractor = Extractor(page_archive_file=page_archive_file, filenames_only=filenames_only)
    if show_progress is True:
        extractor = tqdm(extractor)
    for file_info, file_data in extractor:
        yield file_info, file_data


def read_page_archive_files(page_archive_files: List[str],
                            filenames_only: bool = False,
                            show_progress: bool = False) -> Generator[Tuple[dict, Union[str, None]], None, None]:
    """Read PageXML files from a list of archive file (e.g. zip, tar or 7z).

    :param page_archive_files: the name of the archive file
    :type page_archive_files: List[str]
    :param filenames_only: whether to return only the archived filenames or also the content
    :type filenames_only: bool
    :param show_progress: flag to determine whether a TQDM progress bar is shown
    :type show_progress: bool
    :return: a generator that yields a tuple of archived file name and content
    :rtype: Generator[Tuple[str, str], None, None]
    """
    if isinstance(page_archive_files, str):
        page_archive_files = [page_archive_files]
    for page_archive_file in page_archive_files:
        if show_progress is True:
            extractor = tqdm(read_page_archive_file(page_archive_file, filenames_only=filenames_only),
                             desc=f'extracting PageXML files from {page_archive_file}')
        else:
            extractor = read_page_archive_file(page_archive_file, filenames_only=filenames_only)
        for page_data in extractor:
            yield page_data
