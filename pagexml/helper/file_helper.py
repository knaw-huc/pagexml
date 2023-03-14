import os
import tarfile
import zipfile
from tarfile import TarFile
from typing import Dict, Generator, IO, List, Literal, Tuple, Union
from zipfile import ZipFile

import py7zr
from tqdm import tqdm

ZIP_EXTENSIONS = {'.zip', '.7z', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2'}


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


def get_archived_files_infos(archive_handle: Union[TarFile, ZipFile]) -> List[Union[zipfile.ZipInfo, tarfile.TarInfo]]:
    if isinstance(archive_handle, ZipFile):
        return archive_handle.infolist()
    elif isinstance(archive_handle, TarFile):
        return archive_handle.getmembers()


def get_archived_file_names(archive_handle: Union[TarFile, ZipFile]) -> List[str]:
    if isinstance(archive_handle, ZipFile):
        return archive_handle.namelist()
    elif isinstance(archive_handle, TarFile):
        return archive_handle.getnames()


def read_tar_handle(archive_fname: str, archive_handle: TarFile, filenames_only: bool = False):
    # print('reading from tar handle:', archive_fname)
    for tarfile_info in archive_handle:
        if tarfile_info.isdir():
            continue
        archived_dir, archived_file, archived_file_ext = parse_archived_filename(tarfile_info.name)
        file_info = {
            'source_file': [archive_fname],
            'archived_filename': archived_file,
            'archived_filepath': tarfile_info.name
        }
        if archived_file_ext in ZIP_EXTENSIONS:
            # print('read_tar_handle\ttarred file is archive:', archived_file)
            file_reader = archive_handle.extractfile(tarfile_info)
            for inner_file_info, file_data in read_inner_archive(tarfile_info.name,
                                                                 archived_file_ext,
                                                                 file_reader, file_info,
                                                                 filenames_only=filenames_only):
                yield inner_file_info, file_data

        elif filenames_only is True:
            yield file_info, None
        else:
            file_reader = archive_handle.extractfile(tarfile_info)
            file_content = file_reader.read()
            yield file_info, file_content


def read_inner_archive(archived_filename: str, archived_file_ext: str,
                       archived_file_handle: Union[IO[bytes], bytes],
                       file_info: Dict[str, any], filenames_only: bool = False):
    # print('archived_file_ext:', archived_file_ext)
    archiver, read_mode = get_archiver_mode(archived_filename)
    # print('archiver:', archiver, '\tread_mode:', read_mode)
    open_func, read_func = get_archive_functions(archiver)
    # print('archived_file_handle:', archived_file_handle)
    # print('open_func:', open_func)
    if archiver == 'tar':
        inner_archive_handle = open_func(fileobj=archived_file_handle, mode=read_mode)
    else:
        inner_archive_handle = open_func(archived_file_handle, mode=read_mode)
    # print('inner_archive_handle:', inner_archive_handle)
    # with open_func(fileobj=archived_file_handle, mode='r') as inner_archive_handle:
    #     print('inner_archive_handle:', inner_archive_handle)
    for inner_file_info, file_data in read_func(archived_filename,
                                                inner_archive_handle,
                                                filenames_only=filenames_only):
        inner_file_info['source_file'] = file_info['source_file'] + inner_file_info['source_file']
        yield inner_file_info, file_data
    inner_archive_handle.close()


def read_zip_handle(archive_fname: str, archive_handle: ZipFile,
                    filenames_only: bool = False) -> Generator[Tuple[dict, Union[str, None]], None, None]:
    # archived_filenames = archive_handle.namelist()
    archived_file_infos = archive_handle.infolist()
    for archived_file_info in archived_file_infos:
        archived_filename, is_dir = archived_file_info.filename, archived_file_info.is_dir()
        if is_dir is True:
            continue
        archived_dir, archived_file, archived_file_ext = parse_archived_filename(archived_filename)
        # for archived_fname in archived_filenames:
        # archived_fname_dir, archived_fname_file, archived_fname_ext = parse_archived_filename(archived_fname)
        # file_reader = archive_handle.extract(archived_file_info)
        with archive_handle.open(archived_filename) as fh:
            file_info = {
                'source_file': [archive_fname],
                'archived_filename': archived_file,
                'archived_filepath': archived_filename
            }
            if archived_file_ext in ZIP_EXTENSIONS:
                for inner_file_info, file_data in read_inner_archive(archived_filename,
                                                                     archived_file_ext,
                                                                     fh, file_info,
                                                                     filenames_only=filenames_only):
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


def get_archive_functions(archiver: str):
    if archiver == "py7zr":
        open_func = py7zr.SevenZipFile
        read_func = read_7z_handle
    elif archiver == "tar":
        open_func = tarfile.open
        read_func = read_tar_handle
    elif archiver == "zip":
        open_func = zipfile.ZipFile
        read_func = read_zip_handle
    else:
        raise ValueError(f'unknown archive extension "{archiver}", expected "tar.gz", "tgz", "zip" or "7z"')
    return open_func, read_func


class Extractor:

    def __init__(self, page_archive_file: str, filenames_only: bool = False):
        archiver, read_mode = get_archiver_mode(page_archive_file)
        self.page_archive_file = page_archive_file
        self.filenames_only = filenames_only
        self.archiver = archiver
        self.read_mode = read_mode
        self.open_func, self.handle_func = get_archive_functions(archiver)

    def __iter__(self):
        # print('Extractor - open_func:', self.open_func)
        with self.open_func(self.page_archive_file, mode=self.read_mode) as archive_handle:
            extractor = self.handle_func(self.page_archive_file, archive_handle,
                                         filenames_only=self.filenames_only)
            for file_info, file_data in extractor:
                yield file_info, file_data


def get_archiver_mode(page_archive_file: str) -> \
        Tuple[Literal["tar", "zip", "py7zr"], Literal["r", "r:", "r:gz", "r:bz2"]]:
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
