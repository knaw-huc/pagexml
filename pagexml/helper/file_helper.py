import os
import zipfile

import py7zr


ZIP_EXTENSIONS = {'.zip', '.gz', '.bz2', '.7z'}


def parse_zipped_filename(zipped_fname):
    if zipped_fname.count('\\') > 0 and zipped_fname.count('/') == 0:
        dir_separator = '\\'
    elif zipped_fname.count('\\') == 0 and zipped_fname.count('/') > 0:
        dir_separator = '/'
    elif zipped_fname.count('\\') == 0 and zipped_fname.count('/') == 0:
        dir_separator = None
    else:
        dir_separator = os.sep
        zipped_fname = os.path.normpath(zipped_fname)

    if dir_separator == os.sep:
        zipped_fname_dir, zipped_fname_file = os.path.split(zipped_fname)
    elif dir_separator is None:
        zipped_fname_dir = ''
        zipped_fname_file = zipped_fname
    else:
        # directory separator is different from OS default. Use string split
        *zipped_fname_dirs, zipped_fname_file = zipped_fname.split(dir_separator)
        zipped_fname_dir = '\\'.join(zipped_fname_dirs)
    zipped_fname_base, zipped_fname_ext = os.path.splitext(zipped_fname_file)
    return zipped_fname_dir, zipped_fname_file, zipped_fname_ext


def read_zip_handle(zip_fname, zip_handle, filenames_only: bool = False):
    zipped_filenames = zip_handle.namelist()
    for zipped_fname in zipped_filenames:
        zipped_fname_dir, zipped_fname_file, zipped_fname_ext = parse_zipped_filename(zipped_fname)
        with zip_handle.open(zipped_fname) as fh:
            file_info = {
                'zipinfo': [zip_fname],
                'filename': zipped_fname_file,
                'filepath': zipped_fname
            }
            if zipped_fname_ext in ZIP_EXTENSIONS:
                if zipped_fname_ext == '.zip':
                    zip_func = zipfile.ZipFile
                    read_func = read_zip_handle
                elif zipped_fname_ext == '.7z':
                    zip_func = py7zr.SevenZipFile
                    read_func = read_7z_handle
                with zip_func(fh, mode='r') as inner_zip_handle:
                    for inner_file_info, file_data in read_func(zipped_fname,
                                                                inner_zip_handle,
                                                                filenames_only=filenames_only):
                        inner_file_info['zipinfo'] = file_info['zipinfo'] + inner_file_info['zipinfo']
                        yield inner_file_info, file_data
            else:
                file_data = fh.read() if filenames_only is False else None
                yield file_info, file_data


def read_7z_handle(page_7z_file, page_7z_handle, filenames_only: bool = False):
    for zipped_fname, zipped_file_data in page_7z_handle.readall().items():
        zipped_fname_dir, zipped_fname_file, zipped_fname_ext = parse_zipped_filename(zipped_fname)
        file_info = {
            'zipinfo': [page_7z_file],
            'filename': zipped_fname_file,
            'filepath': zipped_fname
        }
        if filenames_only is True:
            yield file_info, None
        else:
            yield file_info, zipped_file_data


def read_page_7z_file(page_7z_file, filenames_only: bool = False):
    with py7zr.SevenZipFile(page_7z_file, mode='r') as zh:
        for file_info, file_data in read_7z_handle(page_7z_file, zh, filenames_only=filenames_only):
            yield file_info, file_data


def read_page_zipfile(page_zipfile: str, filenames_only: bool = False):
    if page_zipfile.endswith('.zip'):
        with zipfile.ZipFile(page_zipfile, 'r') as zh:
            for file_info, file_data in read_zip_handle(page_zipfile, zh, filenames_only=filenames_only):
                yield file_info, file_data
            '''
            filenames = sorted(zh.namelist())
            for fname in filenames:
                with zh.open(fname) as fh:
                    yield fname, fh.read()
            '''
    elif page_zipfile.endswith('.7z'):
        for file_info, file_data in read_page_7z_file(page_zipfile):
            yield file_info, file_data
    else:
        print('unexpected zip format:', page_zipfile)


def read_page_zipfiles(page_zipfiles, filenames_only: bool = False):
    for page_zipfile in page_zipfiles:
        print('extracting PageXML files from zipfile', page_zipfile)
        for page_data in read_page_zipfile(page_zipfile, filenames_only=filenames_only):
            yield page_data
