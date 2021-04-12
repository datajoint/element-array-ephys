import datajoint as dj
import pathlib


dj.config['enable_python_native_blobs'] = True


def find_valid_full_path(potential_root_directories, path):
    """
    Given multiple potential root directories and a single path
    Search and return one directory that is the parent of the given path
        :param potential_root_directories: potential root directories
        :param path: the path to search the root directory
        :return: (fullpath, root_directory)
    """
    path = pathlib.Path(path)

    # turn to list if only a single root directory is provided
    if isinstance(potential_root_directories, (str, pathlib.Path)):
        potential_root_directories = [potential_root_directories]

    # search routine
    for root_dir in potential_root_directories:
        root_dir = pathlib.Path(root_dir)
        if path.exists():
            if root_dir in list(path.parents):
                return path, root_dir
        else:
            if (root_dir / path).exists():
                return root_dir / path, root_dir

    raise FileNotFoundError('Unable to identify root-directory (from {})'
                            ' associated with {}'.format(potential_root_directories, path))
