import json
import random
import logging
import numpy as np
from pathlib import Path
from tempfile import gettempdir
from collections import OrderedDict

__all__ = [
    'ANSI', 'my_stream_handler', 'query_yes_no', 'increment_dir', 'random_string',
    'get_temp_file_path', 'read_file', 'json_load', 'json_dump', 'print_to_file',
    'SimpleTable', 'print_dict_as_table', 'MaxLengthList',
]


def docstring_example():
    """ A dummy function to show the docstring format that can be parsed by Pylance. \\
    Hyperlink https://github.com \\
    Hyperlink with text [GitHub](https://github.com) \\
    Code `mycv.utils.general`

    Args:
        xxxx (type): xxxx xxxx xxxx xxxx.
        xxxx (type, optional): xxxx xxxx xxxx xxxx. Defaults to 'xxxx'.

    ### Bullets:
        - xxxx
        - xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx \
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx \
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx

    ### Code examples:
        >>> torch.load('tensors.pt')
        # Load all tensors onto the CPU
        >>> with open('tensor.pt', 'rb') as f:
        ...     buffer = io.BytesIO(f.read())

    ### Code examples::

        # comment
        model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )
    """
    return 0


class ANSI():
    """ ANSI escape codes with colorizing functions

    Reference:
    - https://en.wikipedia.org/wiki/ANSI_escape_code
    - https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
    """
    # basic colors
    black   = '\u001b[30m'
    red     = r = '\u001b[31m'
    green   = g = '\u001b[32m'
    yellow  = y = '\u001b[33m'
    blue    = b = '\u001b[34m'
    magenta = m = '\u001b[35m'
    cyan    = c = '\u001b[36m'
    white   = w = '\u001b[37m'
    # bright colors
    bright_black   = '\u001b[90m'
    bright_red     = br_r = '\u001b[91m'
    bright_green   = br_g = '\u001b[92m'
    bright_yellow  = br_y = '\u001b[93m'
    bright_blue    = br_b = '\u001b[94m'
    bright_magenta = br_m = '\u001b[95m'
    bright_cyan    = br_c = '\u001b[96m'
    bright_white   = br_w = '\u001b[97m'
    # background colors
    background_black   = '\u001b[40m'
    background_red     = bg_r = '\u001b[41m'
    background_green   = bg_g = '\u001b[42m'
    background_yellow  = bg_y = '\u001b[43m'
    background_blue    = bg_b = '\u001b[44m'
    background_magenta = bg_m = '\u001b[45m'
    background_cyan    = bg_c = '\u001b[46m'
    background_white   = bg_w = '\u001b[47m'
    # misc
    end       = '\u001b[0m'
    bold      = '\u001b[1m'
    underline = udl = '\u001b[4m'
    all_colors_short = [
        'black',               'r',    'g',    'y',    'b',    'm',    'c',    'w',
        'bright_black',     'br_r', 'br_g', 'br_y', 'br_b', 'br_m', 'br_c', 'br_w',
        'background_black', 'bg_r', 'bg_g', 'bg_y', 'bg_b', 'bg_m', 'bg_c', 'bg_w',
    ]
    all_colors_long = [
        'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
        'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
        'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white',
        'background_black', 'background_red', 'background_green', 'background_yellow',
        'background_blue', 'background_magenta', 'background_cyan', 'background_white'
    ]

    @classmethod
    def colorstr_example(cls):
        for c in cls.all_colors_long:
            line = ''.join([
                cls.colorstr(f'{c}',                c=c, b=False, ul=False), ', ',
                cls.colorstr(f'{c} bold',           c=c, b=True,  ul=False), ', ',
                cls.colorstr(f'{c} underline',      c=c, b=False, ul=True), ', ',
                cls.colorstr(f'{c} bold underline', c=c, b=True,  ul=True),
            ])
            print(line)

    @classmethod
    def colorstr(cls, msg: str, c='b', b=False, ul=False):
        """ Colorize a string. 

        Args:
            msg (str): string
            c (str): color. Examples: 'red', 'r', 'br_r', ...
            b (bool): bold
            ul (bool): underline
        """
        msg = str(msg)
        if c is not None:
            # msg = eval(f'cls.{c}') + msg
            msg = getattr(cls, c) + msg
        if b:
            msg = cls.bold + msg
        if ul:
            msg = cls.underline + msg
        msg = msg + cls.end
        return msg

    @classmethod
    def printc(cls, *strings, c='blue', b=False, ul=False, **kwargs):
        """ Print with color and style

        Args:
            msg (str): string
            c (str): color. Examples: 'red', 'r', 'br_r', ...
            b (bool): bold
            ul (bool): underline
        """
        strings = [cls.colorstr(s, c, b, ul) for s in strings]
        print(*strings, **kwargs)

    @classmethod
    def errorstr(cls, msg: str):
        msg = cls.bright_red + str(msg) + cls.end
        return msg

    @classmethod
    def warningstr(cls, msg: str):
        msg = cls.yellow + str(msg) + cls.end
        return msg

    @classmethod
    def infostr(cls, msg: str):
        msg = cls.bright_blue + str(msg) + cls.end
        return msg

    @classmethod
    def successstr(cls, msg: str):
        msg = cls.bright_green + str(msg) + cls.end
        return msg
    sccstr = successstr

    @classmethod
    def titlestr(cls, msg: str):
        msg = cls.bold + str(msg) + cls.end
        return msg

    @classmethod
    def headerstr(cls, msg: str):
        msg = cls.underline + str(msg) + cls.end
        return msg

    @classmethod
    def highlightstr(cls, msg: str):
        msg = cls.cyan + str(msg) + cls.end
        return msg
    hlstr = highlightstr

    @classmethod
    def underlinestr(cls, msg: str):
        msg = cls.underline + str(msg) + cls.end
        return msg
    udlstr = underlinestr


def colorstr_example():
    ANSI.colorstr_example()


class LevelFormatter(logging.Formatter):
    """ Formatter for logging that uses different colors for different levels.
    """
    _level_formats = {
        logging.WARNING: ANSI.warningstr('[%(asctime)s] %(message)s'),
        logging.ERROR:   ANSI.errorstr('[%(asctime)s] %(message)s'),
    }

    def format(self, record):
        # adapted from https://stackoverflow.com/q/14844970
        # Save the default format configured by the user
        format_default = self._style._fmt
        # Replace the original format with one customized by logging level
        self._style._fmt = self._level_formats.get(record.levelno, format_default)
        # Call the original format method
        result = super().format(record)
        # Restore the original format configured by the user
        self._style._fmt = format_default
        return result

def my_stream_handler():
    """ Create a stream handler with a custom formatter for logging.

    Returns:
        logging.StreamHandler: a stream handler
    """
    handler = logging.StreamHandler()
    formatter = LevelFormatter(fmt='[%(asctime)s] %(message)s', datefmt='%Y-%b-%d %H:%M:%S')
    handler.setFormatter(formatter)
    return handler


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. \\
    The return value is True for 'y' or 'yes', and False for 'n' or 'no'.

    Args:
        question (str): a string that is presented to the user.

    Returns:
        bool: True for 'y' or 'yes', and False for 'n' or 'no'.
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}

    while True:
        print(question + " [y/n]: ", end='')
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print("Please respond with yes/no or y/n.")


def increment_dir(dir_root='runs/', name='exp'):
    """ Get increamental directory name. E.g., exp_1, exp_2, exp_3, ...

    Args:
        dir_root (str, optional): root directory. Defaults to 'runs/'.
        name (str, optional): dir prefix. Defaults to 'exp'.

    Returns:
        str: directory name
    """
    assert isinstance(dir_root, (str, Path))
    dir_root = Path(dir_root)
    n = 0
    while (dir_root / f'{name}_{n}').is_dir():
        n += 1
    name = f'{name}_{n}'
    return name


def random_string(length: int):
    """ Generate a random string of given length.

    Args:
        length (int): length of the string.

    Returns:
        str: a random string
    """
    dictionary = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choices(dictionary, k=length))


def get_temp_file_path(suffix='.tmp'):
    """ Get a temporary file path.

    Args:
        suffix (str, optional): suffix of the file. Defaults to '.tmp'.

    Returns:
        Path: a temporary file path
    """
    tmp_path = Path(gettempdir()) / (random_string(16) + suffix)
    if tmp_path.is_file():
        print(f'{tmp_path} already exists!! Generating another one...')
        tmp_path = Path(gettempdir()) / (random_string(16) + suffix)
    return tmp_path


def read_file(fpath):
    with open(fpath, mode='r') as f:
        s = f.read()
    return s

def print_to_file(msg, fpath, mode='a'):
    with open(fpath, mode=mode) as f:
        print(msg, file=f)

def json_load(fpath):
    with open(fpath, mode='r') as f:
        d = json.load(fp=f)
    return d

def json_dump(obj, fpath, indent=2):
    with open(fpath, mode='w') as f:
        json.dump(obj, fp=f, indent=indent)


class SimpleTable(OrderedDict):
    """ A simple class for creating a table with a header and a body."""
    def __init__(self, init_keys=[]):
        super().__init__()
        # initialization: assign None to initial keys
        for key in init_keys:
            if not isinstance(key, str):
                ANSI.warningstr(f'Progress bar logger key: {key} is not a string')
            self[key] = None
        self._str_lengths = {k: 8 for k,v in self.items()}

    def _update_length(self, key, length):
        old = self._str_lengths.get(key, 0)
        if length <= old:
            return old
        else:
            self._str_lengths[key] = length
            return length

    def update(self, border=False):
        """ Update the string lengths, and return header and body

        Returns:
            str: table header
            str: table body
        """
        header = []
        body = []
        for k,v in self.items():
            # convert any object to string
            key = self.obj_to_str(k)
            val = self.obj_to_str(v)
            # get str length
            str_len = max(len(key), len(val)) + 2
            str_len = self._update_length(k, str_len)
            # make header and body string
            keystr = f'{key:^{str_len}}|'
            valstr = f'{val:^{str_len}}|'
            header.append(keystr)
            body.append(valstr)
        header = ''.join(header)
        if border:
            header = ANSI.headerstr(header)
        body = ''.join(body)
        return header, body

    def get_header(self, border=False):
        header = []
        body = []
        for k in self.keys():
            key = self.obj_to_str(k)
            str_len = self._str_lengths[k]
            keystr = f'{key:^{str_len}}|'
            header.append(keystr)
        header = ''.join(header)
        if border:
            header = ANSI.headerstr(header)
        return header

    def get_body(self):
        body = []
        for k,v in self.items():
            val = self.obj_to_str(v)
            str_len = self._str_lengths[k]
            valstr = f'{val:^{str_len}}|'
            body.append(valstr)
        body = ''.join(body)
        return body

    @staticmethod
    def obj_to_str(obj, digits=4):
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, float) or hasattr(obj, 'float'):
            obj = float(obj)
            return f'{obj:.{digits}g}'
        elif isinstance(obj, list):
            strings = [SimpleTable.obj_to_str(item, 3) for item in obj]
            return '[' + ', '.join(strings) + ']'
        elif isinstance(obj, tuple):
            strings = [SimpleTable.obj_to_str(item, 3) for item in obj]
            return '(' + ', '.join(strings) + ')'
        else:
            return str(obj)


def print_dict_as_table(dictionary: dict):
    """ Print a dictionary as a table

    Args:
        dictionary (dict[str -> values]): a dictionary
    """
    table = SimpleTable()
    keys = list(dictionary.keys())
    keys.sort()
    for k in keys:
        table[k] = dictionary[k]
    header, body = table.update()
    print(header)
    print(body)


class MaxLengthList():
    def __init__(self, max_len, dtype=np.float32):
        self._list = np.empty(0, dtype=dtype)
        self._max_len = int(max_len)
        self._next_idx = int(0)

    def add(self, v):
        v = float(v)
        _len = len(self._list)
        if _len < self._max_len:
            self._list = np.append(self._list, v)
        else:
            assert _len == self._max_len, f'invalid length={_len}, max_len={self._max_len}'
            self._list[self._next_idx] = v
            self._next_idx = (self._next_idx + 1) % self._max_len

    def current(self) -> float:
        if len(self._list) == 0:
            print(f'Warning: the length of self._list={self._list} is 0')
            return None
        return self._list[self._next_idx - 1]

    def median(self) -> float:
        return np.median(self._list)

    def max(self) -> float:
        return np.max(self._list)
