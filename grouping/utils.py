import shutil
import matplotlib as mpl
import os
import matplotlib.pyplot as plt


def save_path(dirpath, ext, _name=None, _order=[], **kwargs):
    os.makedirs(dirpath, exist_ok=True)
    keys = sorted(list(kwargs.keys()))
    if not set(_order).issubset(keys):
        raise ValueError(f"Given order {_order} should be a subset of {keys}.")

    for key in _order:
        keys.remove(key)

    keys = _order + keys

    def replace(x):
        if x is True:
            return "T"
        if x is False:
            return "F"
        if x is None:
            return "N"
        return x

    filename = ":".join(f"{k}={replace(kwargs[k])}" for k in keys)

    if _name is not None:
        filename = f"{_name}__{filename}" if filename else _name

    if not filename:
        if ext == "csv":
            filename = "table"
        else:
            filename = "fig"

    filename = filename.replace("(", ":")
    filename = filename.replace(")", "")
    filename = filename.replace(" ", "_")
    filename = filename.replace(",", ":")
    # filename = filename.replace(':', '_')
    filename = filename.replace("@", "_")
    # filename = filename.replace('=', '_')
    filename = filename.replace(".", "_")
    filename = filename.replace("/", "_")

    filename += f".{ext}"
    filepath = os.path.join(dirpath, filename)

    return filepath


def save_fig(
    fig,
    dirpath,
    _name=None,
    ext="pdf",
    _order=[],
    pad_inches=0.1,
    bbox_inches="tight",
    _add_default_fig=True,
    **kwargs,
):
    filepaths = [save_path(dirpath, ext=ext, _name=_name, _order=_order, **kwargs)]

    if _add_default_fig:
        filepaths.append(save_path(dirpath, ext=ext))

    for filepath in filepaths:
        fig.savefig(
            filepath, bbox_inches=bbox_inches, transparent=True, pad_inches=pad_inches
        )
    return filepath


def checkdep_usetex(s):
    """From matplotlib"""
    if not s:
        return False
    if not shutil.which("tex"):
        # _log.warning("usetex mode requires TeX.")
        return False
    try:
        mpl._get_executable_info("dvipng")
    except mpl.ExecutableNotFoundError:
        # _log.warning("usetex mode requires dvipng.")
        return False
    try:
        mpl._get_executable_info("gs")
    except mpl.ExecutableNotFoundError:
        # _log.warning("usetex mode requires ghostscript.")
        return False
    return True


def set_latex_font(math=True, normal=True, extra_preamble=[]):
    if math:
        plt.rcParams["mathtext.fontset"] = "stix"
    else:
        plt.rcParams["mathtext.fontset"] = plt.rcParamsDefault["mathtext.fontset"]

    if normal:
        plt.rcParams["font.family"] = "STIXGeneral"
        # plt.rcParams['text.usetex'] = True
    else:
        plt.rcParams["font.family"] = plt.rcParamsDefault["font.family"]

    # if math or normal:
    #     plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}',
    # [
    #     r'\usepackage{amsfonts}',
    #     # r'\usepackage{amsmath}',
    # ]
    # matplotlib.rc('text',usetex=True)
    # matplotlib.rc('text.latex', preamble=r'\usepackage{color}')

    # plt.rcParams['text.usetex'] = True

    usetex = checkdep_usetex(True)
    # print(f'usetex: {usetex}')
    plt.rc("text", usetex=usetex)
    default_preamble = [
        r"\usepackage{amsfonts}",
        r"\usepackage{amsmath}",
    ]
    preamble = "".join(default_preamble + extra_preamble)
    plt.rc("text.latex", preamble=preamble)
    # mpl.verbose.level = 'debug-annoying'
