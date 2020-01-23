import sys


def whl_urls(python_ver, pkg):
    if python_ver == '38':
        python_ver_1 = python_ver
    else:
        python_ver_1 = python_ver + 'm'
    pycairo_whl_url = \
        'https://storage.googleapis.com/l2f-open-models/giotto' \
        '-learn/windows-binaries/pycairo/pycairo-1.18.2-cp{}' \
        '-cp{}-win_amd64.whl'.format(python_ver, python_ver_1)
    igraph_whl_url = \
        'https://storage.googleapis.com/l2f-open-models/giotto' \
        '-learn/windows-binaries/python-igraph/python_igraph-' \
        '0.7.1.post6-cp{}-cp{}-win_amd64.whl'.\
        format(python_ver, python_ver_1)
    if pkg == 'pycairo':
        return pycairo_whl_url
    elif pkg == 'python-igraph':
        return igraph_whl_url
    else:
        raise ValueError("Second argument must be either 'pycairo' or "
                         "python-igraph")


if __name__ == '__main__':
    whl_urls(sys.argv[1], sys.argv[2])
