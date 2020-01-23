import sys


def pycairo_igraph_whl_urls(python_ver):
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
    return pycairo_whl_url + ' ' + igraph_whl_url


if __name__ == '__main__':
    pycairo_igraph_whl_urls(sys.argv[1])
