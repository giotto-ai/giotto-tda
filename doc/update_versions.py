
import re
import io
from functools import reduce
from operator import add

version_entry_generator = lambda url, slug: f"""<dd><a href="{url}/library.html">{slug}</a></dd>"""

pattern_get_root = """<!--path_to_root:(.*?)-->"""
start_tag = """<!--start_versions_tag-->"""
end_tag = """<!--end_versions_tag-->"""


def url_root_from_line(l):
    """Extract relative path from line"""
    extracted = re.search(pattern_get_root, l)
    return extracted[1]


def _process_file(s, versions):
    """s is a string representation of a file."""
    new_lines = []
    do_add = True
    lines = s.split("\n")
    for ind, line in enumerate(lines):
        if line.strip() == start_tag:
            new_lines.append(line)
            do_add = False
            url_root = url_root_from_line(lines[ind+1])
            new_lines.append(lines[ind+1])
            versions_ = [url_root + '../' + v for v in versions]
            new_lines.append(reduce(add,
                                    map(version_entry_generator,
                                        versions_, versions)))
        elif do_add:
            new_lines.append(line)
        elif line.strip() == end_tag:
            new_lines.append(line)
            do_add = True
        else:
            continue
    return "\n".join(new_lines)


def process_file(file_name, versions):
    with open(file_name, 'r') as f:
        s = f.read()
    new_lines = _process_file(s, versions)
    with open(file_name, 'w') as f:
        f.write(new_lines)
    return 0


if __name__ == '__main__':
    import sys
    import os
    from glob import glob
    path = sys.argv[1]
    file_names = [y for x in os.walk(path)
                  for y in glob(os.path.join(x[0], '*.html'))]
    print(file_names)

    with open('versions', 'r') as f:
        versions = [c[2:].rstrip() for c in f.readlines()]
        versions = list(filter(lambda c: not(c.startswith('.')), versions))
    print(versions)

    for file_name in file_names:
        process_file(file_name, versions)


