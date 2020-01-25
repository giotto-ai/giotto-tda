import os

c = get_config()
ipynb_files = [
    f for f in os.listdir(".") if os.path.isfile(f) and f.endswith(".ipynb")]
c.NbConvertApp.notebooks = ipynb_files