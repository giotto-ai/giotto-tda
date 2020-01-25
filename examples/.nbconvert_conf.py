import os

c = get_config()
ipynb_files = [
    f for f in os.listdir(".") if os.path.isfile(f) and f.endswith(".ipynb")]

c.NbConvertApp.notebooks = ipynb_files
c.NbConvertApp.execute = True
c.NbConvertApp.export_format = 'notebook'
c.NbConvertApp.ExecutePreprocessor.execute = True
c.NbConvertApp.ExecutePreprocessor.startup_timeout = 300
c.NbConvertApp.ExecutePreprocessor.timeout = 600