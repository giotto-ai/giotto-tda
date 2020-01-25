import os

ipynb_files = [
    f for f in os.listdir(".") if os.path.isfile(f) and f.endswith(".ipynb")]

c = get_config()

c.NbConvertApp.notebooks = ipynb_files

c.NbConvertApp.FilesWriter.build_directory = ''
c.NbConvertApp.ClearOutputPreprocessor.enabled = True
c.NbConvertApp.ExecutePreprocessor.enabled = True
c.NbConvertApp.ExecutePreprocessor.export_format = 'notebook'
c.NbConvertApp.ExecutePreprocessor.use_output_suffix = False
c.NbConvertApp.ExecutePreprocessor.startup_timeout = 300
c.NbConvertApp.ExecutePreprocessor.timeout = 600