import paths
import os
#os.environ['COPY_EDIT_DATA']=paths.data_dir
os.environ['CUDA_VISIBLE_DEVICES']='0'
from gtd.utils import Config

from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRuns
print os.environ['COPY_EDIT_DATA']
import sys

#no-profile
profile=False

runs = RetrieveEditTrainingRuns()
config_file = sys.argv[1]
config = Config.from_file('editor_code/configs/editor/'+config_file)
run = runs.new(config)

if profile:
    from gtd.chrono import Profiling, Profiler

    profiler = Profiler.default()

    import editor_code.copy_editor.retriever
    import editor_code.copy_editor.editor
    profiler.add_module(editor_code.copy_editor.editor)
    profiler.add_module(editor_code.copy_editor.retriever)
    Profiling.start()
    run.train()
    Profiler.report(profiler)  # prints out report

else:
    run.train()

