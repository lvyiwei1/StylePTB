import paths
import os
#os.environ['COPY_EDIT_DATA']=paths.data_dir
os.environ['CUDA_VISIBLE_DEVICES']='0'
from gtd.utils import Config

from editor_code.copy_editor.edit_training_run import EditTrainingRuns
print os.environ['COPY_EDIT_DATA']

#no-profile
profile=False

runs = EditTrainingRuns()
config = Config.from_file('editor_code/configs/editor/github_s2s.txt')
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


#runs = EditTrainingRuns()
#config = Config.from_file('editor_code/configs/editor/default.txt')
#run = runs.new(config)
#run.train()

