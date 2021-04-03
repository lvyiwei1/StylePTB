import os
from gtd.io import Workspace


# Set location of local data directory from environment variable
env_var = 'COPY_EDIT_DATA'
if env_var not in os.environ:
    assert False, env_var + ' environmental variable must be set.'
root = os.environ[env_var]

# define workspace
workspace = Workspace(root)

# Training runs
workspace.add_dir('sc_runs')
workspace.add_dir('edit_runs')

# word vectors
workspace.add_dir('word_vectors')

# dataset dir
workspace.add_dir('datasets')
