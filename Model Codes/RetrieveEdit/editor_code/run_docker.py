#!/u/nlp/packages/anaconda2/bin/python

"""
Make sure to set the environment variable ENTAIL_GEN_DATA to point at your
data directory.

To run an interactive docker container as root and use GPU #2:
$ python run_docker.py -r -g 2

To run a command inside the docker container using GPU #3:
$ python run_docker.py -g 3 'echo hello'

The argument `-g 3` is equivalent to setting CUDA_VISIBLE_DEVICES=3 inside the
container.
"""

import argparse
import json
import os

from os.path import dirname, abspath, join
import subprocess

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-r', '--root', action='store_true', help='Run as root in Docker.')
arg_parser.add_argument('-g', '--gpu', default='', help='GPU to use.')
arg_parser.add_argument('-d', '--debug', action='store_true', help='Print command instead of running.')
arg_parser.add_argument('command', nargs='?', default=None,
                        help='Command to execute in Docker. If no command is specified, ' \
                             'you enter interactive mode. ' \
                             'To execute a command with spaces, wrap ' \
                             'the entire command in quotes.')
args = arg_parser.parse_args()

repo_dir = abspath(dirname(__file__))

image = 'kelvinguu/entail-gen:1.2'
data_env_var = 'ENTAIL_GEN_DATA'
data_dir = os.environ[data_env_var]

my_uid = subprocess.check_output(['echo', '$UID']).strip()

docker_args = ["--net host",  # access to the Internet
               "--publish 8888:8888",  # only certain ports are exposed
               "--publish 6006:6006",
               "--publish 8080:8080",
               "--ipc=host",
               "--rm",
               "--volume {}:/data".format(data_dir),
               "--volume {}:/code".format(repo_dir),
               "--env {}=/data".format(data_env_var),
               "--env PYTHONPATH=/code",
               "--env NLTK_DATA=/data/nltk",
               "--env CUDA_VISIBLE_DEVICES={}".format(args.gpu),
               "--workdir /code"]

# interactive mode
if args.command is None:
    docker_args.append('--interactive')
    docker_args.append('--tty')
    args.command = '/bin/bash'

if not args.root:
    docker_args.append('--user={}'.format(my_uid))

if args.gpu == '':
    # run on CPU
    docker = 'docker'
else:
    # run on GPU
    docker = 'nvidia-docker'

pull_cmd = "docker pull {}".format(image)

run_cmd = '{docker} run {options} {image} {command}'.format(docker=docker,
                                                            options=' '.join(docker_args),
                                                            image=image,
                                                            command=args.command)
print 'Data directory: {}'.format(data_dir)
print 'Command to run inside Docker: {}'.format(args.command)

print pull_cmd
print run_cmd
if not args.debug:
    subprocess.call(pull_cmd, shell=True)
    subprocess.call(run_cmd, shell=True)
