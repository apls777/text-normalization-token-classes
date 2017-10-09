import json
import os
from shutil import copyfile
import utils
import logging


def get_session_dir(latest_session: bool = False, session_id: int = 0):
    training_dir = os.path.join('..', 'training')
    latest_session_id = get_latest_session_id(training_dir)

    if latest_session:
        session_id = latest_session_id

    if not session_id:
        session_id = latest_session_id + 1

    # session directory
    session_dir = os.path.join(training_dir, 'session_%d' % session_id)

    return session_dir


def read_config(session_dir: str) -> dict:
    logger = logging.getLogger(__name__)
    config_file = os.path.join(session_dir, 'config.json')

    config = {}
    if os.path.isfile(config_file):
        # read config file
        with open(config_file) as f:
            config = json.load(f)
            logger.info('Config file is read: "%s"' % config_file)

    return config


def write_config(session_dir, config: dict):
    logger = logging.getLogger(__name__)
    logger.debug('Writing a config file')

    # copy required files to the session directory
    files_dir = os.path.join(session_dir, 'files')
    utils.check_path(files_dir)
    for key, filename in config['files'].items():
        new_filename = os.path.join(files_dir, os.path.basename(filename))
        copyfile(filename, new_filename)
        config['files'][key] = new_filename
        logger.debug('File "%s" was copied' % filename)

    # create new config file
    config_file = os.path.join(session_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
        logger.info('Config file is written: "%s"' % config_file)


def get_latest_session_id(training_dir):
    """
    Get an ID of the last saved configuration
    """
    dirs = sorted(os.listdir(training_dir), reverse=True)
    latest_session_id = int(dirs[0].split('_')[1]) if dirs else 0

    return latest_session_id
