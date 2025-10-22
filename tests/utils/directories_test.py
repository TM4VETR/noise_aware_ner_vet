from utils.directories import get_data_dir


def test_get_data_dir_env(monkeypatch):
    """ Tests get_data_dir() using the environment variable DATA_DIR_VET """
    env_path = "/some/env/path"
    monkeypatch.setenv("DATA_DIR_VET", env_path)

    result = get_data_dir()
    assert result == env_path
