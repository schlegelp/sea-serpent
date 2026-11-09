"""Test set-up.

The tests talk to a real SeaTable server. Settings come from environment
variables, or from a file of NAME=value lines named by ``SEASERPENT_TEST_ENV``:

    SEATABLE_SERVER          server address, e.g. http://127.0.0.1:8020
    SEATABLE_TEST_TOKEN      read+write base token for a THROWAWAY base
                             (write tests; skipped if not set)
    SEATABLE_TEST_BASE_UUID  that base's UUID
    SEATABLE_READ_TOKEN      read-only base token (read tests; falls back to
    SEATABLE_READ_TABLE      SEATABLE_TEST_TOKEN and a table it creates)

Write tests only create, change and delete tables whose names start with
``sstest_``, and refuse to run against the hosts in ``NO_WRITE_HOSTS``.

Example:

    SEASERPENT_TEST_ENV=~/.config/sea-serpent/sandbox-test.env pytest tests/
"""
import os
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest

#: Never write to these (production servers).
NO_WRITE_HOSTS = {"flytable.mrc-lmb.cam.ac.uk"}

TABLE_PREFIX = "sstest_"


def _load_env_file():
    path = os.environ.get("SEASERPENT_TEST_ENV")
    if not path:
        return
    for line in Path(path).expanduser().read_text().splitlines():
        m = re.fullmatch(r"([A-Z_][A-Z0-9_]*)=(\S+)", line.strip())
        if m:
            # The file wins over the shell, so a run always goes where the
            # file says (users often have SEATABLE_SERVER set to production).
            os.environ[m.group(1)] = m.group(2)


_load_env_file()


@pytest.fixture(scope="session")
def server():
    s = os.environ.get("SEATABLE_SERVER")
    if not s:
        pytest.skip("SEATABLE_SERVER not set")
    return s.rstrip("/")


@pytest.fixture(scope="session")
def rw_token(server):
    token = os.environ.get("SEATABLE_TEST_TOKEN")
    if not token:
        pytest.skip("SEATABLE_TEST_TOKEN not set")
    if urlparse(server).hostname in NO_WRITE_HOSTS:
        pytest.skip(f"write tests never run against {server}")
    return token


@pytest.fixture(scope="session")
def base_uuid(rw_token):
    uuid_ = os.environ.get("SEATABLE_TEST_BASE_UUID")
    if not uuid_:
        pytest.skip("SEATABLE_TEST_BASE_UUID not set")
    return uuid_


def new_table_name():
    return f"{TABLE_PREFIX}{uuid.uuid4().hex[:8]}"


def create_table(df, server, token, base):
    """Scratch table from a DataFrame, without progress bars."""
    import seaserpent as ss

    t = ss.Table.from_frame(
        df, table_name=new_table_name(), base=base, auth_token=token, server=server
    )
    t.progress = False
    return t


@pytest.fixture
def make_table(server, rw_token, base_uuid):
    """Factory for scratch tables: created from a DataFrame, deleted at the end."""
    made = []

    def make(df):
        made.append(create_table(df, server, rw_token, base_uuid))
        return made[-1]

    yield make

    for t in made:
        try:
            t.delete(skip_confirmation=True)
        except Exception as e:  # pragma: no cover - clean-up only
            print(f"could not delete {t.name}: {e}")


@pytest.fixture(scope="session")
def read_table(server, request):
    """A table to read from: SEATABLE_READ_TOKEN/TABLE, else a scratch table."""
    import pandas as pd
    import seaserpent as ss

    token = os.environ.get("SEATABLE_READ_TOKEN")
    name = os.environ.get("SEATABLE_READ_TABLE")
    if token and name:
        yield ss.Table(name, base_token=token, server=server, progress=False)
        return

    # Fall back to a scratch table (same skip rules as the write tests)
    rw = request.getfixturevalue("rw_token")
    base = request.getfixturevalue("base_uuid")
    df = pd.DataFrame({"id": [f"r{i}" for i in range(25)], "n": list(range(25))})
    t = create_table(df, server, rw, base)
    yield t
    t.delete(skip_confirmation=True)
