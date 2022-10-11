import logging
import requests
import json
import urllib

import pandas as pd

from .utils import get_account, find_base
from .base import Table


logger = logging.getLogger(__name__)
logger.setLevel('INFO')


def list_users(server=None, auth_token=None):
    """List all users.

    Parameters
    ----------
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        provided explicitly or be set as ``SEATABLE_TOKEN``
                        environment variable. User must be admin!
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    pd.DataFrame

    """
    account = get_account(server=server, auth_token=auth_token)

    url = f'{account.server_url}/api/v2.1/admin/users/?page=1&per_page=1000'

    r = requests.get(url, headers=account.token_headers)

    r.raise_for_status()

    return pd.DataFrame.from_records(r.json()['data'])


def find_user(user, server=None, auth_token=None):
    """Find given user(s).

    Parameters
    ----------
    user :              str
                        Search term (name, email, etc).
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        provided explicitly or be set as ``SEATABLE_TOKEN``
                        environment variable. User must be admin!
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    pd.DataFrame

    """
    account = get_account(server=server, auth_token=auth_token)

    url = f'{account.server_url}/api/v2.1/admin/search-user/?query={user}'

    r = requests.get(url, headers=account.token_headers)

    r.raise_for_status()

    return pd.DataFrame.from_records(r.json()['user_list'])


def share_base(base, users, permission='r', table=None, server=None, auth_token=None):
    """Share base (or view) to given users.

    Parameters
    ----------
    base :              str
                        Which base to share.
    users :             str | list thereof
                        The unique identifier (something like '123[..]ad32@auth.local')
                        for each user.
    permission :        'r' | 'rw'
                        Whether to allow read-only (default) or read-write.
    table :             str, optional
                        Which table to share. By default this will share the
                        "Default View" but you can specify by using e.g.
                        `table="MyBase:MyView". This only works in the
                        Enterprise version!
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        provided explicitly or be set as ``SEATABLE_TOKEN``
                        environment variable.
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    pd.DataFrame

    """
    assert permission in ('r', 'rw')

    account = get_account(server=server, auth_token=auth_token)

    if not table:
        workspace_id, base_name, _, _ = find_base(base=base, auth_token=auth_token, server=server)
        url = f'{account.server_url}/api/v2.1/workspace/{workspace_id}/dtable/{base_name}/share/'
    else:
        if ':' in table:
            table, view = table.split(':')
        else:
            view = 'Default View'
        table = Table(table, base=base, server=server, auth_token=auth_token)
        if view not in table.views:
            raise ValueError(f'View "{view}" does not exist in "{table.name}"')

        url = (f'{account.server_url}/api/v2.1/workspace/{table.workspace_id}'
               f'/dtable/{table.base_name}/user-view-shares/')

    if isinstance(users, str):
        users = [users]

    for email in users:
        assert email.endswith('@auth.local')

        data = dict(permission=permission)

        if table:
            data['to_user'] = email
            data['table_id'] = table.id
            data['view_id'] = [v['_id'] for v in table.meta['views'] if v['name'] == view][0]
        else:
            data['email'] = email

        r = requests.post(url,
                          headers=account.token_headers,
                          data=data)

        if 'already shared to' in r.content.decode():
            print(f'Base {base_name} already shared with user {email}')
        else:
            r.raise_for_status()
            print(f'Base {base_name} successfully shared with user {email}')


def list_common_datasets(by_group=False, server=None, auth_token=None):
    """List common dataset accessible by the user.

    Parameters
    ----------
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        provided explicitly or be set as ``SEATABLE_TOKEN``
                        environment variable. User must be admin!
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    pd.DataFrame

    """
    account = get_account(server=server, auth_token=auth_token)

    url = f'{account.server_url}/api/v2.1/dtable/common-datasets/'

    r = requests.get(url,
                     headers=account.token_headers,
                     params=dict(by_group=by_group))

    r.raise_for_status()

    return pd.DataFrame.from_records(r.json()['dataset_list'])


def import_common_datasets(dataset, target_base, server=None, auth_token=None):
    """Import common dataset to base.

    To import a common dataset into a base, the following conditions have to be met:

     - The destination base is in a group, and
     - You are the admin or owner of this group, and
     - This group has access to the common dataset.

    Parameters
    ----------
    dataset :           str | int
                        Name (str) or ID (int) of the common dataset to import.
    target_base :       str
                        Base to import dataset to.
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        provided explicitly or be set as ``SEATABLE_TOKEN``
                        environment variable. User must be admin!
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    dict

    """
    account = get_account(server=server, auth_token=auth_token)

    if not isinstance(dataset, int):
        available_ds = list_common_datasets(server=server, auth_token=auth_token)
        available_ds = available_ds[available_ds.dataset_name == dataset]
        if available_ds.empty:
            raise ValueError(f'No common dataset with name "{dataset}" found.')
        elif available_ds.shape[0] > 1:
            raise ValueError(f'Multiple common datasets with name "{dataset}" found.')
        dataset_id = available_ds.id.values[0]
    else:
        dataset_id = dataset

    workspace_id, base_name, _, _ = find_base(base=target_base, auth_token=auth_token, server=server)
    base = account.get_base(workspace_id, base_name)

    url = f'{account.server_url}/api/v2.1/dtable/common-datasets/{dataset_id}/import/'

    r = requests.post(url,
                     headers=account.token_headers,
                     data=dict(dst_dtable_uuid=str(base.dtable_uuid)))

    r.raise_for_status()

    return r.json()


def list_script_tasks(server=None, auth_token=None):
    """List script tasks currently running.

    Parameters
    ----------
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        provided explicitly or be set as ``SEATABLE_TOKEN``
                        environment variable. User must be admin!
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    dict

    """
    account = get_account(server=server, auth_token=auth_token)

    url = f'{account.server_url}/api/v2.1/admin/scripts-tasks/'

    r = requests.get(url, headers=account.token_headers)

    r.raise_for_status()

    return r.json()
