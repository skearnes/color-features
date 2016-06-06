"""
Script utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from IPython.parallel import Client


def get_map(cluster_id=None):
    """
    Get the proper mapping function.

    Parameters
    ----------
    cluster_id : str, optional
        IPython.parallel cluster ID.
    """
    if cluster_id is not None:
        client = Client(cluster_id=cluster_id)
        client.direct_view().use_dill()
        view = client.load_balanced_view()
        return view.map_sync
    else:
        return map
