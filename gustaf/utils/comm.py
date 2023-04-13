"""
Communication utilities.
"""
has_websockets = False
try:
    # this requires python 3.7+
    from websockets.sync.client import connect
    has_websockets = True
except ImportError as err:
    from gustaf.helpers.raise_if import ModuleImportRaiser

    connect = ModuleImportRaiser("websockets", err)

from gustaf._base import GustafBase


class WebSocketClient(GustafBase):
    """
    Minimal websocket client using `websockets`'s thread
    based implementation.
    """

    def __init__(self, uri, close_timeout=60):
        """
        Attributes
        ----------
        uri: str
        close_timeout: int

        Parameters
        ----------
        uri: str
        close_timeout: int  
        """
        # save config in case we need to re connect
        self.uri = uri
        self.close_timeout = close_timeout
        self.websocket = connect(uri, close_timeout=close_timeout)

    def send_recv(self, message):
        self.websocket.send(message)
        return self.websocket.recv()

    def __del__(self):
        self.websocket.close()
