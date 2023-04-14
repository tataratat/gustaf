"""
Communication utilities.
"""
try:
    # this requires python 3.7+
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    has_websockets = True

except ImportError as err:
    from gustaf.helpers.raise_if import ModuleImportRaiser

    connect = ModuleImportRaiser("websockets", err)
    has_websockets = False

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
        """
        Send message and return received answer.

        Parameters
        ----------
        message: str

        Returns
        -------
        response: str
        """
        try:
            self.websocket.ping()
        except (ConnectionClosed, RuntimeError):
            self._logd("connection error. trying to reconnect.")
            # try to reconnect
            self.websocket = connect(
                self.uri, close_timeout=self.close_timeout
            )

        self.websocket.send(message)

        return self.websocket.recv()

    def __del__(self):
        """
        Call close before deleting
        """
        self.websocket.close()
