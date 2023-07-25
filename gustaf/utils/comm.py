"""
Experimental communication utilities.
WebSocketClient requires `websockets` packages and python 3.7+.
"""
try:
    # this requires python 3.7+
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

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

    def send_recv(self, message, recv_hook=None):
        """
        Send message and return received answer.

        Parameters
        ----------
        message: str
        recv_hook: callable
          Hook that takes recv function as an argument.

        Returns
        -------
        response: Any
          str, unless recv_hook returns otherwise.
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

        if recv_hook is None:
            return self.websocket.recv()
        else:
            return recv_hook(self.websocket.recv)
