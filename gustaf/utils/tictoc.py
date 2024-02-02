"""gustaf/gustaf/utils/tictoc.py.

Timer that tics, tocs and logs.
"""

from time import perf_counter as now

from gustaf._base import GustafBase


class Tic(GustafBase):
    """
    Timer class for easier time measurement.
    """

    __slots__ = ("_title", "_names", "_laps", "_logger")

    # add short cut to perf_counter in case you just want pure "now"
    now = now

    def __init__(self, title="untitled", log_level="debug"):
        """
        Configures the timer and logger. Marks the starting point.

        Parameters
        ----------
        title: str
          Title for this measurement. Defaults to "untitled".
        log_level: str
          Valid options are {"info", "debug", "warning"}. Defaults to "debug".
        """
        # select logger based on logger level
        self._logger = eval(f"self._log{str(log_level).lower()[0]}")

        # initialize names
        self._names = []

        # give name
        self._title = str(title)

        # start
        self._laps = [now()]

    def toc(self, name=None, log=False):
        """
        Adds now to the measurements.

        Parameters
        ----------
        name: str
          Name of this specific measurement lap.
          By default, it assigns a number to this lap.
        log: bool
          If True, will log lapsed time. Defaults to None.

        Returns
        -------
        None
        """
        # measure now
        self._laps.append(now())

        self._names.append(
            f"lap-{len(self._laps) - 2}" if name is None else str(name)
        )

        if log:
            self._logger(
                f"{self._title} - {name}:  {self._laps[-2] - self._laps[-1]}"
            )

    def summary(self, log=True, print_=False):
        """
        Prints measurement summary to the log.

        Parameters
        ----------
        log: bool
          Logs the summary. Default is True.
        print_: bool
          Prints the summary. Default is False.


        Returns
        -------
        records: tuple
          Lap timings in a tuple consisting of a list of the lap names and
           times for each lap from the timer start (cumulative time).
        """
        start = self._laps[0]
        cumulative = [f"{lap-start:.10f}" for lap in self._laps[1:]]

        if log or print_:
            message = [f"\n+++ {self._title} - time logs +++\n"]

            name_width = int(max([len(n) for n in self._names]))

            # extract info
            diff = [
                f"{l1 - l0:.10f}"
                for l0, l1 in zip(self._laps[:-1], self._laps[1:])
            ]
            diff_width = int(max([len(d) for d in diff]))
            cumulative_width = int(max([len(c) for c in cumulative]))

            message.append(
                f"{'names'.ljust(name_width)} | "
                f"{'diff'.rjust(diff_width)} | "
                f"{'cumulative'.rjust(cumulative_width)}\n"
            )

            for n, d, c in zip(self._names, diff, cumulative):
                this_line = (
                    f"{n.ljust(name_width)} | "
                    f"{d.rjust(diff_width)} | "
                    f"{c.rjust(cumulative_width)}\n"
                )
                message.append(this_line)

            if log:
                self._logger(*message)
            if print_:
                print(*message)  # noqa: T201

        return self._names.copy(), cumulative
