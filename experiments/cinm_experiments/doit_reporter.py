"""A doit reporter that renders a single tqdm progress bar instead of one
printed line per task. This repo's dodo.py pipelines easily generate tens of
thousands of leaf tasks (BO search seeds x working groups x functions), where
doit's default ConsoleReporter -- one line per task, printed as it's
selected/skipped/executed -- is unreadable noise.

Usage, in a dodo.py:

    from cinm_experiments.doit_reporter import TqdmReporter
    DOIT_CONFIG = {..., "reporter": TqdmReporter}

Reporter classes can only be selected this way (via DOIT_CONFIG), not by
name on the command line -- `doit -r console` still works to fall back to
the default for a single invocation, see doit's cmd_run.Run._execute.
"""
from __future__ import annotations

from doit.reporter import ConsoleReporter
from tqdm import tqdm


class TqdmReporter(ConsoleReporter):
    """Failure/error reporting is unchanged from ConsoleReporter (same
    replay in complete_run, see below); only per-task progress is
    redirected from a printed line to a tqdm bar.

    Getting a real total (not just an open-ended counter) requires this
    repo's doit fork: doit's runner (runner.py Runner.run_tasks) pulls one
    node at a time from the dependency-graph walk and, for that same node,
    immediately calls get_status followed by its terminal event
    (add_success/skip_uptodate/...) before moving to the next node -- so
    get_status alone can never report a total ahead of completions, only in
    lockstep with them. This fork adds a second hook, update_total(), fired
    from TaskDispatcher._add_task() when a @create_after task-creator
    expands: that happens as one synchronous batch (generate_tasks(...)
    materializes every subtask at once), before the runner starts resolving
    any of them individually -- see doit/control.py and
    doit/reporter.py:ConsoleReporter.update_total upstream of this repo's
    doit checkout. initialize() seeds the total with whatever was already
    known before execution started (plain, non-delayed task creators);
    update_total() grows it as each delayed creator's tasks get discovered."""

    desc = "progress bar (tqdm) instead of one line per task"

    def __init__(self, outstream, options):
        super().__init__(outstream, options)
        self.pbar = tqdm(total=0, unit="task", file=outstream, dynamic_ncols=True)

    @staticmethod
    def _is_tracked(task) -> bool:
        # Group/placeholder tasks (has_subtask, no actions of their own) and
        # private tasks (leading underscore, doit's own convention) aren't
        # real work -- counting them would swamp the bar with thousands of
        # zero-cost nodes.
        return bool(task.actions) and task.name[0] != '_'

    def initialize(self, tasks, selected_tasks):
        # Seed with every eagerly-loaded task already known before execution
        # starts. @create_after subtasks aren't among these yet -- their
        # loader placeholder has no actions, so _is_tracked already excludes
        # it -- those arrive later through update_total.
        self.pbar.total = sum(1 for t in tasks.values() if self._is_tracked(t))
        self.pbar.refresh()

    def update_total(self, new_tasks):
        n = sum(1 for t in new_tasks if self._is_tracked(t))
        if n:
            self.pbar.total += n
            self.pbar.refresh()

    def execute_task(self, task):
        if self._is_tracked(task):
            self.pbar.set_description(task.name[:60], refresh=False)

    def add_failure(self, task, fail):
        super().add_failure(task, fail)
        if self._is_tracked(task):
            self.pbar.update(1)

    def add_success(self, task):
        if self._is_tracked(task):
            self.pbar.update(1)

    def skip_uptodate(self, task):
        # Not real work -- shrink the denominator instead of advancing
        # progress, so a pipeline with many already-done tasks doesn't
        # report a fast rate/ETA from skips and then stall once it reaches
        # the actually-expensive remaining tasks.
        if self._is_tracked(task):
            self.pbar.total -= 1
            self.pbar.refresh()

    def skip_ignore(self, task):
        if self._is_tracked(task):
            tqdm.write("!! %s" % task.title(), file=self.outstream)
            self.pbar.total -= 1
            self.pbar.refresh()

    def _write_failure(self, result, write_exception=True):
        # Same content as ConsoleReporter._write_failure, but through
        # tqdm.write instead of self.write (plain outstream.write) so it
        # doesn't get mangled by the bar's own carriage-return redraw --
        # this fires immediately from add_failure, while the bar is still
        # active, not just from complete_run's after-the-fact replay.
        msg = '%s - taskid:%s' % (result['exception'].get_name(), result['task'].name)
        tqdm.write(msg, file=self.outstream)
        if write_exception:
            tqdm.write(result['exception'].get_msg(), file=self.outstream)

    def complete_run(self):
        self.pbar.close()
        super().complete_run()
