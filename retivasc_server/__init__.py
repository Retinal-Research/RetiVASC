"""RetiVASC server: the web UI plus full AutoMorph v2 analysis for a lab or clinical intranet.

One process serves the built web UI and a small JSON API. Users sign in; their photos
go into a job queue that runs AutoMorph v2 on this machine's GPU, one job at a time;
results stay on this machine and are deleted after the retention period.
"""

__version__ = "0.1.0"
