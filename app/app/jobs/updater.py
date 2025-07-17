from apscheduler.schedulers.background import BackgroundScheduler
from app.jobs import jobs

class Scheduler:

	def __init__(self):
		"""A class for background scheduler."""
		pass
	
	# --------------------------------------------------------------------------------------------------------------- #
	def start(self):
		"""Used to start the tasks."""
		# scheduler = BackgroundScheduler()
		# tasks = jobs.Tasks()
		# scheduler.add_job(tasks.sample_job, 'interval', seconds=5)
		# scheduler.start()