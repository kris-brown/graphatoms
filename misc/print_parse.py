#External modules

################################################################################

"""
Printing and parsing functions

print_time
"""

def print_time(floatHours : float) -> str:
	intHours = int(floatHours)
	return "%02d:%02d" % (intHours,(floatHours-intHours)*60)
