#!/bin/bash

# Run all the shell scripts in the current directory starting with "run_"
for script in run_*.sh; do
	# Skip if the script is the current script itself
	if [[ "$script" == "$(basename "$0")" ]]; then
		continue
	fi
	if [[ -x "$script" ]]; then
		echo "Running $script..."
		# pass all command line arguments to the script
		./"$script" "$@"
	else
		echo "Skipping $script (not executable)"
	fi
done