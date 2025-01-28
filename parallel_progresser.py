import sys
import shutil
import multiprocessing as mp
import equation_functions as ef


progress_dict[id] += 1 / total_prog * 100

# Function to display progress of all workers on individual lines
def progress_report(progress_dict, processes):
    # Get terminal height and width
    terminal_width = shutil.get_terminal_size().columns

    # Print initial lines for all processes
    for name in processes:
        print(f"{name}: 0.00%")

    while(any(progress_dict[name] < 100 for name in processes)):
        # Move the cursor back up to the first process line
        sys.stdout.write(f"\033[{len(processes)}F")

        # Print updated progress for each process
        for name in processes:
            progress_value = progress_dict.get(name, 0)
            progress_str = f"{name}: {progress_value:.2f}%"

            # Truncate progress string if it exceeds terminal width
            if(len(progress_str) > terminal_width):
                progress_str = progress_str[:terminal_width - 3] + "..."

            print(progress_str)

    # Move the cursor down after all processes are completed
    sys.stdout.write(f"\033[{len(processes)}E")
    print("All processes completed.")
