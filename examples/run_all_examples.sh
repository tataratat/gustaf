#!/bin/bash

# run all examples
# iterate over all files with .py extension
for file in *.py
do
    # run the file
    echo "Running $file"
    # run file and check for errors
    python $file
    # check if the last command was successful
    if [ $? -ne 0 ]; then
        echo "Error running $file"
        # push file name into a list
        failed_files="$failed_files $file"
    fi
    sleep 5
    echo "Done"
done
echo "The following examples had errors: $failed_files"
