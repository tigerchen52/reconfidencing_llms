    #!/bin/bash

    Command="sbatch wrapCode.sh"
    Submit_Output="$($Command 2>&1)"
    JobId=`echo $Submit_Output | grep 'Submitted batch job' | awk '{print $4}'`
    echo $JobId

    # --> Sleep here for a few seconds to wait until the job is actually launched
    Host=`scontrol show job $JobId | grep ' NodeList' | awk -F'=' '{print $2}'`
    echo $Host